import torch
from Models.MyResNet import *
from Models.DataLoaderSkin import *
import torchutils as tu
from Models.loss import *
import torch.optim as optim
from Models.Util import *
from torch.autograd import Variable
from pytorch_lightning import seed_everything
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from Models.lars import LARS
from Models.EMATeacher import *
from torch_ema import ExponentialMovingAverage
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class SSL(nn.Module):
    def __init__(self, opt):
        super(SSL, self).__init__()
        seed_everything(opt.seed, workers=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        opt = argparse.Namespace(**vars(opt))
        self.opt = opt
        self.loader_L, self.loader_UL, self.loader_Test =  getDataLoaders(self.opt.dataset, self.opt.itr, opt.seed,
                                                                          opt.pL, opt.addValDataWithTrain, self.opt.bs_l, self.opt.bs_u)
        self.labeled_iter = iter(self.loader_L)
        if self.loader_UL is not None:
            self.unlabeled_iter = iter(self.loader_UL)

        y = torch.from_numpy(self.loader_L.dataset.lblArr).cuda(self.opt.gpuid)
        self.uniqueLbls = torch.unique(y)
        self.opt.n_class = len(self.uniqueLbls)

        self.cw_L = None
        if self.opt.use_cw:
            self.cw_L = calWeights_GPU(y, self.uniqueLbls, self.opt.gpuid)
        self.cw_U = None
        self.criterion_class = CELoss(self.uniqueLbls, self.opt.gpuid)
        self.criterion_consist = SimSiam()

        self.sm = nn.Softmax(dim=1)
        self.trainingPara = getTrainingPara(self.opt.dataset, self.opt.type)

        self.student = self.loadModel(self.opt.projectionDim)
        self.teacher = EMATeacher(self.student, self.opt.ema_model, self.trainingPara.total_iterations)
        self.setOptimizer(self.opt.lr, self.trainingPara.total_iterations, nwarmup=300)

        self.probArr_UN_EMA = None
        self.feaArr_EMA = None
        if self.loader_UL is not None:
            nUL = len(self.loader_UL.dataset.lblArr)
            nL = len(self.loader_L.dataset.lblArr)
            self.nL = nL
            self.nUL = nUL

            self.probArr_UN_EMA = torch.ones(nUL, self.opt.n_class) / self.opt.n_class
            self.probArr_UN_EMA = self.probArr_UN_EMA.cuda(self.opt.gpuid)

            self.feaArr_EMA = torch.zeros(nUL+nL, self.opt.projectionDim).cuda(self.opt.gpuid)
            self.classPrototypes = torch.zeros(self.opt.n_class, self.opt.projectionDim)
            self.classPrototypes = self.classPrototypes.cuda(self.opt.gpuid)
            self.initFeaArrAndPrototypes()

    def initFeaArrAndPrototypes(self):
        y_all = []
        idx_all = []
        for i, (I, _, _, y, idx) in enumerate(self.loader_L):
            _, z, _ = self.teacher(I.cuda(self.opt.gpuid))
            self.feaArr_EMA[idx,:] = z
            y_all.extend(y)
            idx_all.extend(idx)
        y_all = torch.stack(y_all, 0)
        idx_all = torch.stack(idx_all, 0)
        for c in range(self.opt.n_class):
            tmpidx = idx_all[y_all == c]
            self.classPrototypes[c,:] = self.feaArr_EMA[tmpidx,:].mean(0)

        for i, (I, _, _, _, idx) in enumerate(self.loader_UL):
            _, z, _ = self.teacher(I.cuda(self.opt.gpuid))
            idx = idx + self.nL
            self.feaArr_EMA[idx,:] = z

    def getCurrentDecay(self, final_decay):
        base_momentum = 0
        lambda_rate = 10
        current_step = self.teacher.get_num_updates()
        progress = current_step / self.trainingPara.total_iterations
        momentum = final_decay - (final_decay - base_momentum) * math.exp(-lambda_rate * progress)
        return momentum

    def updateFeaRunningAvg(self, idx, z):
        m = self.getCurrentDecay(self.opt.ema_fea)
        self.feaArr_EMA[idx,:] = m * self.feaArr_EMA[idx,:] + (1 - m) * z.detach()
        
    def updateProbRunningAvg(self, idx, logits):
        m = self.getCurrentDecay(self.opt.ema_pl)
        if self.probArr_UN_EMA is not None:
            self.probArr_UN_EMA[idx,:] = m * self.probArr_UN_EMA[idx,:] + (1 - m) * self.sm(logits.detach())

    def setOptimizer(self, lr, total_iterations, nwarmup):
        if self.opt.optimizer == 'ADAM':
            self.optimizer = optim.Adam(self.student.parameters(), lr=lr, weight_decay=5e-4) #, betas=(0.95,0.999)
        elif self.opt.optimizer == 'SGD':
            self.optimizer = optim.SGD(self.student.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
        elif self.opt.optimizer == 'LARS':
            self.optimizer = LARS(self.student.parameters(), lr=lr)
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=nwarmup,
                                                         num_training_steps=total_iterations)


    def loadModel(self, projectDim):
        if self.opt.modelName == 'resnet50':
            net = MyResNet('resnet50', self.opt.pretrain, self.opt.n_class, projectDim)
        elif self.opt.modelName == 'resnet34':
            net = MyResNet('resnet34', self.opt.pretrain, self.opt.n_class, projectDim)

        pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print('Total trainable parameters = ', (pytorch_total_params // 1000000))
        return net.cuda(self.opt.gpuid)


    def getNextBatch_L(self):
        try:
            Iw, Is_1, Is_2, lbls, idx = next(self.labeled_iter)
            if len(lbls) <= 1:
                self.labeled_iter = iter(self.loader_L)
                Iw, Is_1, Is_2, lbls, idx = next(self.labeled_iter)
        except StopIteration:
            self.labeled_iter = iter(self.loader_L)
            Iw, Is_1, Is_2, lbls, idx = next(self.labeled_iter)
        lbls = Variable(lbls.cuda(self.opt.gpuid))
        Iw = Variable(Iw.cuda(self.opt.gpuid), requires_grad=False)
        Is_1 = Variable(Is_1.cuda(self.opt.gpuid), requires_grad=False)
        Is_2 = Variable(Is_2.cuda(self.opt.gpuid), requires_grad=False)
        return Iw, Is_1, Is_2, lbls, idx

    def getNextBatch_UL(self):
        try:
            Iw, Is_1, Is_2, lbls, idx = next(self.unlabeled_iter)
            if len(lbls) <= 1:
                self.unlabeled_iter = iter(self.loader_UL)
                Iw, Is_1, Is_2, lbls, idx = next(self.unlabeled_iter)
        except StopIteration:
            self.unlabeled_iter = iter(self.loader_UL)
            Iw, Is_1, Is_2, lbls, idx = next(self.unlabeled_iter)
        lbls = Variable(lbls.cuda(self.opt.gpuid))
        Iw = Variable(Iw.cuda(self.opt.gpuid), requires_grad=False)
        Is_1 = Variable(Is_1.cuda(self.opt.gpuid), requires_grad=False)
        Is_2 = Variable(Is_2.cuda(self.opt.gpuid), requires_grad=False)
        return Iw, Is_1, Is_2, lbls, idx

    def updatePrototypes(self, x, y):
        x = x.detach()
        m = self.opt.ema_fea #self.getCurrentDecay(self.opt.ema_fea)
        for c in range(self.opt.n_class):
            idx = y == c
            if idx.sum() > 0:
                mv = x[y == c, :].mean(dim=0)
                self.classPrototypes[c, :] = m * self.classPrototypes[c, :] + (1 - m) * mv

    def train_sup(self, epoch):
        loss_tot, logits_all, gt_all = 0, [], []

        self.student.train()
        for i in range(self.trainingPara.iterPerEpoch):
            _, Is, _, lbls, idx_l = self.getNextBatch_L()

            Is = Variable(Is.cuda(self.opt.gpuid), requires_grad=False)
            lbls = Variable(lbls.cuda(self.opt.gpuid), requires_grad=False)

            logits, _, _ = self.student(Is)
            loss =  self.criterion_class(logits, lbls, self.cw_L)

            self.optimizer.zero_grad()
            loss.backward()
            loss_tot += loss.item()
            self.optimizer.step()
            self.scheduler.step()
            self.teacher.update()

            gt_all.extend(lbls)
            logits_all.extend(logits.data.detach())

        gt_all = torch.stack(gt_all, 0)
        logits_all = torch.stack(logits_all, 0)

        reClass, desc = getScores2(gt_all, logits_all)
        return loss_tot, reClass, desc

    def test_oneImg(self, I):
        with torch.no_grad():
            logits, z, p = self.teacher(I)
            if self.opt.w_proto > 0:
                sim = get_fea_proto_sim(p, self.classPrototypes) / self.opt.temperature
            else:
                sim = get_fea_proto_sim(z, self.classPrototypes) / self.opt.temperature
        return logits, sim, z, p

    def test(self):
        self.teacher.eval()
        gt_all, logits_all, logits_all_proto = [], [], []
        for i, (I, lbls) in enumerate(self.loader_Test):
            I = Variable(I.cuda(self.opt.gpuid))
            lbls = Variable(lbls.cuda(self.opt.gpuid))
            logits, logits_proto, _, _ = self.test_oneImg(I)
            logits_all.extend(logits)
            logits_all_proto.extend(logits_proto)
            gt_all.extend(lbls)
        gt_all = torch.stack(gt_all, 0)
        logits_all = torch.stack(logits_all, 0)
        logits_all_proto = torch.stack(logits_all_proto, 0)
        # reClass, desc = getScores2(gt_all, logits_all)
        reClass, desc1 = getScores_new2(gt_all, logits_all)
        reClass_proto, desc = getScores2(gt_all, logits_all_proto)
        reClass_com, desc = getScores2(gt_all, (logits_all + logits_all_proto)/2)


        y_true = gt_all.detach().cpu().numpy()

        y_pred_main = torch.argmax(logits_all, dim=1).detach().cpu().numpy()
        y_pred_proto = torch.argmax(logits_all_proto, dim=1).detach().cpu().numpy()
        y_pred_com = torch.argmax((logits_all + logits_all_proto) / 2, dim=1).detach().cpu().numpy()


        return reClass, reClass_proto, reClass_com, desc1, y_true, y_pred_main, y_pred_proto, y_pred_com
       


    def getLblsAndMask(self, prob):
        prob, pred = torch.max(prob, dim=1)
        mask = prob.ge(self.opt.thr)
        pred = Variable(pred, requires_grad=False)
        mask = Variable(mask, requires_grad=False)
        return pred, mask.float(), prob
    
    def trainSemiSup_TS(self, epoch):
        loss_tot, simloss, w = 0, 0, 0
        Y_l_all, Y_u_all, Y_u_p_all, m_all = [], [], [], []
        self.student.train()
        for i in range(self.trainingPara.iterPerEpoch):
            Il_w, Il_s1, _, y_l, idx_l = self.getNextBatch_L()
            Iu_w, Iu_s1, _, y_u, idx_un = self.getNextBatch_UL()
            idx_l = self.nUL + idx_un
            bs_l, bs_u = Il_w.shape[0], Iu_w.shape[0]

            loss = 0

            # Teacher get mask
            lo, _, fea_z, _ = self.test_oneImg(torch.cat((Il_w, Iu_w, Iu_s1)))
            fl = fea_z[:bs_l,]
            fuw, fus = fea_z[bs_l:,].chunk(2)
            self.updatePrototypes(fl, y_l)
            self.updateFeaRunningAvg(idx_un, fuw)
            self.updateProbRunningAvg(idx_un, lo[bs_l:bs_l+bs_u, :])
            pseudo_label, mask, _ = self.getLblsAndMask(self.probArr_UN_EMA[idx_un, :])
            del lo, fea_z, fl, fuw, fus

            # student
            logits, _, p_student = self.student(torch.cat((Il_s1, Iu_s1)))
            if self.opt.w_class > 0:
                loss = self.criterion_class(logits[:bs_l, :], y_l, self.cw_L)

            if self.opt.w_pl > 0 and mask.sum() > 0:
                loss += self.criterion_class(logits[bs_l:, :], pseudo_label, self.cw_U, mask)


            #with rampup
            if epoch > 5 and self.opt.w_consist > 0:
                p_student_un = p_student[bs_l:, ]
               
                rampup_len = 30
                w_consist_adapt = self.opt.w_consist * np.exp(-5 * (1 - min(epoch / rampup_len, 1))**2)

                ss = self.criterion_consist(p_student_un, self.feaArr_EMA[idx_un, :]).mean()
                simloss += ss.data.item()
                if w_consist_adapt > 0:
                    loss += w_consist_adapt * ss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.teacher.update()

            loss_tot += loss.item()
            Y_l_all.extend(y_l)
            Y_u_all.extend(y_u)
            Y_u_p_all.extend(pseudo_label)
            m_all.extend(mask)

        Y_l_all = torch.stack(Y_l_all, 0)
        Y_u_all = torch.stack(Y_u_all, 0)

        accPseudoLbls, bacc, ns = 0, 0, 0
        if self.opt.w_pl > 0:
            Y_u_p_all = torch.stack(Y_u_p_all, 0)
            m_all = torch.stack(m_all, 0)

            if torch.sum(m_all) > 0:
                idx = m_all == 1
                selected_gt = Y_u_all[idx].view(-1, 1).data.cpu().numpy()
                selected_pl = Y_u_p_all[idx].view(-1, 1).data.cpu().numpy()
                accPseudoLbls = accuracy_score(selected_gt, selected_pl)
                bacc = balanced_accuracy_score(selected_gt, selected_pl)
                if self.opt.use_cw:
                    self.cw_U = calWeights_GPU(Y_u_p_all[idx], self.uniqueLbls, self.opt.gpuid)
                ns = torch.sum(m_all).cpu().numpy()
        return loss_tot, simloss, ns, accPseudoLbls, bacc, w
    
    def printStatPL(self, y_all, y_s, pl_s):
        unique_lbls = torch.unique(y_all)
        print('lbl \t tot \t selected \t acc')
        for lbl in unique_lbls:
            idx = pl_s == lbl
            if idx.float().sum() > 0:
                tot = torch.sum(y_all == lbl).data.item()
                selected = torch.sum(idx).data.item()
                acc = accuracy_score(y_s[idx].data.cpu().numpy(), pl_s[idx].data.cpu().numpy()) * 100
                print('%1d\t%5d\t%5d\t%2.2f%%' % (lbl, tot, selected, acc))
        acc = accuracy_score(y_s.data.cpu().numpy(), pl_s.data.cpu().numpy()) * 100
        print('\t%5d\t%5d\t%2.2f%%' % (len(y_all), len(y_s), acc))
        print()

    def iterate(self):
        print('n_epochs: ', self.trainingPara.n_epochs)
        print('iterPerEpoch: ', self.trainingPara.iterPerEpoch)
        best_bacc = -1.0
        class_names = ["MEL","NV","BCC","AKIEC","BKL","DF","VASC"]
        nsupEpochs = 0
        simloss = 0
        w = 0
        results = []
        for epoch in range(self.trainingPara.n_epochs):
            accPseudoLbls, bacc, n_un, accTr, thr = 0,0,0,0,self.opt.thr
            lr = tu.get_lr(self.optimizer)

            if self.opt.type == 'FS' or epoch < nsupEpochs:
                lossTr, re_tr, desc = self.train_sup(epoch)
                accTr = re_tr[0]
            else:
                lossTr, simloss, n_un, accPseudoLbls, bacc, w = self.trainSemiSup_TS(epoch)
           
            reClass, reClass_proto, reClass_com, desc, y_true, y_pred_main, y_pred_proto, y_pred_com = self.test()

            print('%s %3d %.6f  |' %(self.opt.type, epoch, lr), end='')
            print('%2.4f'% w, end='|')
            print(' %.4f %5d %3.2f%% %3.2f%%\t| %5.3f %5.3f %3.2f%%\t||'
                  % (thr, n_un, accPseudoLbls*100, bacc*100, lossTr, simloss, accTr), end='')
            print(' %3.2f  %3.2f %3.2f  %3.2f\t'
                  % (reClass[0], reClass[1], reClass[2], reClass[3]), end='|')
            print(' %3.2f  %3.2f  %3.2f\t'
                  % (reClass_proto[0], reClass_proto[1], reClass_proto[2]), end='|')
            print(' %3.2f  %3.2f  %3.2f\t'
                  % (reClass_com[0], reClass_com[1], reClass_com[2]), end='|')
            if self.probArr_UN_EMA is not None:
                print('%.3f' % self.probArr_UN_EMA.max().data.cpu().numpy(), end=': ')
                

            val_bacc = float(reClass[0])  # adjust index if needed!

            # if val_bacc > best_bacc:
            #     best_bacc = val_bacc
            #     print(f"\n New best epoch {epoch} | Best BAcc = {best_bacc:.2f}")

            #     # Save CM for the combined prediction (often best)
            #     # self.save_cm(
            #     #     y_true=y_true,
            #     #     y_pred=y_pred_com,
            #     #     class_names=class_names,
            #     #     epoch=epoch,
            #     #     out_dir="results/confusion_matrices",
            #     #     tag="test_best"
            #     # )
            print()
            reClass.extend(reClass_proto)
            results.append(reClass)

        print()
        remean = np.mean(results[-5:], axis=0)
        for d in remean:
            print(' %3.2f  '% (d), end='\t')
        print()
        return remean, desc
    def save_cm(self, y_true, y_pred, class_names, epoch, out_dir, tag="test"):
        os.makedirs(out_dir, exist_ok=True)
        labels = np.arange(len(class_names))

        # Counts
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
        disp.plot(values_format="d", xticks_rotation=45)
        plt.title(f"CM Counts [{tag}] - Epoch {epoch}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"cm_counts_{tag}_epoch_{epoch}.png"), dpi=300)
        plt.close()

        # Normalized (row-wise recall)
        cmn = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
        disp = ConfusionMatrixDisplay(cmn, display_labels=class_names)
        disp.plot(values_format=".2f", xticks_rotation=45)
        plt.title(f"CM Normalized [{tag}] - Epoch {epoch}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"cm_norm_{tag}_epoch_{epoch}.png"), dpi=300)
        plt.close()

        # Print per-class recall
        recalls = np.diag(cmn)
        print(f"\n[{tag}] Epoch {epoch} per-class Recall:")
        for c, r in zip(class_names, recalls):
            print(f"  {c:5s}: {r*100:.2f}%")