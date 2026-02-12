import torch
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import f1_score, precision_score, balanced_accuracy_score, precision_recall_fscore_support
from imblearn.metrics import sensitivity_score, specificity_score
import numpy as np
import argparse
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR
import math

N_CLASSES = 7
CLASS_NAMES = [ 'Melanoma', 'Melanocytic nevus', 'Basal cell carcinoma', 'Actinic keratosis', 'Benign keratosis', 'Dermatofibroma', 'Vascular lesion']

def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def printw(w):
    for v in w:
        print('%0.5f'%(v), end = ', ')
    print()


def calWeights_GPU(y, unique_lbls, gpuid):
    weights = torch.zeros(len(unique_lbls)).cuda(gpuid)
    for lbl in unique_lbls:
        if lbl != -1:
            tmp = torch.sum((y == lbl).float())
            if tmp == 0: tmp = 1
            weights[lbl] = 1.0 / tmp
    weights = weights / torch.sum(weights)
    return Variable(weights).view(-1)


def getTrainingPara(dataset, type):
    total_iterations = 4000
    iterPerEpoch = total_iterations // 30
    n_epochs = total_iterations // iterPerEpoch

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=n_epochs)
    parser.add_argument("--iterPerEpoch", type=int, default=iterPerEpoch)
    parser.add_argument("--total_iterations", type=int, default=total_iterations)
    trainingPara = parser.parse_args()
    return trainingPara


def getThr(epoch, T, st_epoch=25, ed_epoch=75):
    a = 1
    if epoch < st_epoch:
        return 1
    elif epoch < ed_epoch:
        thr = (a-T)*(ed_epoch - epoch)/(ed_epoch-st_epoch) + T
        return thr
    else: return T



def getScores_new2(gt_all, logits):
    # probs for prediction + AUROC
    probs = F.softmax(logits, dim=1)
    # pred_all = probs.argmax(dim=1)
    _, pred_all = torch.max(probs, dim=1)
    
    gt_np = gt_all.detach().cpu().numpy()
    pred_np = pred_all.detach().cpu().numpy()

    perClassAcc, AUROC_avg, Accus_avg, Senss_avg, Specs_avg, F1_avg = \
        compute_metrics_test2(gt_np, pred_np, probs.detach().cpu().numpy())

    bacc = balanced_accuracy_score(gt_np, pred_np)
    acc = accuracy_score(gt_np, pred_np)
    f1_macro = f1_score(gt_np, pred_np, average='macro', zero_division=0)

    retVals = [bacc * 100, acc * 100, Accus_avg * 100, f1_macro * 100]
    desc = ['bacc', 'acc', 'acc_old', 'f1_macro']
    return retVals, desc


def compute_metrics_test2(gt, pred, probs):
    """
    gt: numpy array shape [N] with class ids
    pred: numpy array shape [N] with predicted class ids
    probs: numpy array shape [N, C] with probabilities (recommended for AUROC)
    """
    C = probs.shape[1]

    AUROCs, Accus, Senss, Specs, Pre, F1s = [], [], [], [], [], []

    for i in range(C):
        gt_bin = (gt == i)
        pred_bin = (pred == i)

        # AUROC needs both classes present in gt_bin
        try:
            AUROCs.append(roc_auc_score(gt_bin, probs[:, i]))
        except ValueError:
            AUROCs.append(0.0)

        Accus.append(accuracy_score(gt_bin, pred_bin))

        # If you use your own specificity_score / sensitivity_score, keep them.
        # Otherwise, here are simple safe versions:
        tn = np.sum((gt_bin == 0) & (pred_bin == 0))
        fp = np.sum((gt_bin == 0) & (pred_bin == 1))
        fn = np.sum((gt_bin == 1) & (pred_bin == 0))
        tp = np.sum((gt_bin == 1) & (pred_bin == 1))

        Specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        Senss.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)

        Pre.append(precision_score(gt_bin, pred_bin, zero_division=0))
        F1s.append(f1_score(gt_bin, pred_bin, zero_division=0))

    AUROC_avg = float(np.mean(AUROCs))
    Accus_avg = float(np.mean(Accus))
    Senss_avg = float(np.mean(Senss))
    Specs_avg = float(np.mean(Specs))
    F1_avg = float(np.mean(F1s))

    return Accus, AUROC_avg, Accus_avg, Senss_avg, Specs_avg, F1_avg



def compute_metrics_test(gt, pred, logits):
    AUROCs, Accus, Senss, Specs, Pre, F1 = [], [], [], [], [], []
    logits = logits.cpu().detach().numpy()
    CLASS_NAMES = ['Melanoma', 'Melanocytic nevus', 'Basal cell carcinoma', 'Actinic keratosis', 'Benign keratosis',
                   'Dermatofibroma', 'Vascular lesion']
    indexes = range(len(CLASS_NAMES))
    # pred_np = pred.cpu().detach().numpy()
    pred_np = pred
    for i, cls in enumerate(CLASS_NAMES):
        # gt_np = (gt == i).long().cpu().detach().numpy()
        # pred_np = (pred == i).long().cpu().detach().numpy()
        gt_np = (gt == i)
        pred_np = (pred == i)
        try:
            AUROCs.append(roc_auc_score(gt_np, logits[:, i]))
        except ValueError:
            AUROCs.append(0)

        try:
            Accus.append(accuracy_score(gt_np, pred_np))
        except ValueError:
            Accus.append(0)

        # tn, fp, fn, tp = confusion_matrix(gt_np, pred_np).ravel()
        try:
            # Specs.append( tn / (tn + fp))
            Specs.append(specificity_score(gt_np, pred_np))
        except ValueError:
            Specs.append(0)

        try:
            # Senss.append(tp / (tp + fn))
            Senss.append(sensitivity_score(gt_np, pred_np))
        except ValueError:
            Senss.append(0)

        try:
            Pre.append(precision_score(gt_np, pred_np, zero_division=0))
        except ValueError:
            Pre.append(0)

        try:
            F1.append(f1_score(gt_np, pred_np, zero_division=0))
        except ValueError:
            F1.append(0)

    AUROC_avg = np.array(AUROCs).mean()
    Accus_avg = np.array(Accus).mean()
    Senss_avg = np.array(Senss).mean()
    Specs_avg = np.array(Specs).mean()
    pre_avg = np.array(Pre).mean()
    F1_avg = np.array(F1).mean()
    return Accus, AUROC_avg, Accus_avg, Senss_avg, Specs_avg, F1_avg

def getScores_(predLbls_cpu, gt_all_cpu):
    bacc = balanced_accuracy_score(gt_all_cpu, predLbls_cpu)
    acc = accuracy_score(gt_all_cpu, predLbls_cpu)
    f1_macro = f1_score(gt_all_cpu, predLbls_cpu, average='macro')
    return bacc*100, acc*100, f1_macro*100

def getScores2(gt_all, lo_class_w):
    prob_class = F.softmax(lo_class_w, dim=1)
    _, pred_class = torch.max(prob_class, dim=1)
    reClass_w, desc = getScores(gt_all, pred_class)
    return reClass_w, desc

def get_fea_proto_sim(fea, proto):
    fea = F.normalize(fea, dim=1, eps=1e-8)
    proto = F.normalize(proto.detach(), dim=1, eps=1e-8)
    sim = torch.matmul(fea, proto.T)
    return sim


def getScores(gt_all, pred_labels):
    # _, pred_labels = torch.max(logits, dim=1)

    gt_all_cpu = gt_all.detach().cpu().numpy()
    pred_labels_cpu = pred_labels.detach().cpu().numpy()

    bacc = balanced_accuracy_score(gt_all_cpu, pred_labels_cpu)
    acc = accuracy_score(gt_all_cpu, pred_labels_cpu)
    f1_macro = f1_score(gt_all_cpu, pred_labels_cpu, average='macro')

    retVals = [bacc * 100, acc * 100, f1_macro * 100]
    desc = ['balanced_acc', 'acc', 'f1_macro']

    return retVals, desc


def getScores_new(gt_all, logits):
    # _, predLbls_gpu = torch.max(prob_all, dim=1)
    prob_class = F.softmax(logits, dim=1)
    _, pred_all = torch.max(prob_class, dim=1)

    gt_all_cpu, predLbls_cpu = gt_all.data.cpu().numpy(), pred_all.data.cpu().numpy()
    perCalssAcc, AUROC_avg, Accus_avg, Senss_avg, Specs_avg, F1_avg = \
        compute_metrics_test(gt_all_cpu,predLbls_cpu,logits)
    
    bacc = balanced_accuracy_score(gt_all_cpu, predLbls_cpu)
    acc = accuracy_score(gt_all_cpu, predLbls_cpu)
    f1_macro = f1_score(gt_all_cpu, predLbls_cpu, average='macro')

    retVals = [bacc* 100, acc* 100, Accus_avg* 100, f1_macro* 100]
    desc = ['baacc', 'acc', 'acc_old', 'f1_macro']
    return retVals, desc

def compute_metrics_test_new(gt, pred, competition=True):
    """
    Computes accuracy, precision, recall and F1-score from prediction scores.
    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.
        competition: whether to use competition tasks. If False, 
          use all tasks
    Returns:
        List of AUROCs of all classes.
    """

    AUROCs, Accus, Senss, Specs, Pre, F1 = [], [], [], [], [], []
    # gt_np = gt.cpu().detach().numpy()
    # if cfg.uncertainty == 'U-Zeros':
    #     gt_np[np.where(gt_np==-1)] = 0
    # if cfg.uncertainty == 'U-Ones':
    #     gt_np[np.where(gt_np==-1)] = 1
    # pred_np = pred.cpu().detach().numpy()
    gt_np = gt
    pred_np = pred
    THRESH = 0.18
    #     indexes = TARGET_INDEXES if competition else range(N_CLASSES)
    #indexes = range(n_classes)
    
#     pdb.set_trace()
    indexes = range(len(CLASS_NAMES))
    
    for i, cls in enumerate(indexes):
        try:
            AUROCs.append(roc_auc_score(gt_np[i], pred_np[i]))
        except ValueError as error:
            print('Error in computing accuracy for {}.\n Error msg:{}'.format(i, error))
            AUROCs.append(0)
        
        try:
            Accus.append(accuracy_score(gt_np[i], (pred_np[i]>=THRESH)))
        except ValueError as error:
            print('Error in computing accuracy for {}.\n Error msg:{}'.format(i, error))
            Accus.append(0)
        
        try:
            Senss.append(sensitivity_score(gt_np[i], (pred_np[i]>=THRESH)))
        except ValueError:
            print('Error in computing precision for {}.'.format(i))
            Senss.append(0)
        

        try:
            Specs.append(specificity_score(gt_np[i], (pred_np[i]>=THRESH)))
        except ValueError:
            print('Error in computing F1-score for {}.'.format(i))
            Specs.append(0)

        try:
            Pre.append(precision_score(gt_np[i], (pred_np[i]>=THRESH)))
        except ValueError:
            print('Error in computing F1-score for {}.'.format(i))
            Pre.append(0)
    
        try:
            F1.append(f1_score(gt_np[i], (pred_np[i]>=THRESH)))
        except ValueError:
            print('Error in computing F1-score for {}.'.format(i))
            F1.append(0)
    AUROC_avg = np.array(AUROCs).mean()
    Accus_avg = np.array(Accus).mean()
    Senss_avg = np.array(Senss).mean()
    Specs_avg = np.array(Specs).mean()
    pre_avg = np.array(Pre).mean()
    F1_avg = np.array(F1).mean()
    return Accus, AUROC_avg, Accus_avg, Senss_avg, Specs_avg, F1_avg