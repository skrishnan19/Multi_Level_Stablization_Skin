import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from Models.Util import *
import torch.nn.functional as F
from torch import Tensor

class CELoss(nn.Module):
    def __init__(self, unique_lbls, gpuid):
        super(CELoss, self).__init__()
        self.sm = nn.Softmax(dim=1)
        self.unique_lbls = unique_lbls
        self.gpuid = gpuid

    def forward(self, logits, targets, cw=None, mask = None):
        loss = F.cross_entropy(logits, targets, reduction='none').view(-1)
        if mask is None:
            mask = torch.ones_like(loss).float()
        if cw is None:
            cw = torch.ones_like(self.unique_lbls).float()
        cw = cw[targets]

        assert cw.shape == mask.shape
        cw = cw * mask

        assert loss.shape == cw.shape
        loss = loss * cw
        loss = loss.sum() / cw.sum()
        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, T=0.1):
        super(ContrastiveLoss, self).__init__()
        self.T = T
        self.sm = nn.Softmax(dim=1)

    def __getSim(self, x, prototypes):
        x = F.normalize(x, dim=1)
        prototypes = F.normalize(prototypes, dim=1)
        sim = torch.matmul(x, prototypes.T) / self.T
        return sim

    def getProb(self, x, prototypes):
        sim = self.__getSim(x, prototypes)
        return self.sm(sim)

    def forward(self, x, y, prototypes, cw=None, mask = None):
        sim = self.__getSim(x, prototypes)
        prob = self.sm(sim)
        prob = torch.gather(prob, dim=1, index=y.view(-1,1))
        loss = -torch.log(prob.clamp(1e-10)).view(-1)
        if cw is not None:
            cw = cw[y]
            assert loss.shape == cw.shape
            loss = loss * cw
        if mask is not None:
            loss = loss[mask==1]
        return self.T * loss.mean()




class SimSiam(nn.Module):
    def __init__(self):
        super(SimSiam, self).__init__()

    def D(self, p, z):
        sim = F.cosine_similarity(p, z.detach(), dim=-1)
        return 2 - 2*sim #.mean()

    def forward(self, fea_s, fea_t):
        return self.D(fea_s, fea_t)


class KL_Loss(nn.Module):
    def __init__(self, temperature=0.1):
        super(KL_Loss, self).__init__()
        self.T = temperature
        self.sm = nn.Softmax(dim=1)

    def forward(self, student_logits, teacher_logits):
        eps = 1e-10
        soft_targets = self.sm(teacher_logits / self.T)
        soft_targets = torch.clamp(soft_targets, eps, 1 - eps)

        soft_prob = self.sm(student_logits / self.T)
        soft_prob = torch.clamp(soft_prob, eps, 1 - eps)

        loss = soft_targets * torch.log(soft_prob)
        loss = -torch.sum(loss, dim=1)
        loss = (self.T ** 2) * loss.mean()
        return loss


class BalSCL(nn.Module):
    def __init__(self, gpuid, moco_K, temperature, nclasses, cw):
        super(BalSCL, self).__init__()
        self.temperature = temperature
        self.gpuid = gpuid
        self.nclasses = nclasses
        self.moco_K = moco_K
        self.cw = cw
        self.sm = nn.Softmax(dim=1)

    def __getSim(self, x, x_moco):
        x = F.normalize(x, dim=1)
        x_moco = F.normalize(x_moco, dim=1)

        sim = x.mm(x_moco.T)
        sim = torch.div(sim, self.temperature)

        # # For numerical stability
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim = sim - sim_max.detach()
        return sim



    # def getProb_inst(self, x, x_moco, y_moco):
    #     sim = self.__getSim(x, x_moco)
    #     bs = x.shape[0]
    #     y_moco = y_moco.view(-1, 1)
    #
    #     sim_class = torch.zeros(bs, self.nclasses).cuda(self.gpuid)
    #     for c in range(self.nclasses):
    #         idx = (y_moco == c).view(-1)
    #         if idx.sum() > 0:
    #             sim_class[:, c] = sim[:, idx].sum(1) / idx.sum()
    #
    #     prob = self.sm(sim_class)
    #     return prob



    # def getProb(self, x, x_moco, y_moco):
    #     sim = self.__getSim(x, x_moco)
    #     exp_sim = self.sm(sim)
    #     bs = x.shape[0]
    #     y_moco = y_moco.view(-1, 1)
    #
    #     sim_class = torch.zeros(bs, self.nclasses).cuda(self.gpuid)
    #     for c in range(self.nclasses):
    #         idx = (y_moco == c).view(-1)
    #         if idx.sum() > 0:
    #             sim_class[:, c] = exp_sim[:, idx].sum(1) / idx.sum()
    #
    #     prob = sim_class / torch.sum(sim_class, keepdim=True, dim=1).clamp(1e-10)
    #     return sim_class

    def getSimClass(self, x, x_moco, y_moco):
        sim = self.__getSim(x, x_moco)
        exp_sim = torch.exp(sim)

        bs = x.shape[0]
        y_moco = y_moco.view(-1, 1)

        sim_class = torch.zeros(bs, self.nclasses).cuda(self.gpuid)
        for c in range(self.nclasses):
            idx = (y_moco == c).view(-1)
            if idx.sum() > 0:
                sim_class[:, c] = exp_sim[:, idx].sum(1) / idx.sum()
        return sim, sim_class

    def getProb(self, x, x_moco, y_moco):
        _, sim_class = self.getSimClass(x, x_moco, y_moco)
        prob = sim_class / torch.sum(sim_class, keepdim=True, dim=1).clamp(1e-10)
        return prob

    def forward(self, x, y, x_moco, y_moco):
        sim, sim_class = self.getSimClass(x, x_moco, y_moco)
        prob = sim_class / torch.sum(sim_class, keepdim=True, dim=1).clamp(1e-10)
        loss = sim - torch.log(sim_class.sum(1).clamp(1e-10)).view(-1,1).repeat(1, sim.shape[1])

        y = y.view(-1, 1)
        y_moco = y_moco.view(-1, 1)
        mask_p = torch.eq(y, y_moco.T).float().cuda(self.gpuid)
        sv = mask_p.sum(1).clamp(1e-10)
        loss = (loss * mask_p).sum(1)/sv
        if self.cw is not None:
            cw = self.cw[y]
            loss = loss.view(-1,1)
            assert cw.shape == loss.shape
            loss = loss * cw
            loss /= cw.sum()
        loss = -loss.mean()
        return loss #, prob

    def forward_CE(self, x, y, x_moco, y_moco):
        y = y.view(-1,1)
        sim, sim_class = self.getSimClass(x, x_moco, y_moco)
        prob = sim_class / torch.sum(sim_class, keepdim=True, dim=1).clamp(1e-10)
        p = torch.gather(prob, dim=1, index=y)
        loss = -torch.log(p.clamp(1e-10)).view(-1,1)
        if self.cw is not None:
            cw = self.cw[y]
            assert cw.shape == loss.shape
            loss = loss * cw
        loss = loss.mean()
        return loss, prob.clone().detach()



