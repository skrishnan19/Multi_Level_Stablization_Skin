from torchvision.models import resnet18, resnet50, resnet34, ResNet50_Weights, ResNet18_Weights, ResNet34_Weights
import torch.nn as nn
import torch
import copy
from torch import Tensor
from functools import partial
from typing import Any, Callable, List, Optional, Type, Union
import torch.nn.functional as F
from Models.Util import *

import torchvision.models as models


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class MLP(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=256, out_dim=256, usenorm=True, usedout=False):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(hidden_dim, out_dim)
        self.usenorm = usenorm
        self.usedout = usedout
        self.dout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.l1(x)
        if self.usenorm:
            x = self.bn(x)
        x = self.relu(x)
        if self.usedout:
            x = self.dout(x)
        x = self.l2(x)
        return x



class MyResNet(nn.Module):
    def __init__(self, arch, pretrain, num_classes, projectDim):
        super(MyResNet, self).__init__()
        self.sm = nn.Softmax(dim=1)
        if arch == 'resnet50':
            if pretrain:
                self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            else:
                self.resnet = resnet50(weights=None)
        elif arch == 'resnet18':
            if pretrain:
                self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        nfea = self.resnet.fc.in_features
        self.encoder = nn.Sequential(*list(self.resnet.children())[:-1])
        self.dout = nn.Dropout(p=0.5)
    
        self.predictor = nn.Sequential(nn.Linear(projectDim, projectDim *2),
                                       nn.BatchNorm1d(projectDim * 2),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(projectDim * 2, projectDim))
    

        self.projector = MLP(nfea, projectDim*2, projectDim)
     
        
        self.fc = MLP(projectDim, projectDim//2, num_classes, usenorm=False, usedout=True)
       
        
    def freezeNet(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreezeNet(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(self, x):
        z = torch.flatten(self.encoder(x), 1)
        z = self.projector(z)

        z_ = F.relu(z)
        logits = self.fc(z_)
        p = self.predictor(z_)
		
        return logits, z, p

    
   
   




