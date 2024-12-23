import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones.hacnn import HACNN

class HACNNBuilder(nn.Module):
    def __init__(self, cfg):
        super(HACNNBuilder, self).__init__()
        self.cfg = cfg
        self.num_classes = cfg.DATASETS.NUMBER_OF_CLASSES
        self.feature_dim = cfg.SOLVER.FEATURE_DIMENSION

        self.base = HACNN(num_classes=self.num_classes, feat_dim=self.feature_dim)
    
    def forward(self, x):
        return self.base(x)