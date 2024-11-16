"""
Rethinking the Distribution Gap of Person Re-identification with Camera-based Batch Normalization

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones.resnet_backbone import ResNet_Backbone
from utils.weight_utils import weights_init_kaiming, weights_init_classifier

class ResNetBuilder(nn.Module):
    in_planes = 2048

    def __init__(self, cfg):
        super().__init__()
        last_stride=1
        feature_dim = cfg.SOLVER.FEATURE_DIMENSION
        self.num_pids = cfg.DATASETS.NUMBER_OF_CLASSES
        self.base = ResNet_Backbone(last_stride)
        # model_path = cfg.MODEL.PRETRAIN_PATH
        self.base.load_param(cfg.MODEL.PRETRAIN_PATH)
        bn_neck = nn.BatchNorm1d(feature_dim, momentum=None)
        bn_neck.bias.requires_grad_(False)
        self.bottleneck = nn.Sequential(bn_neck)
        self.bottleneck.apply(weights_init_kaiming)
        if self.num_pids is not None:
            self.classifier = nn.Linear(feature_dim, self.num_pids, bias=False)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        feat_before_bn = self.base(x)
        feat_before_bn = F.avg_pool2d(feat_before_bn, feat_before_bn.shape[2:])
        feat_before_bn = feat_before_bn.view(feat_before_bn.shape[0], -1)
        feat_after_bn = self.bottleneck(feat_before_bn)
        if self.num_pids is not None:
            classification_results = self.classifier(feat_after_bn)
            return feat_after_bn, classification_results
        else:
            return feat_after_bn