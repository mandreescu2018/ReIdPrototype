"""
Rethinking the Distribution Gap of Person Re-identification with Camera-based Batch Normalization

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones.resnet_backbone import ResNet_Backbone
from .backbones.resnet_BoT_backbone import ResNet
from utils.weight_utils import weights_init_kaiming, weights_init_classifier

class BagOfTricksBuilder(nn.Module):
    
    def __init__(self, cfg):
        last_stride=1

        super(BagOfTricksBuilder, self).__init__()
        self.base = ResNet(last_stride)
        self.base.load_param(cfg.MODEL.PRETRAIN_PATH)

        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.in_planes = cfg.SOLVER.FEATURE_DIMENSION
        self.num_classes = cfg.DATASETS.NUMBER_OF_CLASSES
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x):

        global_feat = self.gap(self.base(x))  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)  # normalize for angular softmax

        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, weights_only=True)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
