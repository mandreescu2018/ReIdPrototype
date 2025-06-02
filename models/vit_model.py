import torch
import torch.nn as nn
import copy
from config.constants import *
from config.vit_config import TransformerConfig
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
from utils.weight_utils import weights_init_classifier, weights_init_kaiming
from .backbones.vit_pytorch import TransReID

id_loss_factory = {
    'arcface': Arcface,
    'cosface': Cosface,
    'amsoftmax': AMSoftmax,
    'circle': CircleLoss
}

def shuffle_unit(features, shift, group, begin=1):

    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x

class vit_builder_base(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        self.transformer_config = TransformerConfig(cfg)
        self.in_planes = self.transformer_config.embedding_dimension
        self.num_classes = cfg.DATASETS.NUMBER_OF_CLASSES

        print(f'using Transformer_type: {cfg.MODEL.TRANSFORMER_TYPE} as a backbone'.format())

        self.base = TransReID(self.transformer_config)

        if cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
            self.base.load_param(cfg.MODEL.PRETRAIN_PATH)
            print(f'Loading pretrained ImageNet model......from {cfg.MODEL.PRETRAIN_PATH}')
    
    def _init_bottleneck_layers(self, num_layers=5):
        """ Initialize bottleneck layers for the model. """
        self.bottlenecks = []
        for _ in range(num_layers):
            bottleneck = nn.BatchNorm1d(self.in_planes)
            bottleneck.bias.requires_grad_(False)
            bottleneck.apply(weights_init_kaiming)
            self.bottlenecks.append(bottleneck)
            
        if num_layers == 1:
            self.bottleneck = self.bottlenecks[0]
        else:
            self.bottleneck, self.bottleneck_1, self.bottleneck_2, self.bottleneck_3, self.bottleneck_4 = self.bottlenecks
    
    def _init_classifier_layers(self, num_classifiers=5):
        """ Initialize classifier layers for the model. """
        in_planes = self.transformer_config.embedding_dimension
        
        if self.config.LOSS.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
            self.classifier = id_loss_factory[self.config.LOSS.ID_LOSS_TYPE](in_planes, 
                                                                             self.config.DATASETS.NUMBER_OF_CLASSES, 
                                                                             s=self.config.SOLVER.COSINE_SCALE, 
                                                                             m=self.config.SOLVER.COSINE_MARGIN)        
        else:
            # Initialize multiple linear classifiers for local features
            if num_classifiers == 1:
                self.classifier = nn.Linear(in_planes, self.num_classes, bias=False)
                self.classifier.apply(weights_init_classifier)
            else:
                classifiers = []
                for _ in range(num_classifiers):
                    classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
                    classifier.apply(weights_init_classifier)
                    classifiers.append(classifier)
                self.classifier, self.classifier_1, self.classifier_2, self.classifier_3, self.classifier_4 = classifiers


class build_transformer(vit_builder_base):
    def __init__(self, cfg):
        super(build_transformer, self).__init__(cfg)
        # self.gap = nn.AdaptiveAvgPool2d(1) - unnecessary ?
        
        self.ID_LOSS_TYPE = cfg.LOSS.ID_LOSS_TYPE

        self._init_classifier_layers(num_classifiers=1)  # Initialize classifier layers
        self._init_bottleneck_layers(num_layers=1)  # Initialize bottleneck layers

    def forward(self, x, label=0, cam_label= 0, view_label=0):
        global_feat = self.base(x, cam_label=cam_label, view_label=view_label)

        feat = self.bottleneck(global_feat)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)

            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat


class build_transformer_local(vit_builder_base):
    def __init__(self, cfg):
        super(build_transformer_local, self).__init__(cfg=cfg)

        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b2 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )

        self.ID_LOSS_TYPE = cfg.LOSS.ID_LOSS_TYPE
        
        self._init_classifier_layers()            
        self._init_bottleneck_layers()

        self.shuffle_groups = cfg.MODEL.SHUFFLE_GROUP
        print(f'using shuffle_groups size:{self.shuffle_groups}')
        self.shift_num = cfg.MODEL.SHIFT_NUM
        print(f'using shift_num size:{self.shift_num}')
        self.divide_length = cfg.MODEL.DEVIDE_LENGTH
        print(f'using divide_length size:{self.divide_length}')
        self.rearrange = cfg.MODEL.RE_ARRANGE

    def forward(self, x, label=None, cam_label= None, view_label=None):  # label is unused if self.cos_layer == 'no'

        features = self.base(x, cam_label=cam_label, view_label=view_label)

        # global branch
        b1_feat = self.b1(features) # [64, 129, 768]
        global_feat = b1_feat[:, 0]

        # JPM branch
        feature_length = features.size(1) - 1
        patch_length = feature_length // self.divide_length
        token = features[:, 0:1]

        if self.rearrange:
            x = shuffle_unit(features, self.shift_num, self.shuffle_groups)
        else:
            x = features[:, 1:]
        # Processing local feature segment 1 for JPM branch
        b1_local_feat = x[:, :patch_length]
        b1_local_feat = self.b2(torch.cat((token, b1_local_feat), dim=1))
        local_feat_1 = b1_local_feat[:, 0]

        # Processing local feature segment 2 for JPM branch
        b2_local_feat = x[:, patch_length:patch_length*2]
        b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1))
        local_feat_2 = b2_local_feat[:, 0]

        # Processing local feature segment 3 for JPM branch
        b3_local_feat = x[:, patch_length*2:patch_length*3]
        b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
        local_feat_3 = b3_local_feat[:, 0]

        # Processing local feature segment 4 for JPM branch
        b4_local_feat = x[:, patch_length*3:patch_length*4]
        b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
        local_feat_4 = b4_local_feat[:, 0]

        feat = self.bottleneck(global_feat)

        local_feat_1_bn = self.bottleneck_1(local_feat_1)
        local_feat_2_bn = self.bottleneck_2(local_feat_2)
        local_feat_3_bn = self.bottleneck_3(local_feat_3)
        local_feat_4_bn = self.bottleneck_4(local_feat_4)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
                cls_score_1 = self.classifier_1(local_feat_1_bn)
                cls_score_2 = self.classifier_2(local_feat_2_bn)
                cls_score_3 = self.classifier_3(local_feat_3_bn)
                cls_score_4 = self.classifier_4(local_feat_4_bn)
            return [cls_score, cls_score_1, cls_score_2, cls_score_3,
                        cls_score_4
                        ], [global_feat, local_feat_1, local_feat_2, local_feat_3,
                            local_feat_4]  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                return torch.cat(
                    [feat, local_feat_1_bn / 4, local_feat_2_bn / 4, local_feat_3_bn / 4, local_feat_4_bn / 4], dim=1)
            else:
                return torch.cat(
                    [global_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4], dim=1)

