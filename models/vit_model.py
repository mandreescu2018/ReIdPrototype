import torch
import torch.nn as nn
import copy
from config.constants import *
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
from utils.weight_utils import weights_init_classifier, weights_init_kaiming
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID

factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID
}

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


def set_classifier(cfg):
    num_classes = cfg.DATASETS.NUMBER_OF_CLASSES
    in_planes = HIDDEN_SIZE_VIT_BASE
    classifier = None
    
    if cfg.LOSS.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
        classifier = id_loss_factory[cfg.LOSS.ID_LOSS_TYPE](in_planes, num_classes, s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)        
    return classifier

class TransformerConfig:
    def __init__(self, cfg):
        self.config = cfg
        self._img_size = None
        self._embedding_dimension = HIDDEN_SIZE_VIT_BASE
        
    @property
    def camera(self):
        return self.config.DATASETS.NUMBER_OF_CAMERAS if self.config.MODEL.SIE_CAMERA else 0
    
    @property
    def view(self):
        return self.config.DATASETS.NUMBER_OF_TRACKS if self.config.MODEL.SIE_VIEW else 0
    
    @property
    def img_size(self):
        if self._img_size is None:
            self._img_size = self.config.INPUT.SIZE_TRAIN
        return self._img_size
    
    @property
    def sie_xishu(self):
        return self.config.MODEL.SIE_COEFFICIENT
    
    @property
    def stride_size(self):
        return self.config.MODEL.STRIDE_SIZE
    
    @property
    def drop_path_rate(self):
        return self.config.MODEL.DROP_PATH
    
    @property
    def patch_size(self):
        return VIT_PATCH_SIZE
    
    @property
    def input_channels(self):
        return DEFAULT_INPUT_CHANNELS
    
    @property
    def embedding_dimension(self):
        if self.config.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
            self._embedding_dimension = 384
        return self._embedding_dimension
    
    @property
    def drop_out_rate(self):
        return self.config.MODEL.DROP_OUT
    
    @property
    def attn_drop_rate(self):
        return self.config.MODEL.ATT_DROP_RATE
    
    @property
    def local_feature(self):
        if self.config.MODEL.NAME == "vit_transformer_jpm":
            return True
        return False
    

class build_transformer(nn.Module):
    def __init__(self, cfg):
        super(build_transformer, self).__init__()
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        transformer_config = TransformerConfig(cfg)

        self.in_planes = transformer_config.embedding_dimension

        print(f'using Transformer_type: {cfg.MODEL.TRANSFORMER_TYPE} as a backbone')

        self.base = factory_T_type[cfg.MODEL.TRANSFORMER_TYPE](transformer_config)
        
        if cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
            self.base.load_param(cfg.MODEL.PRETRAIN_PATH)
            print(f'Loading pretrained ImageNet model......from {cfg.MODEL.PRETRAIN_PATH}')

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = set_classifier(cfg)
        if self.classifier is None:
            self.classifier = nn.Linear(self.in_planes, cfg.DATASETS.NUMBER_OF_CLASSES, bias=False)
            self.classifier.apply(weights_init_classifier)
        
        self.ID_LOSS_TYPE = cfg.LOSS.ID_LOSS_TYPE
        
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

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


class build_transformer_local(nn.Module):
    def __init__(self, cfg):
        super(build_transformer_local, self).__init__()
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        transformer_config = TransformerConfig(cfg)

        self.in_planes = transformer_config.embedding_dimension

        print(f'using Transformer_type: {cfg.MODEL.TRANSFORMER_TYPE} as a backbone'.format())

        self.base = factory_T_type[cfg.MODEL.TRANSFORMER_TYPE](transformer_config)

        if cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
            self.base.load_param(cfg.MODEL.PRETRAIN_PATH)
            print(f'Loading pretrained ImageNet model......from {cfg.MODEL.PRETRAIN_PATH}')

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

        self.num_classes = cfg.DATASETS.NUMBER_OF_CLASSES
        self.ID_LOSS_TYPE = cfg.LOSS.ID_LOSS_TYPE
        self.classifier = set_classifier(cfg)
        if self.classifier is None:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_1.apply(weights_init_classifier)
            self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_2.apply(weights_init_classifier)
            self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_3.apply(weights_init_classifier)
            self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_4.apply(weights_init_classifier)        

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_4.bias.requires_grad_(False)
        self.bottleneck_4.apply(weights_init_kaiming)

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
        # lf_1
        b1_local_feat = x[:, :patch_length]
        b1_local_feat = self.b2(torch.cat((token, b1_local_feat), dim=1))
        local_feat_1 = b1_local_feat[:, 0]

        # lf_2
        b2_local_feat = x[:, patch_length:patch_length*2]
        b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1))
        local_feat_2 = b2_local_feat[:, 0]

        # lf_3
        b3_local_feat = x[:, patch_length*2:patch_length*3]
        b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
        local_feat_3 = b3_local_feat[:, 0]

        # lf_4
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

