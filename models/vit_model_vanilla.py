import torch
import torch.nn as nn
import copy
from config.constants import *
from config.vit_config import TransformerConfig
# from vitutils.vit_config import TransformerConfig
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
from utils.weight_utils import weights_init_classifier, weights_init_kaiming
from .backbones.vanilla_vit import VisionTransformer
# from .backbones.visual_transformer_pytorch import get_vit_model
import torchvision.models.vision_transformer as vits
from .backbones.visual_transformer_pytorch import TorchvisionVIT

def get_vit_model(model_name, pretrained=False, num_classes=1000, num_id_classes=1000):
    """
    Get a Vision Transformer model from torchvision with specified parameters.
    
    Args:
        model_name (str): Name of the Vision Transformer model.
        pretrained (bool): Whether to load pretrained weights.
        num_classes (int): Number of output classes for classification.
        
    Returns:
        nn.Module: The Vision Transformer model.
    """
    if model_name == 'vit_b_16':
        model = vits.vit_b_16(pretrained=pretrained)
    elif model_name == 'vit_b_32':
        model = vits.vit_b_32(pretrained=pretrained)
    elif model_name == 'vit_l_16':
        model = vits.vit_l_16(pretrained=pretrained)
    elif model_name == 'vit_l_32':
        model = vits.vit_l_32(pretrained=pretrained)
    else:
        raise ValueError(f"Model {model_name} is not supported.")
    
    model.heads = nn.Linear(model.heads.head.in_features, model.heads.head.in_features)

    return model

def vit_base_patch16_224_TransReID_vanilla(trans_config):
    
    model = VisionTransformer(trans_config,
                      num_transformer_layers=VIT_BASE_LAYERS, 
                      num_heads=VIT_BASE_HEADS)

    return model


factory_T_type = {
    'vit_base_patch16_224_TransReID_vanilla': vit_base_patch16_224_TransReID_vanilla,
    
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
    in_planes = VIT_BASE_HIDDEN_SIZE
    classifier = None
    
    if cfg.LOSS.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
        classifier = id_loss_factory[cfg.LOSS.ID_LOSS_TYPE](in_planes, num_classes, s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)        
    return classifier


class build_transformer_vanilla(nn.Module):
    def __init__(self, cfg):
        super(build_transformer_vanilla, self).__init__()
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        transformer_config = TransformerConfig(cfg)

        self.in_planes = transformer_config.embedding_dimension

        print(f'using Transformer_type: {cfg.MODEL.TRANSFORMER_TYPE} as a backbone')

        # self.base = factory_T_type[cfg.MODEL.TRANSFORMER_TYPE](transformer_config)
        # self.base = self.base.to(cfg.MODEL.DEVICE)
        if cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
            self.base = TorchvisionVIT(cfg.MODEL.TRANSFORMER.TYPE, pretrained=True, num_id_classes= cfg.DATASETS.NUMBER_OF_CLASSES)
        else:
            self.base = TorchvisionVIT(cfg.MODEL.TRANSFORMER.TYPE, pretrained=False, num_id_classes= cfg.DATASETS.NUMBER_OF_CLASSES)
            self.base.load_param(cfg.MODEL.PRETRAIN_PATH)
        
        # if cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
        #     self.base.load_param(cfg.MODEL.PRETRAIN_PATH)
        #     print(f'Loading pretrained ImageNet model......from {cfg.MODEL.PRETRAIN_PATH}')

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
        global_feat = self.base(x)

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


