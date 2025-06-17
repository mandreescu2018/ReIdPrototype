import torch
import torch.nn as nn
import os
from config.constants import *
from config.vit_config import TransformerConfig
# from vitutils.vit_config import TransformerConfig
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
from utils.weight_utils import weights_init_classifier, weights_init_kaiming
from .backbones.vanilla_vit import VisionTransformer
# from .backbones.visual_transformer_pytorch import get_vit_model
import torchvision.models.vision_transformer as vits
from .backbones.vit_pat_pytorch import part_Attention_ViT

class build_part_attention_vit(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # model_path_base = cfg.MODEL.PRETRAIN_PATH
        # if pretrain_tag == 'lup':
        #     path = lup_path_name[cfg.MODEL.TRANSFORMER_TYPE]
        # else:
        #     path = imagenet_path_name[cfg.MODEL.TRANSFORMER_TYPE]
        self.model_path = cfg.MODEL.PRETRAIN_PATH
        self.pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: part token vit as a backbone')

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = cfg.DATASETS.NUMBER_OF_CLASSES
        self.transformer_config = TransformerConfig(cfg)

        self.base = part_Attention_ViT(transformer_config= self.transformer_config,
                                       pretrain_tag=self.pretrain_choice)
        # self.base = factory[cfg.MODEL.TRANSFORMER_TYPE]\
        #     (img_size=cfg.INPUT.SIZE_TRAIN,
        #     stride_size=cfg.MODEL.STRIDE_SIZE,
        #     drop_path_rate=cfg.MODEL.DROP_PATH,
        #     drop_rate= cfg.MODEL.DROP_OUT,
        #     attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
        #     pretrain_tag=pretrain_tag)
        # if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
        #     self.in_planes = 384
        # elif cfg.MODEL.TRANSFORMER_TYPE == 'deit_tiny_patch16_224_TransReID':
        #     self.in_planes = 192
        # elif cfg.MODEL.TRANSFORMER_TYPE == 'vit_large_patch16_224_TransReID':
        #     self.in_planes = 1024
        # if self.pretrain_choice == 'imagenet':
        if cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
            self.base.load_param(cfg.MODEL.PRETRAIN_PATH)
            print(f'Loading pretrained ImageNet model......from {cfg.MODEL.PRETRAIN_PATH}')

        # self.base.load_param(self.model_path)
        # print('Loading pretrained ImageNet model......from {}'.format(self.model_path))

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        layerwise_tokens = self.base(x) # B, N, C
        layerwise_cls_tokens = [t[:, 0] for t in layerwise_tokens] # cls token
        part_feat_list = layerwise_tokens[-1][:, 1: 4] # 3, 768

        layerwise_part_tokens = [[t[:, i] for i in range(1,4)] for t in layerwise_tokens] # 12 3 768
        feat = self.bottleneck(layerwise_cls_tokens[-1])

        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, layerwise_cls_tokens, layerwise_part_tokens
        else:
            return feat if self.neck_feat == 'after' else layerwise_cls_tokens[-1]

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i: # drop classifier
                continue
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading trained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

    def compute_num_params(self):
        total = sum([param.nelement() for param in self.parameters()])
        return total
        # logger = logging.getLogger('PAT.train')
        # logger.info("Number of parameter: %.2fM" % (total/1e6))        