""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929

The official jax code is released and available at https://github.com/google-research/vision_transformer

Status/TODO:
* Models updated to be compatible with official impl. Args added to support backward compat for old PyTorch weights.
* Weights ported from official jax impl for 384x384 base and small models, 16x16 and 32x32 patches.
* Trained (supervised on ImageNet-1k) my custom 'small' patch model to 77.9, 'base' to 79.4 top-1 with this code.
* Hopefully find time and GPUs for SSL or unsupervised pretraining on OpenImages w/ ImageNet fine-tune in future.

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2020 Ross Wightman
"""
import math
from functools import partial
from itertools import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch._six import container_abcs
import collections.abc as container_abcs

from config.constants import *
from .transformer_parts import Mlp_ReID, PatchEmbed_overlap, HybridEmbed, Attention, DropPath
from utils.weight_utils import init_weights, trunc_normal

to_2tuple = nn.modules.utils._ntuple(2)

class TransformerEncoderBlock(nn.Module):

    def __init__(self, transformer_config, drop_path=0.):
        super().__init__()

        embedding_dim = transformer_config.hidden_size
        norm_layer = transformer_config.norm_layer

        self.norm1 = norm_layer(embedding_dim)
        self.attn = Attention(transformer_config)
        
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(embedding_dim)
        self.mlp = Mlp_ReID(transformer_config)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class TransReID(nn.Module):
    """ Transformer-based Object Re-Identification
    """
    def __init__(self, 
                 transformer_config, 
                 num_classes=1000, 
                 hybrid_backbone=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = transformer_config.hidden_size  # num_features for consistency with other models
        self.local_feature = transformer_config.local_feature
        
        num_transformer_layers = transformer_config.num_layers
        norm_layer = transformer_config.norm_layer

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, 
                img_size=transformer_config.img_size, 
                in_chans=transformer_config.input_channels, 
                embed_dim=self.embed_dim)
        else:
            self.patch_embed = PatchEmbed_overlap(
                img_size=transformer_config.img_size, 
                patch_size=transformer_config.patch_size, 
                stride_size=transformer_config.stride_size, 
                in_channels=transformer_config.input_channels,
                embed_dim=self.embed_dim)

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        
        self.cam_num = transformer_config.camera
        self.view_num = transformer_config.view
        self.sie_xishu = transformer_config.sie_xishu
        self.drop_rate = transformer_config.drop_out_rate

        self.drop_path_rate = transformer_config.drop_path_rate

        self._initialize_sie_embedding()
            
        self.pos_drop = nn.Dropout(p=self.drop_rate)
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, num_transformer_layers)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(transformer_config,
                                    drop_path=dpr[i])
            for i in range(num_transformer_layers)])

        self.norm = norm_layer(self.embed_dim)

        # Classifier head
        self.fc = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal(self.cls_token, std=.02)
        trunc_normal(self.pos_embed, std=.02)

        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.fc = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
    
    def _initialize_sie_embedding(self):
        """Initialize SIE Embedding
        """
        if self.cam_num > 1 and self.view_num > 1:
            sie_embed_size = self.cam_num * self.view_num
        elif self.cam_num > 1:
            sie_embed_size = self.cam_num
        elif self.view_num > 1:
            sie_embed_size = self.view_num
        else:
            sie_embed_size = 0

        if sie_embed_size > 0:
            self.sie_embed = nn.Parameter(torch.zeros(sie_embed_size, 1, self.embed_dim))
            trunc_normal(self.sie_embed, std=.02)


    def forward_features(self, x, camera_id, view_id):
        batch_size = x.shape[0]
        # create the patch embedding
        x = self.patch_embed(x)

        # create a cls token for each image in the batch
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # concatenate the cls token embedding and patch embedding
        x = torch.cat((cls_tokens, x), dim=1)

        # add the positional embedding combined with Side Information Embedding
        if self.cam_num > 0 and self.view_num > 0:
            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[camera_id * self.view_num + view_id]
        elif self.cam_num > 0:
            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[camera_id]
        elif self.view_num > 0:
            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[view_id]
        else:
            x = x + self.pos_embed

        # apply dropout to patch embedding
        x = self.pos_drop(x)

        # pass position and patch embedding through the transformer encoder
        if self.local_feature:
            for blk in self.blocks[:-1]:
                x = blk(x)
            return x

        else:
            for blk in self.blocks:
                x = blk(x)

            x = self.norm(x)

            return x[:, 0]

    def forward(self, x, cam_label=None, view_label=None):
        x = self.forward_features(x, cam_label, view_label)
        return x

    def load_param(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')
        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for k, v in param_dict.items():
            if 'head' in k or 'dist' in k:
                continue
            if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = self.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
                # To resize pos embedding when using model at different size from pretrained weights
                if 'distilled' in model_path:
                    print('distill need to choose right cls token in the pth')
                    v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
                v = resize_pos_embed(v, self.pos_embed, self.patch_embed.num_y, self.patch_embed.num_x)
            try:
                self.state_dict()[k].copy_(v)
            except:
                print('===========================ERROR=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape, self.state_dict()[k].shape))


def resize_pos_embed(posemb, posemb_new, hight, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = posemb_new.shape[1]

    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]
    ntok_new -= 1

    gs_old = int(math.sqrt(len(posemb_grid)))
    print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape, posemb_new.shape, hight, width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb
