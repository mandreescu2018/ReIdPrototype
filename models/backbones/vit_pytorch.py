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
from .transformer_parts import Mlp_Alpha, PatchEmbed_overlap
from utils.weight_utils import init_weights, trunc_normal, init_patch_embed_weights

to_2tuple = nn.modules.utils._ntuple(2)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Attention(nn.Module):
    """ Multi-Head Attention module with support for qkv_bias and qk_scale
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads. specifies how many attention heads to use. 
        The input dimension (dim) is split across these heads
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: False
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        attn_drop (float, optional): Dropout ratio for attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio after projection. Default: 0.0
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        # a linear layer that projects the input into concatenated queries, keys, and values for all heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        # self.proj is a linear layer that projects the concatenated outputs 
        # of all heads back to the original embedding dimension
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape # B=batch size, N=number of patches, C=embedding dimension
        # qkv is a tensor of shape (B, N, 3, num_heads, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        # Attention scores are computed as the scaled dot product between queries and keys, 
        # then normalized with softmax.
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # Dropout is applied to the attention weights.
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        # The output has the same shape as the input and can be used in subsequent transformer layers.
        return x

class TransformerEncoderBlock(nn.Module):

    def __init__(self, 
                 embedding_dim, 
                 num_heads, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop=0., 
                 attn_drop=0.,
                 drop_path=0., 
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(embedding_dim)
        self.attn = Attention(
            embedding_dim, 
            num_heads=num_heads, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            attn_drop=attn_drop, 
            proj_drop=drop)
        
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(embedding_dim)
        mlp_hidden_dim = int(embedding_dim * mlp_ratio)
        self.mlp = Mlp_Alpha(in_features=embedding_dim, 
                       hidden_features=mlp_hidden_dim, 
                       drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, 1)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class TransReID(nn.Module):
    """ Transformer-based Object Re-Identification
    """
    def __init__(self, 
                 transformer_config, 
                 num_classes=1000, 
                 qk_scale=None,
                 hybrid_backbone=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = transformer_config.embedding_dimension  # num_features for consistency with other models
        self.local_feature = transformer_config.local_feature
        
        num_heads=transformer_config.num_heads
        num_transformer_layers = transformer_config.num_layers
        mlp_ratio = transformer_config.mlp_ratio
        qkv_bias = transformer_config.qkv_bias
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
        self.attn_drop_rate = transformer_config.attn_drop_rate

        self._initialize_sie_embedding()
            
        # self._print_input_param()

        self.pos_drop = nn.Dropout(p=self.drop_rate)
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, num_transformer_layers)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embedding_dim=self.embed_dim, 
                                    num_heads=num_heads, 
                                    mlp_ratio=mlp_ratio, 
                                    qkv_bias= qkv_bias, 
                                    qk_scale=qk_scale,
                                    drop=self.drop_rate, 
                                    attn_drop=self.attn_drop_rate, 
                                    drop_path=dpr[i], 
                                    norm_layer=norm_layer)
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
