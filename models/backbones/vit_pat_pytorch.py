import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.weight_utils import init_weights, trunc_normal, init_patch_embed_weights
from .transformer_parts import Mlp_ReID, PatchEmbed_overlap, HybridEmbed, Attention, DropPath

to_2tuple = nn.modules.utils._ntuple(2)

# for Part_Attention
def generate_2d_mask(H=16, W=8, left=0, top=0, width=8, height=8, part=-1, cls_label=True, device='cuda'):
    H, W, left, top, width, height = \
        int(H), int(W), int(left), int(top), int(width), int(height)
    assert left + width <= W and top + height <= H
    l, w = sorted(random.sample(range(left, left + width + 1), 2))
    t, h = sorted(random.sample(range(top, top + height + 1), 2))
    # l,w,t,h = left, left+width, top, top+height ### for test
    mask = torch.zeros([H, W], device=device)
    mask[t : h + 1, l : w + 1] = 1
    mask = mask.flatten(0)
    mask_ = torch.zeros([len(mask) + 4], device=device)
    mask_[4:] = mask
    mask_[part] = 1
    mask_[0] = 1 if cls_label else 0 ######### cls token
    mask_ = mask_.unsqueeze(1) # N x 1
    mask_ = mask_ @ mask_.t() # N x N
    return mask_

class IBN(nn.Module):
    def __init__(self, planes):
        super(IBN, self).__init__()
        half1 = int(planes/2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)
    
    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out   

class PatchEmbed_conv_stem(nn.Module):
    """ Image to Patch Embedding with overlapping patches
    """
    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, embed_dim=768, stem_conv=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size_tuple = to_2tuple(stride_size)
        self.num_x = (img_size[1] - patch_size[1]) // stride_size_tuple[1] + 1
        self.num_y = (img_size[0] - patch_size[0]) // stride_size_tuple[0] + 1
        print('using stride: {}, and patch number is num_y{} * num_x{}'.format(stride_size, self.num_y, self.num_x))
        self.num_patches = self.num_x * self.num_y
        self.img_size = img_size
        self.patch_size = patch_size

        self.stem_conv = stem_conv
        if self.stem_conv:
            hidden_dim = 64
            stem_stride = 2
            stride_size = patch_size = patch_size[0] // stem_stride
            self.conv = nn.Sequential(
                nn.Conv2d(in_chans, hidden_dim, kernel_size=7, stride=stem_stride, padding=3,bias=False),
                IBN(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,padding=1,bias=False),
                IBN(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,padding=1,bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            )
            in_chans = hidden_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)

    def forward(self, x):
        if self.stem_conv:
            x = self.conv(x)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2) # [64, 8, 768]
        return x

class part_Attention_Block(nn.Module):
    def __init__(self, transformer_config,
                 drop_path=0.,norm_layer=nn.LayerNorm):
        super().__init__()
        dim = transformer_config.hidden_size
        self.norm1 = norm_layer(dim)
        self.part_attn = Attention(transformer_config)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        # mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_ReID(transformer_config)

    def forward(self, x, mask = None):
        # part attention
        x = x + self.drop_path(self.part_attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class part_Attention_ViT(nn.Module):
    def __init__(self, 
                 transformer_config, 
                 num_classes=1000,
                 pretrain_tag = 'Imagenet', 
                 hybrid_backbone=None, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = transformer_config.hidden_size  # num_features for consistency with other models

        depth = transformer_config.num_layers
        self.pretrain_tag = pretrain_tag
        
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
            
        self.drop_rate = transformer_config.drop_out_rate
        self.drop_path_rate = transformer_config.drop_path_rate
        
        # self.pretrain_tag = kwargs['pretrain_tag']
        # if hybrid_backbone is not None:
        #     self.patch_embed = HybridEmbed(
        #         hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=self.embed_dim)
        # elif kwargs['pretrain_tag'] == 'lup':
        #     self.patch_embed = PatchEmbed_conv_stem(
        #         img_size=img_size, patch_size=patch_size, stride_size=stride_size, in_chans=in_chans,
        #         embed_dim=self.embed_dim, stem_conv = True)
        # else:
        #     self.patch_embed = PatchEmbed_overlap(
        #         img_size=img_size, patch_size=patch_size, stride_size=stride_size, in_chans=in_chans,
        #         embed_dim=self.embed_dim)

        # MoCo V3
        # self.patch_embed.proj.weight.requires_grad = False
        # self.patch_embed.proj.bias.requires_grad = False

        num_patches = self.patch_embed.num_patches + 4
        self.num_patches = num_patches
        # self.num_heads = num_heads

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.part_token1 = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.part_token2 = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.part_token3 = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))

        # print('using drop_out rate is : {}'.format(drop_rate))
        # print('using attn_drop_out rate is : {}'.format(attn_drop_rate))
        # print('using drop_path rate is : {}'.format(drop_path_rate))

        self.pos_drop = nn.Dropout(p=self.drop_rate)
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, depth)]  # stochastic depth decay rule
        
        #### mask
        # self.mask = self.attn_mask_generate(num_patches, self.patch_embed.num_y, self.patch_embed.num_x)
        
        self.blocks = nn.ModuleList([
            part_Attention_Block(transformer_config, 
                                 drop_path=dpr[i], 
                                 norm_layer=transformer_config.norm_layer)
            # part_Attention_Block(
            #     dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            #     drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
            
        self.depth = depth
        self.norm = transformer_config.norm_layer(self.embed_dim)

        # Classifier head
        self.fc = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal(self.cls_token, std=.02)
        trunc_normal(self.part_token1, std=.02)
        trunc_normal(self.part_token2, std=.02)
        trunc_normal(self.part_token3, std=.02)
        trunc_normal(self.pos_embed, std=.02)

        self.apply(init_weights)

    def attn_mask_generate(self, N=132, H=16, W=8, device='cuda'):
        mask = torch.ones(N,1, device=device)
        mask[1 : 4, 0] = 0
        mask_ = (mask @ mask.t()).bool()
        mask_ |= generate_2d_mask(H,W,0,0,W,H/2,1,False, device).bool()
        mask_ |= generate_2d_mask(H,W,0,H/4,W,H/2,2, False, device).bool()
        mask_ |= generate_2d_mask(H,W,0,H/2,W,H/2,3, False, device).bool()
        mask_[1 : 4, 0] = True
        mask_[0, 1 : 4] = True
        return mask_
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'part_token1', 'part_token2', 'part_token3'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.fc = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        part_token1 = self.part_token1.expand(B, -1, -1)
        part_token2 = self.part_token2.expand(B, -1, -1)
        part_token3 = self.part_token3.expand(B, -1, -1)
        x = torch.cat((cls_tokens, part_token1, part_token2, part_token3, x), dim=1)

        x = x + self.pos_embed

        x = self.pos_drop(x)
        layerwise_tokens = []

        mask = torch.ones([B, 1, self.num_patches, self.num_patches], device=x.device.type)
        # if self.training:
            # mask[:, 0] = self.mask
        # for i in range(B):
        mask[:, 0] = self.attn_mask_generate(self.num_patches, self.patch_embed.num_y, self.patch_embed.num_x, x.device.type)
        for blk in self.blocks:
            x = blk(x, mask)
            layerwise_tokens.append(x)
        layerwise_tokens = [self.norm(t) for t in layerwise_tokens]
        return layerwise_tokens

    def forward(self, x):
        x = self.forward_features(x)
        return x

    def load_param(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')
        count = 0
        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for k, v in param_dict.items():
            if 'head' in k or 'dist' in k or 'pre_logits' in k: # ViT-L
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
                    v = resize_pos_embed_part_vit(v, self.pos_embed, self.patch_embed.num_y, self.patch_embed.num_x)
                elif self.pretrain_tag == 'lup':
                    v_old = v
                    b, n, c = v.size()
                    v = torch.zeros([b,n+3,c], dtype=v_old.dtype)
                    v[:, :3] = v_old[:, 0]
                    v[:, 3:] = v_old
                    print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(v_old.shape, v.shape, self.patch_embed.num_y, self.patch_embed.num_x))
                else:
                    v = resize_pos_embed_part_vit(v, self.pos_embed, self.patch_embed.num_y, self.patch_embed.num_x)
            elif 'cls_token' in k:
                self.state_dict()['part_token1'].copy_(v)
                self.state_dict()['part_token2'].copy_(v)
                self.state_dict()['part_token3'].copy_(v)
                self.state_dict()[k].copy_(v)
                count += 4
                continue
            elif 'attn' in k:
                self.state_dict()[k.replace('attn', 'part_attn')].copy_(v)
                count += 1
                continue
            try:
                self.state_dict()[k].copy_(v)
                count += 1
            except:
                print('===========================ERROR=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape, self.state_dict()[k].shape))
        print('Load %d / %d layers.'%(count,len(self.state_dict().keys())))

    def compute_num_params(self):
        total = sum([param.nelement() for param in self.parameters()])
        print("Number of parameter: %.2fM" % (total/1e6))

def resize_pos_embed_part_vit(posemb, posemb_new, hight, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = posemb_new.shape[1]

    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]
    ntok_new -= 4

    gs_old = int(math.sqrt(len(posemb_grid)))
    print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape, posemb_new.shape, hight, width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_token, posemb_token, posemb_token, posemb_grid], dim=1)
    return posemb
