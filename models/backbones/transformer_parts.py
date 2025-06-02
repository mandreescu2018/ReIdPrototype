import math
import torch
import torch.nn as nn
from config.constants import *
from utils.weight_utils import init_patch_embed_weights

to_2tuple = nn.modules.utils._ntuple(2)

class PatchEmbed_overlap(nn.Module):
    """ Image to Patch Embedding with overlapping patches
    """
    def __init__(self, img_size=224, patch_size=16, stride_size=20, in_channels=3, embed_dim=VIT_BASE_HIDDEN_SIZE):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size_tuple = to_2tuple(stride_size)
        self.num_x = (self.img_size[1] - patch_size[1]) // stride_size_tuple[1] + 1
        self.num_y = (self.img_size[0] - patch_size[0]) // stride_size_tuple[0] + 1
        print(f'using stride: {stride_size}, and patch number is num_y: {self.num_y} * num_x: {self.num_x}')
        self.num_patches = self.num_x * self.num_y

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=stride_size)
        
        init_patch_embed_weights(self)        

    def forward(self, x):
        B, C, H, W = x.shape

        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)

        x = x.flatten(2).transpose(1, 2) # [64, 8, 768]
        return x

class PatchEmbed_Alpha(nn.Module):
    """ Image to Patch Embedding - No overlapping patches
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


class PatchEmbedding(nn.Module):
    """Vanilla implementation - No overlapping patches"""
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_channels=3, 
                 embed_dim=VIT_BASE_HIDDEN_SIZE):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, embed_dim, H/patch, W/patch]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x

class Mlp_Alpha(nn.Module):
    def __init__(self, 
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU, 
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# The original ViT paper ("An Image is Worth 16x16 Words") from Google does use dropout in the MLP.
# However, some minimal implementations omit dropout after GELU for simplicity or regularization tuning purposes.
# Different ViT variants (like DeiT, Swin Transformer, etc.) vary in how they structure the MLP: 
# sometimes using dropout only once, or not at all.
class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 drop=0.,
                 embedding_dim=VIT_BASE_HIDDEN_SIZE):
        super().__init__()

        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_features, out_features),
            nn.Dropout(drop)
        )

    def forward(self, x):
        return self.mlp(self.layer_norm(x))
    
class MultiHeadSelfAttentionBlock(nn.Module):
    """
    MultiHeadSelfAttentionBlock is a block that contains a multi-head self-attention mechanism.
    """
    def __init__(self, 
                 embedding_dim: int=768,
                 num_heads: int=12,
                 attn_dropout: float=0):
        super(MultiHeadSelfAttentionBlock, self).__init__()

        # layer normalization is a technique to normalize 
        # the distribution of intermediate layers in the network
        # Normalization make everything have the same mean and same std
        # normalize along the embedding dimension, it's like making all of the stair in the staircase the same size
        self.norm_layer = nn.LayerNorm(normalized_shape=embedding_dim)

        # create multihead attention layer
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim, 
                                                    num_heads=num_heads, 
                                                    dropout=attn_dropout, 
                                                    batch_first=True)  # batch_first=True means that the input and output tensors are 
                                                                       # provided as (batch, seq, feature) -> (batch, number_of_patches, embedding_dim)       

    def forward(self, x):
        x = self.norm_layer(x)
        attn_output, _ = self.multihead_attn(query=x, 
                                             key=x, 
                                             value=x,
                                             need_weights=False)
        return attn_output

class TransformerEncoderBlock(nn.Module):
    def __init__(self, 
                 embedding_dim, 
                 heads=8, 
                 mlp_dim=2048, 
                 dropout=0.1):
        super().__init__()
        
        self.attn = MultiHeadSelfAttentionBlock(embedding_dim=embedding_dim, 
                                                num_heads=heads, 
                                                attn_dropout=dropout)
        self.mlp = Mlp(in_features=embedding_dim, 
                       hidden_features=mlp_dim, 
                       out_features=embedding_dim,
                       drop=dropout)        

    def forward(self, x):
        x = self.norm1(x)
        x = x + self.attn(query=x, 
                          key=x, 
                          value=x)[0]
        x = x + self.mlp(x)
        return x