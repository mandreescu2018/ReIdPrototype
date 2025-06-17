import math
import torch
import torch.nn as nn
from config.constants import *
from utils.weight_utils import init_patch_embed_weights

to_2tuple = nn.modules.utils._ntuple(2)

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def drop_path(self, x):
        """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

        This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
        the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
        See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
        changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
        'survival rate' as the argument.

        """
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output
    
    def forward(self, x):
        return self.drop_path(x)


class part_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        # add mask to q k v
        mask = mask.to(q.device.type)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(~mask.bool(), torch.tensor(-1e3, dtype=torch.float16)) # mask
        attn = attn.softmax(dim=-1)
        attn = torch.mul(attn, mask) ###
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

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
    def __init__(self, transformer_config, qk_scale=None):
        super().__init__()
        dim = transformer_config.hidden_size
        self.num_heads = transformer_config.num_heads

        head_dim = dim // self.num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        # a linear layer that projects the input into concatenated queries, keys, and values for all heads
        self.qkv = nn.Linear(dim, dim * 3, bias=transformer_config.qkv_bias)
        self.attn_drop = nn.Dropout(transformer_config.attn_drop_rate)
        # self.proj is a linear layer that projects the concatenated outputs 
        # of all heads back to the original embedding dimension
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(transformer_config.drop_out_rate)

    def forward(self, x, mask=None):
        B, N, C = x.shape # B=batch size, N=number of patches, C=embedding dimension
        # qkv is a tensor of shape (B, N, 3, num_heads, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        # Attention scores are computed as the scaled dot product between queries and keys, 
        # then normalized with softmax.
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(~mask.bool(), torch.tensor(-1e3, dtype=torch.float16)) # mask
        attn = attn.softmax(dim=-1)
        
        # Dropout is applied to the attention weights.
        attn = self.attn_drop(attn)
        if mask is not None:
            mask = mask.to(q.device)
            attn = torch.mul(attn, mask) ###
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        # The output has the same shape as the input and can be used in subsequent transformer layers.
        return x

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

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding - No overlapping patches
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

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

class Mlp_ReID(nn.Module):
    def __init__(self,
                 transformer_config,
                 out_features=None,
                 act_layer=nn.GELU):
        super().__init__()
        in_features = transformer_config.hidden_size
        hidden_features = int(in_features * transformer_config.mlp_ratio)
        # mlp_hidden_dim = int(embedding_dim * transformer_config.mlp_ratio)
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(transformer_config.drop_out_rate)

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