import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch._six import container_abcs
# import collections.abc as container_abcs
from config.constants import *
from .transformer_parts import PatchEmbed_overlap, Mlp

to_2tuple = nn.modules.utils._ntuple(2)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, 
                 embedding_dim, 
                 heads=8, 
                 mlp_dim=2048, 
                 dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attn = nn.MultiheadAttention(embedding_dim, 
                                          heads, 
                                          dropout=dropout, 
                                          batch_first=True)
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

class VisionTransformer(nn.Module):
    def __init__(self,
                 transformer_config, 
                 num_classes=1000,
                 embed_dim=VIT_BASE_HIDDEN_SIZE, 
                 num_transformer_layers=12, 
                 num_heads=8, 
                 mlp_dim=2048, 
                 dropout=0.1):
        super().__init__()
        
        self.patch_embed = PatchEmbed_overlap(transformer_config.img_size, 
                                              transformer_config.patch_size,
                                              stride_size=transformer_config.stride_size, 
                                              in_channels=transformer_config.input_channels, 
                                              embed_dim=embed_dim)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + self.patch_embed.num_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # self.local_feature = transformer_config.local_feature
        # if self.local_feature:
        #    num_transformer_layers -= 1
        # num_transformer_layers = 
        self.transformer = nn.Sequential(
            *[TransformerEncoderBlock(embed_dim, 
                                      num_heads, 
                                      mlp_dim, 
                                      dropout) for _ in range(num_transformer_layers)]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x, cam_label, view_label):
        B = x.size(0)
        # create the patch embedding
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        # create a cls token for each image in the batch
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        # concatenate the cls token embedding and patch embedding
        x = torch.cat((cls_tokens, x), dim=1)  # [B, 1 + num_patches, embed_dim]
        x = x + self.pos_embed
        x = self.dropout(x)

        x = self.transformer(x)  # [B, 1 + num_patches, embed_dim]
        x = self.norm(x[:, 0])   # take [CLS] token
        return x
        # return self.head(x)

if __name__ == "__main__":
    print(to_2tuple((3, 2)))