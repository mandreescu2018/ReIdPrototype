from torch import nn
from functools import partial
from config.constants import *

class TransformerConfig:
    transformer_factory = {
        'vit_base_patch16_224_TransReID': {'num_heads': VIT_BASE_HEADS, 'num_layers': VIT_BASE_LAYERS, 'mlp_ratio': 4.0, 'qkv_bias': True},
        'vit_small_patch16_224_TransReID': {'num_heads': VIT_SMALL_HEADS, 'num_layers': VIT_SMALL_LAYERS, 'mlp_ratio': 3.0, 'qkv_bias': False},
        'deit_small_patch16_224_TransReID': {'num_heads': DEIT_HEADS, 'num_layers': VIT_BASE_LAYERS, 'mlp_ratio': 4.0, 'qkv_bias': True},
    }
    def __init__(self, cfg):
        self.config = cfg
        self._img_size = None
        self._embedding_dimension = VIT_BASE_HIDDEN_SIZE
        
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
        if self.config.MODEL.TRANSFORMER.TYPE == 'deit_small_patch16_224_TransReID':
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
    
    @property
    def num_heads(self):
        if self.config.MODEL.TRANSFORMER.NUM_HEADS is None:
            return self.transformer_factory[self.config.MODEL.TRANSFORMER.TYPE]['num_heads']
        return self.config.MODEL.TRANSFORMER.NUM_HEADS
    
    @property
    def num_layers(self):
        if self.config.MODEL.TRANSFORMER.LAYERS is None:
            return self.transformer_factory[self.config.MODEL.TRANSFORMER.TYPE]['num_layers']
        return self.config.MODEL.TRANSFORMER.LAYERS

    @property
    def norm_layer(self):
        return partial(nn.LayerNorm, eps=1e-6)
    
    @property
    def mlp_ratio(self):
        if self.config.MODEL.TRANSFORMER.MLP_RATIO is None:
            return self.transformer_factory[self.config.MODEL.TRANSFORMER.TYPE]['mlp_ratio']
        return self.config.MODEL.TRANSFORMER.MLP_RATIO
    
    @property
    def qkv_bias(self):
        if self.config.MODEL.TRANSFORMER.QKV_BIAS is None:
            return self.transformer_factory[self.config.MODEL.TRANSFORMER.TYPE]['qkv_bias']
        return self.config.MODEL.TRANSFORMER.QKV_BIAS