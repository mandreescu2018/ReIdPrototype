import torch
from .vit_model import build_transformer, build_transformer_local
from .mobilenet_v2 import MobileNetV2
from .resnet_CBN import ResNetBuilder
from .simple_model import SimpleReIDModel
from .resnet_BoT import BagOfTricksBuilder
# from config.factories_dict import model_factory

model_factory = {
    'vit_transformer': build_transformer,
    'vit_transformer_jpm': build_transformer_local,
    'mobilenet_v2': MobileNetV2,
    'resnet50': BagOfTricksBuilder,
    # 'resnet50': ResNetBuilder,
    'simple_resnet50': SimpleReIDModel
}

class ModelLoader:
    def __init__(self, cfg):
        self.cfg = cfg
        self._model = None

    @property
    def model(self):
        if self._model is None:            
            self._model = model_factory[self.cfg.MODEL.NAME](self.cfg).to(self.cfg.DEVICE)
        return self._model

    def load_param(self, optimizer=None, scheduler=None):
        start_epoch = 0
        if self.cfg.MODEL.PRETRAIN_CHOICE == 'resume':
            checkpoint = torch.load(self.cfg.MODEL.PRETRAIN_PATH)
            self._model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
        elif self.cfg.MODEL.PRETRAIN_CHOICE == 'test':
            checkpoint = torch.load(self.cfg.TEST.WEIGHT)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        return self.model, optimizer, scheduler, start_epoch
    
    