import torch
from .vit_model import build_transformer, build_transformer_local
from .mobilenet_v2 import MobileNetV2
from .resnet_CBN import ResNetBuilder
from .simple_model import SimpleReIDModel
from .resnet_BoT import BagOfTricksBuilder
from .hacnn_model import HACNNBuilder
from utils.device_manager import DeviceManager

model_factory = {
    'vit_transformer': build_transformer,
    'vit_transformer_jpm': build_transformer_local,
    'mobilenet_v2': MobileNetV2,
    'resnet50': BagOfTricksBuilder,
    # 'resnet50': ResNetBuilder,
    'simple_resnet50': SimpleReIDModel,
    'hacnn': HACNNBuilder
}

class ModelLoader:
    def __init__(self, cfg):
        self.cfg = cfg
        self._model = None
        self._start_epoch = 0
        self._optimizer = None
        self._optimizer_center = None
        self.scheduler = None
        self._center_criterion = None
        self._checkpoint = None

    @property
    def checkpoint(self):
        if self._checkpoint is None:
            if self.cfg.MODEL.PRETRAIN_CHOICE == 'resume':
                self._checkpoint = torch.load(self.cfg.MODEL.PRETRAIN_PATH)
            elif self.cfg.MODEL.PRETRAIN_CHOICE == 'test' or self.cfg.MODEL.PRETRAIN_CHOICE == 'cross_domain':
                self._checkpoint = torch.load(self.cfg.TEST.WEIGHT, weights_only=True)
        return self._checkpoint

    @property
    def model(self):
        if self._model is None:            
            self._model = model_factory[self.cfg.MODEL.NAME](self.cfg).to(DeviceManager.get_device())
        return self._model

    @property
    def start_epoch(self):
        if self._start_epoch == 0 and self.cfg.MODEL.PRETRAIN_CHOICE == 'resume':
            self._start_epoch = self.checkpoint.get('epoch', 0)        
        return self._start_epoch

    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
    
    @property
    def optimizer_center(self):
        return self._optimizer_center
    
    @optimizer_center.setter
    def optimizer_center(self, optimizer_center):
        self._optimizer_center = optimizer_center
    
    @property
    def center_criterion(self):
        return self._center_criterion
    
    @center_criterion.setter
    def center_criterion(self, center_criterion):
        self._center_criterion = center_criterion
    
    @property
    def scheduler(self):
        return self._scheduler
    
    @scheduler.setter
    def scheduler(self, scheduler):
        self._scheduler = scheduler

    def load_param_cross(self, param_dict):
        for i in param_dict:            
            if 'classifier' in i:
                continue
            self.model.state_dict()[i].copy_(param_dict[i])

    def load_param(self):
        
        if self.cfg.MODEL.PRETRAIN_CHOICE == 'resume':
            self._model.load_state_dict(self.checkpoint['model_state_dict'])
            self._optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
            optimizer_center_state_dict = self.checkpoint.get('optimizer_center_state_dict', None)
            if optimizer_center_state_dict is not None:
                self._optimizer_center.load_state_dict(optimizer_center_state_dict)
                self._center_criterion.load_state_dict(self.checkpoint['center_criterion_state_dict'])
            self._optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
            self._scheduler.load_state_dict(self.checkpoint['scheduler_state_dict'])
        elif self.cfg.MODEL.PRETRAIN_CHOICE == 'test':
            self.model.load_state_dict(self.checkpoint['model_state_dict'])
        elif self.cfg.MODEL.PRETRAIN_CHOICE == 'cross_domain':
            self.load_param_cross(self.checkpoint['model_state_dict'])
            
        
    
    