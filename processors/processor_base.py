import os
import time
import torch
import torch.nn as nn
from torch.cuda import amp
import logging
# from utils.meter import AverageMeter
# from utils.metrics import R1_mAP_eval
# from utils import Saver


class ProcessorBase:
    def __init__(self, cfg,                   
                 model,                                   
                 train_loader,
                 val_loader,                  
                 optimizer,
                 optimizer_center,
                 loss_fn,
                 epochs = 10):
        self.config = cfg
        self.model = model        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.optimizer_center = optimizer_center
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.device = cfg.DEVICE
    
    
    def train(self):
        self.model.to(self.device)
        

    def train_step(self):
        self.model.train()

    def validation_step(self):
        self.model.eval()