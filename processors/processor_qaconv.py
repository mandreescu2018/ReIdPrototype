import torch
from torch import amp
from.processor_standard import ProcessorStandard
from models.QAConv import QAConv
from loss import TripletLossMatcher

class ProcessorQaconv(ProcessorStandard):
    def __init__(self, cfg, 
                 model, 
                 train_loader, 
                 val_loader, 
                 optimizer=None, 
                 optimizer_center=None, 
                 center_criterion=None, 
                 loss_fn=None, 
                 scheduler=None, 
                 start_epoch=0):
        super().__init__(cfg, 
                         model, 
                         train_loader, 
                         val_loader, 
                         optimizer, 
                         optimizer_center, 
                         center_criterion, 
                         loss_fn, 
                         scheduler, 
                         start_epoch)
        
        num_features = self.model.num_features
        final_layer_factor = 16
        # self.config.INPUT.SIZE_TRAIN
        # cfg.INPUT.SIZE_TRAIN = (cfg.INPUT.SIZE_TRAIN[0] // final_layer_factor, cfg.INPUT.SIZE_TRAIN[1] // final_layer_factor)

        hei = self.config.INPUT.SIZE_TRAIN[0] // final_layer_factor
        wid = self.config.INPUT.SIZE_TRAIN[1] // final_layer_factor

        matcher = QAConv(num_features, hei, wid).to(self.device)
        
        self.loss_fn = TripletLossMatcher()
        self.loss_fn.matcher = matcher
        

    
    def train_step(self):
        self.model.eval()
        self.loss_fn.train()
        for n_iter, batch in enumerate(self.train_loader):
            self.zero_grading()
            
            inputs = tuple(batch[i].to(self.device) for i in self.config.INPUT.TRAIN_KEYS)
            target = batch[1].to(self.device)
            
            with amp.autocast(self.device):
                outputs = self.model(*inputs)
                loss, acc = self.loss_fn(outputs, target)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.optimizer_center is not None:
                for param in self.center_criterion.parameters():
                    center_loss_item = next((item for item in self.loss_fn.loss_functions if "CenterLoss" in item.__class__.__name__), None)
                    if center_loss_item is not None:
                        center_weight = center_loss_item.weight
                    else:
                        raise ValueError("CenterLoss not found in loss functions")
                    param.grad.data *= (1. / center_weight)
                self.scaler.step(self.optimizer_center)
                self.scaler.update()
            
            self.live_values.update(loss, outputs, target, accuracy=acc)
            
            torch.cuda.synchronize()
            self.log_training_details(n_iter)
