import torch
from torch import amp
from .processor_base import ProcessorBase
import time
import os
from config.constants import *
# from loss.patch_memory import Patchloss

class ProcessorPat(ProcessorBase):
    # def __init__(self, config, model, train_loader, val_loader, optimizer, scheduler, loss_fn, device, **kwargs):
    #     super(ProcessorPat, self).__init__(config, model, train_loader, val_loader, optimizer=optimizer, loss_fn=loss_fn, scheduler=scheduler, **kwargs)
    #     self.patch_centers = kwargs.get("patch_centers", None)
    #     self.pc_criterion = kwargs.get("pc_criterion", None)

    def _initialize_centers(self):
        self.model.train()
        for i, batch in enumerate(self.train_loader):
            # measure data loading time
            with torch.no_grad():
                #input = input.cuda(non_blocking=True)
                input = batch[IMG_INDEX].cuda(non_blocking=True)
                pid = batch[PID_INDEX]
                camid = batch[CAMID_INDEX]
                path = batch[PATH_INDEX]
                #input = input.view(-1, input.size(2), input.size(3), input.size(4))

                # compute output
                _, _, layerwise_feat_list = self.model(input)
                self.patch_centers.get_soft_label(path, layerwise_feat_list[-1], pid=pid, camid=camid)
        print('initialization done')
    
    def train(self):
        super(ProcessorPat, self).train()         
        self.scaler = amp.GradScaler(self.device)
        
        self._initialize_centers()  # Initialize patch centers
        
        for epoch in range(self.start_epoch+1, self.max_epochs+1):
            
            self.live_values.reset_metrics()

            self.scheduler.step(epoch)

            self.live_values.current_start_time = time.time()
            self.live_values.current_epoch = epoch

            self.train_step()       
            self.on_epoch_end()
            
            if epoch % self.config.SOLVER.EVAL_PERIOD == 0 or epoch == 1:
                self.validation_step()
            if epoch % self.config.SOLVER.CHECKPOINT_PERIOD == 0:
                self.save_model(os.path.join(self.config.OUTPUT_DIR, self.config.MODEL.NAME + '_resume_{}.pth'.format(epoch))) 
                

    def train_step(self):
        self.model.train()
        for n_iter, batch in enumerate(self.train_loader):
            self.zero_grading()
            
            # inputs = tuple(batch[i].to(self.device) for i in self.config.INPUT.TRAIN_KEYS)
            img = batch[IMG_INDEX].to(self.device)
            pid = batch[PID_INDEX].to(self.device)
            camid = batch[CAMID_INDEX].to(self.device)
            target = batch[PID_INDEX].to(self.device)
            
            # _C.MODEL.PC_LR = 1.0
            # score, layerwise_global_feat, layerwise_feat_list = model(img)
            # reid_loss = loss_fn(score, layerwise_global_feat[-1], target, all_posvid=all_posvid, 
            # soft_label=cfg.MODEL.SOFT_LABEL, soft_weight=cfg.MODEL.SOFT_WEIGHT, soft_lambda=cfg.MODEL.SOFT_LAMBDA)
            # part views

            # _C.MODEL.PC_SCALE = 0.02
            # _C.MODEL.PC_LOSS = False
            # _C.MODEL.PC_LR = 1.0
            with amp.autocast(self.device):
                # outputs = self.model(*inputs)
                score, layerwise_global_feat, layerwise_feat_list = self.model(img)
                # loss = self.loss_fn(outputs, target)
                patch_agent, position = self.patch_centers.get_soft_label(batch[PATH_INDEX], 
                                                                          layerwise_feat_list[-1], 
                                                                          pid=pid, 
                                                                          camid=camid)
                l_ploss = self.config.MODEL.PC_LR
                feat = torch.stack(layerwise_feat_list[-1], dim=0)
                '''
                loss1: clustering loss(for patch centers)
                '''
                ploss, all_posvid = self.pc_criterion(feat, 
                                                      patch_agent, 
                                                      position, 
                                                      self.patch_centers, 
                                                      vid=target, 
                                                      camid=camid)
                '''
                loss2: reid-specific loss
                (ID + Triplet loss)
                '''
                reid_loss = self.loss_fn([score, 
                                         layerwise_global_feat[-1]], 
                                         target)
                total_loss = reid_loss + l_ploss * ploss

            self.scaler.scale(total_loss).backward()
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
            
            self.live_values.update(total_loss, score, target)
            
            torch.cuda.synchronize()
            self.log_training_details(n_iter)
            

    
        

    


        
        