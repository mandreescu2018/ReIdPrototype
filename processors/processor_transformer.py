import logging
import torch
from torch import amp
from.processor_base import ProcessorBase
import time
import os

class ProcessorTransformer(ProcessorBase):

    def train(self):
         super(ProcessorTransformer, self).train()         
         self.scaler = amp.GradScaler(self.device)

         for epoch in range(self.start_epoch+1, self.epochs+1):
                self.reset_metrics()
                self.scheduler.step(epoch)
                start_time = time.time()
                self.current_epoch = epoch
                self.train_step()
                self.on_epoch_end(start_time)
                self.log_to_wandb()
                if epoch % self.config.SOLVER.EVAL_PERIOD == 0 or epoch == 1:
                    self.validation_step()
                if epoch % self.config.SOLVER.CHECKPOINT_PERIOD == 0:
                    self.save_model_for_resume(os.path.join(self.config.OUTPUT_DIR, self.config.MODEL.NAME + '_resume_{}.pth'.format(epoch))) 

    def train_step(self):
        super(ProcessorTransformer, self).train_step()
        for n_iter, (img, pid, target_cam, target_view) in enumerate(self.train_loader):
            self.zero_grading()
            img = img.to(self.device)
            target = pid.to(self.device)
            target_cam = target_cam.to(self.device)
            target_view = target_view.to(self.device)

            with amp.autocast(self.device):
                score, feat = self.model(img, target, cam_label=target_cam, view_label=target_view)
                loss = self.loss_fn(score, feat, target, target_cam)
                # if self.optimizer_center is not None:
                #     center_loss = self.center_criterion(feat, target)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if self.optimizer_center is not None:
                for param in self.center_criterion.parameters():
                    param.grad.data *= (1. / self.config.LOSS.CENTER_LOSS_WEIGHT)
                self.scaler.step(self.optimizer_center)
                self.scaler.update()
            

            score_element = score[0] if isinstance(score, list) else score
            acc = (score_element.max(1)[1] == target).float().mean()

            self.loss_meter.update(loss.item(), img.shape[0])
            self.acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            self.log_training_details(n_iter)
            

    
        

    


        
        