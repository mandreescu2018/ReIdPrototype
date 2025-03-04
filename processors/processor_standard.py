import torch
from torch import amp
from.processor_base import ProcessorBase
import time
import os

class ProcessorStandard(ProcessorBase):

    def train(self):
         super(ProcessorStandard, self).train()         
         self.scaler = amp.GradScaler(self.device)

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
                    self.save_model_for_resume(os.path.join(self.config.OUTPUT_DIR, self.config.MODEL.NAME + '_resume_{}.pth'.format(epoch))) 

    def train_step(self):
        self.model.train()
        for n_iter, batch in enumerate(self.train_loader):
            self.zero_grading()
            
            inputs = tuple(batch[i].to(self.device) for i in self.config.INPUT.TRAIN_KEYS)
            target = batch[1].to(self.device)
            
            with amp.autocast(self.device):
                outputs = self.model(*inputs)
                loss = self.loss_fn(outputs, target)
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
            
            self.live_values.update(loss, outputs, target)
            
            torch.cuda.synchronize()
            self.log_training_details(n_iter)
            

    
        

    


        
        