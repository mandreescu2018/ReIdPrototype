import torch
from torch import amp
from.processor_base import ProcessorBase
import time
import os

class ProcessorPrototype(ProcessorBase):

    def train(self):
         super(ProcessorPrototype, self).train()         
         self.scaler = amp.GradScaler(self.device)

         for epoch in range(self.start_epoch+1, self.epochs+1):
                self.reset_metrics()
                self.scheduler.step(epoch)
                start_time = time.time()
                self.current_epoch = epoch
                self.train_step()
                self.on_epoch_end(start_time)
                self.log_to_wandb()
                self.dump_metrics_data_to_tensorboard()
                if epoch % self.config.SOLVER.EVAL_PERIOD == 0 or epoch == 1:
                    self.validation_step()
                if epoch % self.config.SOLVER.CHECKPOINT_PERIOD == 0:
                    self.save_model_for_resume(os.path.join(self.config.OUTPUT_DIR, self.config.MODEL.NAME + '_resume_{}.pth'.format(epoch))) 

    def train_step(self):
        super(ProcessorPrototype, self).train_step()
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
            
            def calculate_accuracy(outputs, target):
                index = self.config.LOSS.ID_LOSS_OUTPUT_INDEX if isinstance(outputs, tuple) else 0
                id_classifier_output = outputs[index]
                id_hat_element = id_classifier_output[0] if isinstance(id_classifier_output, list) else id_classifier_output
                acc = (id_hat_element.max(1)[1] == target).float().mean()

                return acc

            acc = calculate_accuracy(outputs, target)            

            self.loss_meter.update(loss.item(), self.train_loader.batch_size)
            self.acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            self.log_training_details(n_iter)
            

    
        

    


        
        