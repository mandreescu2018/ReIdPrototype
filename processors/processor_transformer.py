import logging
import torch
from torch import amp
from.processor_base import ProcessorBase
import time
import os

class ProcessorTransformer(ProcessorBase):

    def train(self):
         super(ProcessorTransformer, self).train()
         self.logger = logging.getLogger("ReIDPrototype.train")
         self.logger.info('Start training')
         self.scaler = amp.GradScaler(self.device)

         for epoch in range(self.start_epoch+1, self.epochs+1):
                # self.evaluator.reset()
                # self.loss_meter.reset()
                # self.acc_meter.reset()
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
                # if epoch % self.checkpoint_period == 0:
                    self.save_model_for_resume(os.path.join(self.config.OUTPUT_DIR, self.config.MODEL.NAME + '_resume_{}.pth'.format(epoch))) 

    def zero_grading(self):
        self.optimizer.zero_grad()
        self.optimizer_center.zero_grad()

    def train_step(self):
        super(ProcessorTransformer, self).train_step()
        for n_iter, (img, pid, target_cam, target_view, _) in enumerate(self.train_loader):
            self.zero_grading()
            img = img.to(self.device)
            target = pid.to(self.device)
            target_cam = target_cam.to(self.device)
            target_view = target_view.to(self.device)

            with amp.autocast(self.device):
                score, feat = self.model(img, target, cam_label=target_cam, view_label=target_view)
                loss = self.loss_fn(score, feat, target, target_cam)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            score_element = score[0] if isinstance(score, list) else score
            acc = (score_element.max(1)[1] == target).float().mean()

            self.loss_meter.update(loss.item(), img.shape[0])
            self.acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            self.log_training_details(n_iter)
            

    def validation_step(self):
         
        super(ProcessorTransformer, self).validation_step()

        for n_iter, (img, pid, camid, camids, target_view, _) in enumerate(self.val_loader):
            
            with torch.no_grad():
                img = img.to(self.device)
                camids = camids.to(self.device)
                target_view = target_view.to(self.device)
                outputs = self.model(img)
                self.evaluator.update((outputs, pid, camid))
        
        cmc, mAP, _, _, _, _, _ = self.evaluator.compute()
        self.logger.info("Validation Results - Epoch: {}".format(self.current_epoch))
        self.logger.info("mAP: {:.3%}".format(mAP))
        for r in [1, 5, 10, 20]:
            self.logger.info("CMC curve, Rank-{:<3}:{:.3%}".format(r, cmc[r - 1]))
        torch.cuda.empty_cache()
        
        self.evaluator.reset()

        return cmc, mAP
        

    


        
        