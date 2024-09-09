import os
import time
import torch
import torch.nn as nn
from torch.cuda import amp
import logging
from utils import AverageMeter
from utils.metrics import R1_mAP_eval
import wandb
# from utils import Saver


class ProcessorBase:
    def __init__(self, cfg,                   
                 model,                                   
                 train_loader,
                 val_loader,                  
                 optimizer=None,
                 optimizer_center=None,
                 center_criterion=None,                 
                 loss_fn=None,
                 scheduler=None):
        self.config = cfg
        self.model = model        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.optimizer_center = optimizer_center
        self.center_criterion = center_criterion
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.epochs = cfg.SOLVER.MAX_EPOCHS
        self.current_epoch = 0
        self.device = cfg.DEVICE
        self.init_meters()
        self.evaluator = R1_mAP_eval(cfg.DATASETS.NUMBER_OF_IMAGES_IN_QUERY, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        self.eval_period = cfg.SOLVER.EVAL_PERIOD
        self.checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
        if optimizer is not None:
            self.set_wandb()
        
        
    def init_meters(self):
        self.acc_meter = AverageMeter()
        self.loss_meter = AverageMeter()
    
    def reset_metrics(self):
        self.acc_meter.reset()
        self.loss_meter.reset()
        self.evaluator.reset()
    
    def train(self):
        self.model.to(self.device)

    def train_step(self):
        self.model.train()

    def validation_step(self):
        self.model.eval()
    
    def inference(self):        
        self.evaluator.reset()
        self.model.to(self.device)
        self.model.eval()
        for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(self.val_loader):
            with torch.no_grad():
                img = img.to(self.device)
                camids = camids.to(self.device)
                target_view = target_view.to(self.device)
                feat = self.model(img, cam_label=camids, view_label=target_view)
                self.evaluator.update((feat, pid, camid))
        # for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(self.val_loader):
        #     with torch.no_grad():
        #         img = img.to(self.device)
        #         camids = camids.to(self.device)
        #         target_view = target_view.to(self.device)
        #         feat = self.model(img, cam_label=camids, view_label=target_view)
        #         self.evaluator.update((feat, pid, camid))
        cmc, mAP, _, _, _, _, _ = self.evaluator.compute()
        print("Inference Results ")
        print("mAP: {:.3%}".format(mAP))
        for r in [1, 5, 10, 20]:
            print("CMC curve, Rank-{:<3}:{:.3%}".format(r, cmc[r - 1]))

    def log_training_details(self, n_iter):
        if (n_iter + 1) % self.config.SOLVER.LOG_PERIOD == 0:
            status_msg = f"Epoch[{self.current_epoch}] "
            status_msg += f"Iteration[{n_iter + 1}/{len(self.train_loader)}] "
            status_msg += f"Loss: {self.loss_meter.avg:.3f}, "
            status_msg += f"Acc: {self.acc_meter.avg:.3f}, "
            status_msg += f"Base Lr: {self.optimizer.param_groups[0]['lr']:.2e}"
            self.logger.info(status_msg)
    

    def on_epoch_end(self, start_time):
        time_per_batch = (time.time() - start_time) / len(self.train_loader)
        self.logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
            .format(self.current_epoch, time_per_batch, self.train_loader.batch_size / time_per_batch))
        
    def log_to_wandb(self):
        wandb.log({
            "Loss": self.loss_meter.avg,
            "Accuracy": self.acc_meter.avg,
            "Learning Rate": self.optimizer.param_groups[0]['lr']
        })
    def set_wandb(self):
       # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="reid-prototype",

            # track hyperparameters and run metadata
            config={
            "learning_rate": self.config.SOLVER.BASE_LR,
            "architecture": self.config.MODEL.NAME,
            "dataset": self.config.DATASETS.NAMES,
            "epochs": self.epochs,
            },
        )
    def save_model_for_resume(self,
                          path: str):
        torch.save({
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                }, path)
    

