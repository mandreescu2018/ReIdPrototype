import os
import time
import torch
import torch.nn as nn
from torch.cuda import amp
import logging
from utils import AverageMeter, WandbLogger, DataFrameLogger
from utils.metrics import R1_mAP_eval
from utils.tensorboard_logger import TensoboardLogger
from utils.device_manager import DeviceManager

class ModelInputProcessor:
    def __init__(self, cfg):
        """
        Initializes the ModelInputProcessor with configuration for different inputs.
        
        Args:
            cfg config object: Configuration for model inputs.
        """
        self.config = cfg
        self.personid_key = cfg.PROCESSOR.TARGET_KEY
        self.batch = None

    def target(self):
        """
        Returns the person ID.
        
        Args:
            input batch 
        
        Returns:
            tensor: person ID.
        """
        return self.batch[self.personid_key]
    
    def process(self, batch):
        """
        Processes the batch based on the configuration.
        
        Args:
            input batch 
        
        Returns:
            tuple: Processed inputs formatted as required by the model.
        """
        self.batch = batch
        inputs = []
        for key in range(self.config.PROCESSOR.INPUT_KEYS):
            inputs.append(batch[key])
        return tuple(inputs), self.target()

class ProcessorBase:
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
        self.start_epoch = start_epoch
        self.device = DeviceManager.get_device().type
        self.init_meters()
        self.evaluator = R1_mAP_eval(cfg.DATASETS.NUMBER_OF_IMAGES_IN_QUERY, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        self.input_processor = ModelInputProcessor(cfg)
        if self.config.WANDB.USE and not self.config.MODEL.PRETRAIN_CHOICE == 'test':
            self.wlogger = WandbLogger(cfg)
        self.tensorboard_logger = TensoboardLogger(cfg.OUTPUT_DIR)
        self.dataframe_logger = DataFrameLogger(cfg.OUTPUT_DIR)
                
    def init_meters(self):
        self.acc_meter = AverageMeter()
        self.loss_meter = AverageMeter()
    
    def reset_metrics(self):
        self.acc_meter.reset()
        self.loss_meter.reset()
        self.evaluator.reset()    
    
    def train(self):
        self.logger = logging.getLogger("ReIDPrototype.train")
        self.logger.info('Start training')

    def train_step(self):
        self.model.train()
    
    def model_evaluation(self):
        self.model.eval()
        for n_iter, batch in enumerate(self.val_loader):
            with torch.no_grad():

                pid = batch[self.config.PROCESSOR.TARGET_KEY]
                camid = batch[self.config.DATALOADER.BATCH_CAM_INDEX]
                inputs = []
                for item in self.config.INPUT.EVAL_KEYS:
                    if item != 'NaN':
                        inputs.append(batch[item].to(self.device))
                    else:
                        inputs.append(None)
                
                outputs = self.model(*inputs)
                feat = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
                self.evaluator.update((feat, pid, camid))
        
        cmc, mAP, _, _, _, _, _ = self.evaluator.compute()

        return cmc, mAP

    def validation_step(self):
        cmc, mAP = self.model_evaluation()
        
        self.logger.info("Validation Results - Epoch: {}".format(self.current_epoch))
        self.logger.info("mAP: {:.3%}".format(mAP))
        for r in [1, 5, 10, 20]:
            self.logger.info("CMC curve, Rank-{:<3}:{:.3%}".format(r, cmc[r - 1]))
        torch.cuda.empty_cache()
        self.dataframe_logger.log_validation(mAP, cmc)
        self.evaluator.reset()

        # return cmc, mAP
    
    def zero_grading(self):
        self.optimizer.zero_grad()
        if self.optimizer_center is not None:
            self.optimizer_center.zero_grad()

    
    def inference(self):        
        self.evaluator.reset()
        cmc, mAP = self.model_evaluation()
       
        print("Inference Results ")
        print(f"mAP: {mAP:.3%}")
        for r in [1, 5, 10, 20]:
            print(f"CMC curve, Rank-{r:<3}:{cmc[r - 1]:.3%}")


    # LOGGING
    def on_epoch_end(self, start_time):
        """Log epoch end data and send to tensorboard and wandb."""
        self.log_epoch_end_data(start_time)        
        self.log_to_wandb()
        self.dump_metrics_data_to_tensorboard()
        self.log_training_to_dataframe()
    
    def log_epoch_end_data(self, start_time):
        """Log epoch end data."""
        time_per_batch = (time.time() - start_time) / len(self.train_loader)
        speed = self.train_loader.batch_size / time_per_batch
        self.logger.info(f"Epoch {self.current_epoch} done. Time per batch: {time_per_batch:.3f}[s] Speed: {speed:.1f}[samples/s]")
        
    def dump_metrics_data_to_tensorboard(self):
        """Send metrics data to tensorboard."""
        self.tensorboard_logger.dump_metric_tb(self.loss_meter.avg, self.current_epoch, f'losses', f'loss')        
        self.tensorboard_logger.dump_metric_tb(self.acc_meter.avg, self.current_epoch, f'losses', f'acc')
        self.tensorboard_logger.dump_metric_tb(self.optimizer.param_groups[0]['lr'], self.current_epoch, f'losses', f'lr')
    
    def log_training_to_dataframe(self):
        self.dataframe_logger.log_training(self.current_epoch, 
                                           self.loss_meter.avg, 
                                           self.acc_meter.avg, 
                                           self.optimizer.param_groups[0]['lr'])

    def log_to_wandb(self):
        """Send logging data to weights and biases."""
        if not self.config.WANDB.USE:
            return
        self.wlogger.log_results(self.loss_meter.avg, self.acc_meter.avg, self.optimizer)
    
    
    def log_training_details(self, n_iter):
        if (n_iter + 1) % self.config.SOLVER.LOG_PERIOD == 0:
            status_msg = f"Epoch[{self.current_epoch}] "
            status_msg += f"Iteration[{n_iter + 1}/{len(self.train_loader)}] "
            status_msg += f"Loss: {self.loss_meter.avg:.3f}, "
            status_msg += f"Acc: {self.acc_meter.avg:.3f}, "
            status_msg += f"Base Lr: {self.optimizer.param_groups[0]['lr']:.2e}"
            self.logger.info(status_msg)
    
    # SAVE MODEL
    def save_model_for_resume(self,
                          path: str):
        torch.save({
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'center_criterion_state_dict': self.center_criterion.state_dict() if self.center_criterion is not None else None,
                'optimizer_center_state_dict': self.optimizer_center.state_dict() if self.optimizer_center is not None else None,
                'scheduler_state_dict': self.scheduler.state_dict(),
                }, path)
    

