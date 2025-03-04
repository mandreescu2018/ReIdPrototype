import torch
from utils.device_manager import DeviceManager
from .metrics_values import MetricsLiveValues
from functional_logging import CompositeLogger

    
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
        self.start_epoch = start_epoch

        self.max_epochs = cfg.SOLVER.MAX_EPOCHS
        self.device = DeviceManager.get_device().type
        self.live_values = MetricsLiveValues(cfg)
        self.live_values.train_loader_length = len(train_loader)
        self.composite_logger = CompositeLogger(cfg)
    
    def train(self):
        self.composite_logger.info('Start training')

    def train_step(self):
        pass
    
    def model_evaluation(self):
        self.model.eval()
        for n_iter, batch in enumerate(self.val_loader):
            with torch.no_grad():

                pid = batch[self.config.INPUT.PERSON_ID_KEY]
                camid = batch[self.config.INPUT.CAMERA_ID_KEY]
                inputs = []
                for item in self.config.INPUT.EVAL_KEYS:
                    if item != 'NaN':
                        inputs.append(batch[item].to(self.device))
                    else:
                        inputs.append(None)
                
                outputs = self.model(*inputs)
                feat = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
                self.live_values.evaluator.update((feat, pid, camid))
        
        cmc, mAP, _, _, _, _, _ = self.live_values.evaluator.compute()
        
        self.live_values.mAP = mAP
        self.live_values.cmc = cmc

        return cmc, mAP

    def validation_step(self):
        cmc, mAP = self.model_evaluation()
        
        self.composite_logger.log_validation(self.live_values)
        torch.cuda.empty_cache()
        self.live_values.reset_metrics()

    def zero_grading(self):
        self.optimizer.zero_grad()
        if self.optimizer_center is not None:
            self.optimizer_center.zero_grad()

    def inference(self):        
        self.live_values.reset_metrics()
        cmc, mAP = self.model_evaluation()
       
        self.composite_logger.info('Inference Results')
        self.composite_logger.log_validation(self.live_values)

    # LOGGING
    def on_epoch_end(self):
        """Log epoch end data and send to tensorboard and wandb."""
        self.live_values.learning_rate = self.optimizer.param_groups[0]['lr']      
        self.composite_logger.on_epoch_end(self.live_values)  
    
    
    def log_training_details(self, n_iter):
        if (n_iter + 1) % self.config.SOLVER.LOG_PERIOD == 0:
            status_msg = f"Epoch[{self.live_values.current_epoch}] "
            status_msg += f"Iteration[{n_iter + 1}/{len(self.train_loader)}] "
            status_msg += f"Loss: {self.live_values.loss_meter.avg:.3f}, "
            status_msg += f"Acc: {self.live_values.acc_meter.avg:.3f}, "
            status_msg += f"Base Lr: {self.optimizer.param_groups[0]['lr']:.2e}"
            self.composite_logger.info(status_msg)
    
    # SAVE MODEL
    def save_model_for_resume(self,
                          path: str):
        torch.save({
                'epoch': self.live_values.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'center_criterion_state_dict': self.center_criterion.state_dict() if self.center_criterion is not None else None,
                'optimizer_center_state_dict': self.optimizer_center.state_dict() if self.optimizer_center is not None else None,
                'scheduler_state_dict': self.scheduler.state_dict(),
                }, path)
    