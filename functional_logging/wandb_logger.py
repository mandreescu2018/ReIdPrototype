import wandb
from .base_logging import BaseLogger

class WandbLogger(BaseLogger):
    def __init__(self, cfg):
        self.config = cfg
        # start a new wandb run 
        wandb.init(
            # set the wandb project where this run will be logged
            project=self.config.LOGGING.WANDB_PROJECT,
            name=self.config.LOGGING.WANDB_NAME,
            resume="must" if self.config.MODEL.PRETRAIN_CHOICE == 'resume' else "allow",
            id=self.config.LOGGING.WANDB_RUN_ID,

            # track hyperparameters
            config={
            "learning_rate": self.config.SOLVER.BASE_LR,
            "architecture": self.config.MODEL.NAME, 
            "dataset": self.config.DATASETS.NAMES,
            "epochs": self.config.SOLVER.MAX_EPOCHS,
            },
        )
    
    def on_epoch_end(self, live_values):
        wandb.log({
            "Loss": live_values.loss_meter.avg,
            "Accuracy": live_values.acc_meter.avg,
            "Learning Rate": live_values.learning_rate,
        })
    
    
    def log_validation(self, live_values):
        pass
