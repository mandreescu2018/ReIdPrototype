import wandb

class WandbLogger:
    def __init__(self, cfg):
        self.config = cfg
        # start a new wandb run 
        wandb.init(
            # set the wandb project where this run will be logged
            project=self.config.WANDB.PROJECT,
            name=self.config.WANDB.NAME,
            resume= "must" if self.config.MODEL.PRETRAIN_CHOICE == 'resume' else "allow",
            id=self.config.WANDB.RUN_ID,

            # track hyperparameters
            config={
            "learning_rate": self.config.SOLVER.BASE_LR,
            "architecture": self.config.MODEL.NAME, 
            "dataset": self.config.DATASETS.NAMES,
            "epochs": self.config.SOLVER.MAX_EPOCHS,
            },
        )
    
    def log_results(self, loss, acc, optimizer):
        wandb.log({
            "Loss": loss,
            "Accuracy": acc,
            "Learning Rate": optimizer.param_groups[0]['lr'],
        })