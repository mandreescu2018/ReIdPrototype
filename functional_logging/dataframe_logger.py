import os
from pathlib import Path
import pandas as pd
from .base_logging import BaseLogger

class DataFrameLogger(BaseLogger):
    def __init__(self, cfg):
        """
        Initialize the logger.
        
        Args:
            save_path (str): Path to save the .csv file.
        """

        self.training_save_path = Path(cfg.OUTPUT_DIR)/"training_log.csv"
        self.validation_save_path = Path(cfg.OUTPUT_DIR)/"validation_log.csv"
        if os.path.isfile(self.training_save_path):
            with open(self.training_save_path, "w") as f:            
                f.truncate()
        if os.path.isfile(self.validation_save_path):
            with open(self.validation_save_path, "w") as f:            
                f.truncate()
        
        self.data = []  # List to store training logs

    def log_training(self, epoch, loss, accuracy=None, lr=None, **kwargs):
        """
        Log training data for the current step or epoch.

        Args:
            epoch (int): Current epoch number.
            loss (float): Current loss value.
            accuracy (float, optional): Current accuracy value.
            lr (float, optional): Current learning rate.
            kwargs: Additional metrics to log.
        """
        entry = {
            "epoch": [epoch],
            "loss": [loss],
            "accuracy": [accuracy],
            "learning_rate": [lr],
        }
        entry.update(kwargs)  # Add any additional metrics
        df = pd.DataFrame(entry)
        self.append_to_csv(df, self.training_save_path)


    def append_to_csv(self, dataframe, filename):
        
        # Check if file exists
        file_exists = os.path.isfile(filename)

        # Write with header only if file does not exist
        dataframe.to_csv(filename, mode='a', header=not file_exists, index=False)

    
    def log_validation(self, live_values):
        """
        Log validation data.

        Args:
            map (float): Current epoch number.
            cmc (float): Cumulative matching curve list.            
        """
        entry = {
            "epoch": [live_values.current_epoch],
            "map": [live_values.mAP]                        
        }
        cmc_curve = {f"rank_{i+1}": [live_values.cmc[i]] for i in range((len(live_values.cmc)))}
        entry.update(cmc_curve) 
        df = pd.DataFrame(entry)
        self.append_to_csv(df, self.validation_save_path)

    def on_epoch_end(self, live_values):
        """
        Log epoch end data.

        Args:
            live_values (LiveValues): Object containing live values.
        """
        self.log_training(
            live_values.current_epoch,
            live_values.loss_meter.avg,
            live_values.acc_meter.avg,
            live_values.learning_rate,
        )
    
    
    
    

