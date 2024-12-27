import os
from pathlib import Path
import pandas as pd

class DataFrameLogger:
    def __init__(self, save_path="logs"):
        """
        Initialize the logger.
        
        Args:
            save_path (str): Path to save the .csv file.
        """
        self.training_save_path = Path(save_path)/"training_log.csv"
        self.validation_save_path = Path(save_path)/"validation_log.csv"
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

    
    def log_validation(self, epoch, map, cmc):
        """
        Log validation data.

        Args:
            map (float): Current epoch number.
            cmc (float): Cumulative matching curve list.            
        """
        entry = {
            "epoch": [epoch],
            "map": [map],
            "rank_1": [cmc[0]],
            "rank_5": [cmc[4]],
            "rank_10": [cmc[9]],
            "rank_20": [cmc[19]],
            
        }
        df = pd.DataFrame(entry)
        self.append_to_csv(df, self.validation_save_path)

    def save(self):
        """
        Save the logged data to a .csv file.
        """
        df = pd.DataFrame(self.data)
        df.to_csv(self.save_path, index=False)
        print(f"Training log saved to {self.save_path}")
