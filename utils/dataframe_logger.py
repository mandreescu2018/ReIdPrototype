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

    def log_training(self, epoch, step, loss, accuracy=None, lr=None, **kwargs):
        """
        Log training data for the current step or epoch.

        Args:
            epoch (int): Current epoch number.
            step (int): Current step number (or iteration).
            loss (float): Current loss value.
            accuracy (float, optional): Current accuracy value.
            lr (float, optional): Current learning rate.
            kwargs: Additional metrics to log.
        """
        entry = {
            "epoch": epoch,
            "step": step,
            "loss": loss.item(),
            "accuracy": accuracy,
            "learning_rate": lr,
        }
        entry.update(kwargs)  # Add any additional metrics
        df = pd.DataFrame(entry, index=[0])
        # self.data.append(entry)
        df.to_csv(self.training_save_path, mode='a', header=False)

    def log_validation(self, map, cmc):
        """
        Log validation data for the current epoch.

        Args:
            epoch (int): Current epoch number.
            loss (float): Current loss value.
            accuracy (float, optional): Current accuracy value.
            kwargs: Additional metrics to log.
        """
        entry = {
            "map": map,
            "rank_1": cmc[0],
            "rank_5": cmc[1],
            "rank_10": cmc[2],
            "rank_20": cmc[3],
            
        }
        df = pd.DataFrame(entry, index=[0])
        df.to_csv(self.validation_save_path, mode='a', header=False)

    def save(self):
        """
        Save the logged data to a .csv file.
        """
        df = pd.DataFrame(self.data)
        df.to_csv(self.save_path, index=False)
        print(f"Training log saved to {self.save_path}")
