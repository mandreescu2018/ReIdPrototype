from .base_logging import BaseLogger
from .tensorboard_logger import TensorboardLogger
from .stream_logger import StreamLogger
from .wandb_logger import WandbLogger
from .dataframe_logger import DataFrameLogger

class CompositeLogger(BaseLogger):
    def __init__(self, cfg):
        self.config = cfg
        self.loggers = []
        self.load_loggers()

    def load_loggers(self):
        self.add_logger(StreamLogger(self.config))
        self.add_logger(DataFrameLogger(self.config))
        if self.config.LOGGING.TENSORBOARD_USE:            
            self.add_logger(TensorboardLogger(self.config.OUTPUT_DIR))
        if self.config.LOGGING.WANDB_USE:
            self.add_logger(WandbLogger(self.config))

    def on_epoch_end(self, live_values):
        for logger in self.loggers:
            logger.on_epoch_end(live_values)
    
    def add_logger(self, logger):
        self.loggers.append(logger)

    
    def info(self, message):
        for logger in self.loggers:
            if 'info' in dir(logger): 
                logger.info(message)

    def log_validation(self, live_values):
        for logger in self.loggers:
            logger.log_validation(live_values)