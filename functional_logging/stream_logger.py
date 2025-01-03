import logging
import os
import sys
from .base_logging import BaseLogger
import time

class StreamLogger(BaseLogger):
    def __init__(self, cfg, if_train=True):
        self.logger_name = cfg.MODEL.NAME + '_' + cfg.DATASETS.NAMES
        self.save_dir = cfg.OUTPUT_DIR  
        self.config = cfg

        if self.logger_name not in logging.Logger.manager.loggerDict.keys():
            self.logger = self.setup_logger(if_train)
        else:
            self.logger = logging.getLogger(self.logger_name)
        
             
        self.logger.setLevel(logging.DEBUG)

    def setup_logger(self, if_train):
        logger = logging.getLogger(self.logger_name)
        logger.setLevel(logging.DEBUG)

        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if self.save_dir:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            if if_train:
                fh = logging.FileHandler(os.path.join(self.save_dir, "train_log.txt"), mode='w')
            else:
                fh = logging.FileHandler(os.path.join(self.save_dir, "test_log.txt"), mode='w')
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        return logger
    
    def info(self, message):
        """Log info message."""
        self.logger.info(message)
    
    def on_epoch_end(self, live_values):
        """Log epoch end data."""
        time_per_batch = (time.time() - live_values.current_start_time) / live_values.train_loader_length
        speed = self.config.SOLVER.IMS_PER_BATCH / time_per_batch
        self.logger.info(f"Epoch {live_values.current_epoch} done. Time per batch: {time_per_batch:.3f}[s] Speed: {speed:.1f}[samples/s]")
    
    def log_validation(self, live_values):
        """Log validation info."""
        self.logger.info("Validation Results - Epoch: {}".format(live_values.current_epoch))
        self.logger.info("mAP: {:.3%}".format(live_values.mAP))
        for r in [1, 5, 10, 20]:
            self.logger.info("CMC curve, Rank-{:<3}:{:.3%}".format(r, live_values.cmc[r - 1]))