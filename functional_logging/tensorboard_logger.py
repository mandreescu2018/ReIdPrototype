from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from .base_logging import BaseLogger

class TensorboardLogger(BaseLogger):
    def __init__(self, log_dir):
        tensorboard_dir = Path(log_dir) / 'tensorboard'        
        tensorboard_dir.mkdir(exist_ok=True, parents=True)       
        self.writer = SummaryWriter(tensorboard_dir)

    def dump_metric_tb(self, value: float, epoch: int, m_type: str, m_desc: str):
        self.writer.add_scalar(f'{m_type}/{m_desc}', value, epoch)
        
    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def log_image(self, tag, image, step):
        self.writer.add_image(tag, image, step)

    def log_histogram(self, tag, values, step):
        self.writer.add_histogram(tag, values, step)

    def close(self):
        self.writer.close()
    
    def on_epoch_end(self, live_values):
        pass
        # """Send metrics data to tensorboard."""
        self.dump_metric_tb(live_values.loss_meter.avg, live_values.current_epoch, f'identity', f'loss')        
        self.dump_metric_tb(live_values.acc_meter.avg, live_values.current_epoch, f'identity', f'acc')
        self.dump_metric_tb(live_values.learning_rate, live_values.current_epoch, f'identity', f'lr')

    def log_validation(self, live_values):
        self.dump_metric_tb(live_values.mAP, live_values.current_epoch, f'ReID', f'mAP')
        self.dump_metric_tb(live_values.cmc[0], live_values.current_epoch, f'ReID', f'cmc1')

    
