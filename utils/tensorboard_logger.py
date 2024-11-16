from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

class TensoboardLogger:
    def __init__(self, log_dir):
        tensorboar_dir = Path(log_dir) / 'tensorboard'        
        tensorboar_dir.mkdir(exist_ok=True, parents=True)       
        self.writer = SummaryWriter(tensorboar_dir)

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