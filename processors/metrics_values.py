from utils import AverageMeter
from utils.metrics import R1_mAP_eval

class MetricsLiveValues:
    def __init__(self, cfg):
        self.config = cfg
        self.acc_meter = AverageMeter()
        self.loss_meter = AverageMeter()
        self.evaluator = R1_mAP_eval(cfg.DATASETS.NUMBER_OF_IMAGES_IN_QUERY, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        self._current_epoch = 0
        self._learning_rate = 0
        self._train_loader_length = 0
        self._current_start_time = None
        self.mAP = 0
        self.cmc = None
    
    @property
    def current_epoch(self):
        return self._current_epoch  

    @current_epoch.setter
    def current_epoch(self, value):
        self._current_epoch = value
    
    @property
    def learning_rate(self):
        return self._learning_rate
    
    @learning_rate.setter
    def learning_rate(self, value):
        self._learning_rate = value

    @property
    def train_loader_length(self):
        return self._train_loader_length
    
    @train_loader_length.setter
    def train_loader_length(self, value):
        self._train_loader_length = value

    @property
    def current_start_time(self):
        return self._current_start_time
    
    @current_start_time.setter
    def current_start_time(self, value):
        self._current_start_time = value

    def reset_metrics(self):
        self.acc_meter.reset()
        self.loss_meter.reset()
        self.evaluator.reset()

    def update(self, loss, outputs, target):
        acc = self.calculate_accuracy(outputs, target)
        self.loss_meter.update(loss.item(), self.config.SOLVER.IMS_PER_BATCH)
        self.acc_meter.update(acc.item(), 1)
    
    def calculate_accuracy(self, outputs, target):
        index = self.config.LOSS.ID_LOSS_OUTPUT_INDEX if isinstance(outputs, tuple) else 0
        id_classifier_output = outputs[index]
        id_hat_element = id_classifier_output[0] if isinstance(id_classifier_output, list) else id_classifier_output
        acc = (id_hat_element.max(1)[1] == target).float().mean()

        return acc 