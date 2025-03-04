from utils import AverageMeter
from utils.metrics import R1_mAP_eval

class MetricsLiveValues:
    def __init__(self, cfg):
        self.config = cfg
        self.acc_meter = AverageMeter()
        self.loss_meter = AverageMeter()
        self.evaluator = R1_mAP_eval(cfg.DATASETS.NUMBER_OF_IMAGES_IN_QUERY, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        self.current_epoch = 0
        self.learning_rate = 0
        self.train_loader_length = 0
        self.current_start_time = None
        self.mAP = 0
        self.cmc = None

    def reset_metrics(self):
        self.acc_meter.reset()
        self.loss_meter.reset()
        self.evaluator.reset()

    def update(self, loss, outputs, target, accuracy=None):
        if accuracy is None:
            acc = self.calculate_accuracy(outputs, target)
        else:
            acc = accuracy
        self.loss_meter.update(loss.item(), self.config.SOLVER.IMS_PER_BATCH)
        self.acc_meter.update(acc.item(), 1)
    
    def calculate_accuracy(self, outputs, target):
        index = self.config.LOSS.ID_LOSS_OUTPUT_INDEX if isinstance(outputs, tuple) else None
        if index is None:
            id_classifier_output = outputs
        else:
            id_classifier_output = outputs[index]
        # id_classifier_output = outputs[index]
        id_hat_element = id_classifier_output[0] if isinstance(id_classifier_output, list) else id_classifier_output
        acc = (id_hat_element.max(1)[1] == target).float().mean()

        return acc 