import torch.nn as nn
from .triplet_loss import TripletLoss
from .softmax_loss import CrossEntropyLabelSmooth

class GeneralLoss:
    def __init__(self, cfg) -> None:
        self.config = cfg
        self._loss = None
    
    @property
    def loss(self):
        raise NotImplementedError

    def __call__(self, outputs, target):
        return self.compute_loss(outputs, target) * self.weight

class TripletLossWrap(GeneralLoss):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.weight = cfg.LOSS.METRIC_LOSS_WEIGHT
        self.output_index = cfg.LOSS.METRIC_LOSS_OUTPUT_INDEX
    
    @property
    def loss(self):
        if self._loss == None:
            self._loss = TripletLoss(self.config.LOSS.TRIPLET_MARGIN)
        return self._loss
    
    def compute_loss(self, outputs, target):
        feat = outputs[self.output_index]
        if isinstance(feat, list):
            loss = [self.loss(feats, target)[0] for feats in feat[1:]]
            loss = sum(loss) / len(loss)
            loss = 0.5 * loss + 0.5 * self.loss(feat[0], target)[0]
        else:
            loss = self.loss(feat, target)[0]
        return loss * self.weight

class CrossEntropyLossWrap(GeneralLoss):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.weight = cfg.LOSS.ID_LOSS_WEIGHT
        self.output_index = cfg.LOSS.ID_LOSS_OUTPUT_INDEX
    
    @property
    def loss(self):
        if self._loss == None:
            if  self.config.LOSS.IF_LABELSMOOTH == 'on':
                self._loss = CrossEntropyLabelSmooth(self.cfg.DATASETS.NUMBER_OF_CLASSES)
            else:
                self._loss = nn.CrossEntropyLoss()
        return self._loss

    def compute_loss(self, outputs, target):
        if isinstance(outputs, tuple):
            score = outputs[self.output_index]
        else:
            score = outputs
            
        if isinstance(score, list):
            loss = [self.loss(scor, target) for scor in score[1:]]
            loss = sum(loss) / len(loss)
            loss = 0.5 * loss + 0.5 * self.loss(score[0], target)
        else:
            loss = self.loss(score, target)
        return loss * self.weight

class LossComposer:
    _factory = {
        'triplet': TripletLossWrap,
        'cross_entropy': CrossEntropyLossWrap
    }

    def __init__(self, cfg) -> None:
        self.config = cfg
        self.loss_fns = []
        self.load_losses()

    def add_loss_fn(self, loss_fn):
        self.loss_fns.append(loss_fn)

    def load_losses(self):
        # identity loss
        if self.config.LOSS.ID_LOSS_TYPE in LossComposer._factory:
            self.add_loss_fn(LossComposer._factory[self.config.LOSS.ID_LOSS_TYPE](self.config))
        # metric loss
        if self.config.LOSS.METRIC_LOSS_TYPE in LossComposer._factory:
            self.add_loss_fn(LossComposer._factory[self.config.LOSS.METRIC_LOSS_TYPE](self.config))

    def __call__(self, outputs, target):
        final_loss = 0
        for loss_fn in self.loss_fns:
            final_loss += loss_fn(outputs, target)
        
        return final_loss
