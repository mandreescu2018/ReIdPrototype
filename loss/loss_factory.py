import torch
import torch.nn as nn
from .triplet_loss import TripletLoss
from .softmax_loss import CrossEntropyLabelSmooth

class MultipleLoss:
    def __init__(self, cfg) -> None:
        self.config = cfg
        self.loss_fns = []
        self.sampler = cfg.DATALOADER.SAMPLER
        self._triplet_loss = None
        self._cross_entropy_loss = None
        
    @property
    def triplet_loss(self):
        if self._triplet_loss is None:
            self._triplet_loss = TripletLoss(self.config.SOLVER.MARGIN)
        return self._triplet_loss
    
    @property
    def cross_entropy_loss(self):
        if self._cross_entropy_loss == None:
            if  self.config.MODEL.IF_LABELSMOOTH == 'on':
                self._cross_entropy_loss = CrossEntropyLabelSmooth(self.cfg.DATASETS.NUMBER_OF_CLASSES)
            else:
                self._cross_entropy_loss = nn.CrossEntropyLoss()
        return self._cross_entropy_loss


    def loos_fn(self):
        return self.loss_fns
    
    def compute_tripple_loss(self, feat, target):
        if isinstance(feat, list):
            loss = [self.triplet_loss(feats, target)[0] for feats in feat[1:]]
            loss = sum(loss) / len(loss)
            loss = 0.5 * loss + 0.5 * self.triplet_loss(feat[0], target)[0]
        else:
            loss = self.triplet_loss(feat, target)[0]
        return loss

    def compute_cross_entropy_loss(self, score, target):
        if isinstance(score, list):
            loss = [self.cross_entropy_loss(scor, target) for scor in score[1:]]
            loss = sum(loss) / len(loss)
            loss = 0.5 * loss + 0.5 * self.cross_entropy_loss(score[0], target)
        else:
            loss = self.cross_entropy_loss(score, target)
        return loss

    def __call__(self, score, feat, target, target_cam):

        cross_entropy_loss = self.compute_cross_entropy_loss(score, target)                
        tri_loss = self.compute_tripple_loss(feat, target)

        ce_weight = self.config.MODEL.ID_LOSS_WEIGHT
        tri_weight = self.config.MODEL.TRIPLET_LOSS_WEIGHT

        return ce_weight * cross_entropy_loss + tri_weight * tri_loss
