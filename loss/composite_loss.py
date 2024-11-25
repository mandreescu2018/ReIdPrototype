import torch
import torch.nn as nn
from .triplet_loss import TripletLoss
from .softmax_loss import CrossEntropyLabelSmooth

class BaseLoss:
    def __init__(self, weight=1.0):
        self.weight = weight

    def compute(self, outputs, targets):
        """
        Compute the loss given the model outputs and targets.
        Args:
            outputs: Model outputs (can be a single output or multiple outputs).
            targets: Ground truth labels or targets.
        Returns:
            Weighted loss value (torch.Tensor).
        """
        raise NotImplementedError("compute method must be implemented in derived classes.")

# class CrossEntropyLoss(BaseLoss):
#     def __init__(self, weight=1.0):
#         super().__init__(weight)
#         self.criterion = nn.CrossEntropyLoss()

#     def compute(self, outputs, targets):
#         return self.weight * self.criterion(outputs, targets)
    
class CrossEntropyLoss(BaseLoss):
    def __init__(self, cfg) -> None:
        super().__init__(cfg.LOSS.ID_LOSS_WEIGHT)
        # self.weight = cfg.LOSS.ID_LOSS_WEIGHT
        self.output_index = cfg.LOSS.ID_LOSS_OUTPUT_INDEX
    
    @property
    def loss(self):
        if self._loss == None:
            if  self.config.LOSS.IF_LABELSMOOTH == 'on':
                self._loss = CrossEntropyLabelSmooth(self.cfg.DATASETS.NUMBER_OF_CLASSES)
            else:
                self._loss = nn.CrossEntropyLoss()
        return self._loss

    class TripletLossWrap(BaseLoss):
        def __init__(self, cfg) -> None:
            super().__init__(cfg.LOSS.METRIC_LOSS_WEIGHT)
            # self.weight = cfg.LOSS.METRIC_LOSS_WEIGHT
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

        def compute(self, outputs, target):
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
    def __init__(self):
        self.losses = []

    def add_loss(self, loss_component):
        """
        Add a loss function component to the composer.
        Args:
            loss_component: An instance of BaseLoss or derived class.
        """
        self.losses.append(loss_component)

    def compute_total_loss(self, outputs, targets):
        """
        Compute the total loss by aggregating all individual losses.
        Args:
            outputs: Model outputs (can be a single output or multiple outputs).
            targets: Ground truth labels or targets.
        Returns:
            Total combined loss value (torch.Tensor).
        """
        total_loss = 0.0
        for loss_fn in self.losses:
            total_loss += loss_fn.compute(outputs, targets)
        return total_loss

if __name__ == "__main__":
    loss_composer = LossComposer()
    loss_composer.add_loss(CrossEntropyLoss())
    # loss_composer.add_loss(TripletLossWrap())
    print(loss_composer.compute_total_loss(outputs, targets))