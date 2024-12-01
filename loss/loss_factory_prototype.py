import torch.nn as nn
from .triplet_loss import TripletLoss
from .softmax_loss import CrossEntropyLabelSmooth
from .center_loss import CenterLoss

class BaseLoss:
    def __init__(self, cfg) -> None:
        self.config = cfg
        self._loss = None
        self.weight = cfg["weight"]
        self.output_index = cfg["output_index"]
        self.label_smooth = cfg.get("label_smooth", None)
        self.margin = cfg.get("margin", None)
    
    @property
    def loss(self):
        raise NotImplementedError

    def __call__(self, outputs, target):
        return self.compute(outputs, target) * self.weight

class CenterLossWrap(BaseLoss):
    def __init__(self, cfg, num_classes, feature_dim) -> None:
        super().__init__(cfg)
        self.num_classes = num_classes
        self.feature_dim = feature_dim
    
    @property
    def loss(self):
        if self._loss == None:
            self._loss = CenterLoss(self.num_classes, self.feature_dim)
        return self._loss
    
    def compute(self, outputs, target):
        feat = outputs[self.output_index]
        return self.loss(feat, target) * self.weight

class TripletLossWrap(BaseLoss):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)        
    
    @property
    def loss(self):
        if self._loss == None:
            self._loss = TripletLoss(self.margin)
        return self._loss
    
    def compute(self, outputs, target):
        feat = outputs[self.output_index]
        if isinstance(feat, list):
            loss = [self.loss(feats, target)[0] for feats in feat[1:]]
            loss = sum(loss) / len(loss)
            loss = 0.5 * loss + 0.5 * self.loss(feat[0], target)[0]
        else:
            loss = self.loss(feat, target)[0]
        return loss * self.weight

class CrossEntropyLossWrap(BaseLoss):
    def __init__(self, cfg, num_classes) -> None:
        super().__init__(cfg)
        self.num_classes = num_classes
    
    @property
    def loss(self):
        if self._loss == None:
            if  self.label_smooth == 'on':
                self._loss = CrossEntropyLabelSmooth(self.num_classes)
            else:
                self._loss = nn.CrossEntropyLoss()
        return self._loss

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

class LossFactory:
    @staticmethod
    def create_loss(loss_conf, num_classes, feature_dim):
        """
        Create a loss function instance based on its type.

        Args:
            loss_conf (dict): losses dict.
            num_classes (int): number of person ids.

        Returns:
            An instance of a loss function (BaseLoss or derived class).
        """
        if loss_conf["type"] == "cross_entropy":
            return CrossEntropyLossWrap(loss_conf, num_classes)
        elif loss_conf["type"] == "triplet":
            return TripletLossWrap(loss_conf)
        elif loss_conf["type"] == "center":
            return CenterLossWrap(loss_conf, num_classes, feature_dim)
        else:
            raise ValueError(f"Unsupported loss type: {loss_conf['type']}")
    

class LossComposer:
    def __init__(self, cfg):
        """
        Initialize the LossComposer with a list of loss configurations.
        
        Args:
            cfg : Configuration yacs object.
        """
        self.config = cfg
        self._center_criterion = None
        self.loss_functions = []

        loss_configs  = cfg.LOSS.COMPONENTS
        num_classes = cfg.DATASETS.NUMBER_OF_CLASSES
        feature_dim = cfg.SOLVER.FEATURE_DIMENSION
        

        for loss_conf in loss_configs:
            loss_fn = LossFactory.create_loss(loss_conf, num_classes, feature_dim)
            if loss_conf["type"] == "center":
                self.set_center_criterion(loss_fn)
            self.loss_functions.append(loss_fn)            

    @property
    def center_criterion(self):
        return self._center_criterion
    
    def set_center_criterion(self, loss_fn):
        self._center_criterion = loss_fn.loss
    
    def __call__(self, outputs, targets):
        """
        Compute the total weighted loss.

        Args:
            outputs: Model outputs (single or multi-output).
            targets: Ground truth targets.

        Returns:
            Total combined loss value (torch.Tensor).
        """
        total_loss = 0.0
        for loss_fn in self.loss_functions:
            total_loss += loss_fn.compute(outputs, targets)
        return total_loss





