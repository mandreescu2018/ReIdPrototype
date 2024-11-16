import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.resnet import ResNet50_Weights
from utils.weight_utils import weights_init_classifier

class SimpleReIDModel(nn.Module):
    def __init__(self, cfg):
        num_classes = cfg.DATASETS.NUMBER_OF_CLASSES
        feature_dim = cfg.SOLVER.FEATURE_DIMENSION
        super(SimpleReIDModel, self).__init__()
        self.backbone = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()  # Remove final classification layer
        self.fc = nn.Linear(feature_dim, num_classes)  # Classifier layer
        self.fc.apply(weights_init_classifier)

    def forward(self, x):
        features = self.backbone(x)  # Extract features
        normalized_features = F.normalize(features)  # Normalize features
        class_scores = self.fc(normalized_features)  # Classify
        return class_scores, normalized_features