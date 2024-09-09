import os
import sys
root = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(root)


import torch
from torch import nn
from utils import set_seeds


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, cfg):
        super(CenterLoss, self).__init__()
        self.device = cfg.DEVICE
        self.num_classes = cfg.DATASETS.NUMBER_OF_CLASSES

        self.centers = nn.Parameter(torch.randn(self.num_classes, cfg.SOLVER.FEATURE_DIMENSION).to(self.device))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        assert x.size(0) == labels.size(0), "features.size(0) is not equal to labels.size(0)"

        batch_size = x.size(0)
        # Compute the Euclidean distances between each sample in the batch (x) 
        # and each class center (self.centers) and store the distances in the distmat tensor.
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        # The choice of -2 in alpha is related to the way Euclidean distances are computed, 
        # which is the square root of the sum of squared differences. The -2 scaling is applied to match this calculation, 
        # as it effectively squares the differences. This is a common technique used to compute distances efficiently in a 
        # batch-wise manner within deep learning models.        
        # Performs a matrix multiplication of the matrices x and self.centers.t(). The matrix input is added to the final result.
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        

        # dist = []
        # for i in range(batch_size):
        #     value = distmat[i][mask[i]]
        #     value = value.clamp(min=1e-12, max=1e+12)  # for numerical stability
        #     dist.append(value)
        # dist = torch.cat(dist)
        # loss = dist.mean()
        # return loss

        dist = distmat * mask.float()
        dist = dist.clamp(min=1e-12, max=1e+12)
        loss = dist.sum() / batch_size
        return loss


if __name__ == '__main__':
    set_seeds()

    center_loss = CenterLoss(device="cuda")
    features = torch.rand(16, 2048).to("cuda")
    targets = torch.Tensor([0, 1, 2, 3, 2, 3, 1, 4, 5, 3, 2, 1, 0, 0, 5, 4]).long()
    targets = targets.to("cuda")

    loss = center_loss(features, targets)
    print(loss)