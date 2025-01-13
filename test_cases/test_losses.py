import sys
sys.path.append('.')

import unittest
import torch
import random
import numpy as np
from utils import set_seeds
from config import cfg

class TestLosses(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print('setupClass')
        set_seeds()
        cfg.SOLVER.FEATURE_DIMENSION = 2048
        cfg.DATASETS.NUMBER_OF_CLASSES = 751

    def test_mse(self):
        self.assertEqual(1, 1)

    def test_bce(self):
        self.assertEqual(1, 1)

    def test_triplet(self):
        from loss import TripletLoss
        features = torch.randn(128, cfg.SOLVER.FEATURE_DIMENSION)
        class_names_tensor = torch.tensor([394, 394, 394, 394, 430, 430, 430, 430,  41,  41,  41,  41, 265, 265,
        265, 265, 523, 523, 523, 523, 497, 497, 497, 497, 414, 414, 414, 414,
        310, 310, 310, 310, 488, 488, 488, 488, 366, 366, 366, 366, 597, 597,
        597, 597, 223, 223, 223, 223, 516, 516, 516, 516, 142, 142, 142, 142,
        288, 288, 288, 288, 143, 143, 143, 143,  97,  97,  97,  97, 633, 633,
        633, 633, 256, 256, 256, 256, 545, 545, 545, 545, 722, 722, 722, 722,
        616, 616, 616, 616, 150, 150, 150, 150, 317, 317, 317, 317, 101, 101,
        101, 101, 747, 747, 747, 747,  75,  75,  75,  75, 700, 700, 700, 700,
        338, 338, 338, 338, 483, 483, 483, 483, 573, 573, 573, 573, 103, 103,
        103, 103])
        triplet = TripletLoss(margin=1)
        loss, _, _ = triplet(features, class_names_tensor)
        # print("loss_t", loss_t.item())
        self.assertEqual(loss.item(), 3.9720332622528076)
    
    def test_center_loss(self):
        from loss import CenterLoss
        cfg.DATASETS.NUMBER_OF_CLASSES = 751
        center_loss = CenterLoss(cfg.DATASETS.NUMBER_OF_CLASSES, cfg.SOLVER.FEATURE_DIMENSION)
        features = torch.rand(16, cfg.SOLVER.FEATURE_DIMENSION).to("cuda")
        targets = torch.Tensor([0, 1, 2, 3, 2, 3, 1, 4, 5, 3, 2, 1, 0, 0, 5, 4]).long()
        targets = targets.to("cuda")

        loss = center_loss(features, targets)
        self.assertEqual(loss.item(), 2721.125)


if __name__ == '__main__':
    unittest.main()