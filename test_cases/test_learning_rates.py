import sys
sys.path.append('.')

import unittest

import torch
import torch.nn as nn
from config import cfg
from solver.scheduler_factory import LearningRateScheduler        


class TestScheduer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print('setupClass')
        
    
    def test_step_lr_scheduler(self):
        """
        Test that StepLR updates the learning rate correctly.
        """
        # Example model and optimizer
        model = nn.Linear(10, 1)

        number_of_epochs = 50
        cfg.SOLVER.SCHEDULER = 'step'
        cfg.SOLVER.BASE_LR = 0.1
        cfg.SOLVER.GAMMA = 0.5
        cfg.SOLVER.STEPS = [10]
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM)
        scheduler = LearningRateScheduler(optimizer, cfg)

        # Before any step
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 0.1, places=6)

        for epoch in range(1, number_of_epochs + 1):
            expected_lr = 0.1* cfg.SOLVER.GAMMA**(epoch//cfg.SOLVER.STEPS[0])
            scheduler.step(epoch)
            self.assertAlmostEqual(optimizer.param_groups[0]['lr'], expected_lr, places=6)
        
if __name__ == '__main__':
    unittest.main()

        