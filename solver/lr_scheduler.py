# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
from bisect import bisect_right
import torch


# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,        
        optimizer,
        cfg,
        last_epoch=-1,
    ):
        self.milestones = cfg.SOLVER.STEPS
        self.gamma = cfg.SOLVER.GAMMA
        self.warmup_factor = cfg.SOLVER.WARMUP_FACTOR
        self.warmup_iters = cfg.SOLVER.WARMUP_ITERS
        self.warmup_method = cfg.SOLVER.WARMUP_METHOD
        self.check_scheduler_params()
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def current_epoch(self, value):
        self.last_epoch = value

    def check_scheduler_params(self):
        if not list(self.milestones) == sorted(self.milestones):
            raise ValueError(
                "Milestones should be a list of increasing integers. Got {}",
                self.milestones,
            )

        if self.warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(self.warmup_method)
            )

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]
