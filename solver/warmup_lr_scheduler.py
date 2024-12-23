import torch
from bisect import bisect_right
from typing import List

class WarmupMultiStepLR(torch.optim.lr_scheduler.LRScheduler):
    def __init__(
        self,        
        optimizer: torch.optim.Optimizer,
        cfg,
        last_epoch: int = -1,
    ):
        self.milestones: List[int] = cfg.SOLVER.STEPS
        self.gamma: float = cfg.SOLVER.GAMMA
        self.warmup_factor: float = cfg.SOLVER.WARMUP_FACTOR
        self.warmup_iters: int = cfg.SOLVER.WARMUP_ITERS
        self.warmup_method: str = cfg.SOLVER.WARMUP_METHOD
        self._check_scheduler_params()
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def current_epoch(self, value: int) -> None:
        self.last_epoch = value

    def _check_scheduler_params(self) -> None:
        if not list(self.milestones) == sorted(self.milestones):
            raise ValueError(
                f"Milestones should be a list of increasing integers. Got {self.milestones}"
            )

        if self.warmup_method not in ("constant", "linear"):
            raise ValueError(
                f"Only 'constant' or 'linear' warmup_method accepted, got {self.warmup_method}"
            )

    def get_lr(self) -> List[float]:
        warmup_factor = self._get_warmup_factor()
        return [
            base_lr * warmup_factor * self._get_decay_factor()
            for base_lr in self.base_lrs
        ]

    def _get_warmup_factor(self) -> float:
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                return self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                return self.warmup_factor * (1 - alpha) + alpha
        return 1.0

    def _get_decay_factor(self) -> float:
        return self.gamma ** bisect_right(self.milestones, self.last_epoch)