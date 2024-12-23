import torch
import torch.nn as nn
import torch.optim as optim

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt

from .warmup_lr_scheduler import WarmupMultiStepLR

class LRSchedulerStrategy(ABC):
    @abstractmethod
    def step(self, optimizer, epoch, **kwargs):
        """
        Adjusts the learning rate for the optimizer.
        
        Args:
            optimizer (torch.optim.Optimizer): Optimizer instance.
            epoch (int): Current epoch.
            kwargs: Additional parameters for specific strategies.
        """
        pass

# class StepLRScheduler(LRSchedulerStrategy):
#     def __init__(self, step_size, gamma=0.1):
#         self.step_size = step_size
#         self.gamma = gamma

#     def step(self, optimizer, epoch, **kwargs):
#         if epoch % self.step_size == 0 and epoch > 0:
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] *= self.gamma
#                 print(f"StepLR: Learning rate updated to {param_group['lr']:.6f}")

class ExponentialLRScheduler(LRSchedulerStrategy):
    def __init__(self, gamma=0.95):
        self.gamma = gamma

    def step(self, optimizer, epoch, **kwargs):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= self.gamma
            print(f"ExponentialLR: Learning rate updated to {param_group['lr']:.6f}")

import math

class CosineAnnealingLRScheduler(LRSchedulerStrategy):
    def __init__(self, T_max, eta_min=0):
        self.T_max = T_max
        self.eta_min = eta_min

    def step(self, optimizer, epoch, **kwargs):
        for param_group in optimizer.param_groups:
            lr = self.eta_min + (param_group['initial_lr'] - self.eta_min) * (
                1 + math.cos(math.pi * epoch / self.T_max)) / 2
            param_group['lr'] = lr
            print(f"CosineAnnealingLR: Learning rate updated to {lr:.6f}")

class StepLRScheduler:
    def __init__(self, optimizer, cfg):
        self.optimizer = optimizer
        self.config = cfg

    @property
    def step_size(self):
        steps = self.config.SOLVER.STEPS
        return steps if isinstance(steps, int) else steps[0]

    @property
    def scheduler(self):
        return torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=self.step_size, 
            gamma=self.config.SOLVER.GAMMA
        )


class LrScheduler:
    def __init__(self, optimizer, cfg):
        """
        Args:
            optimizer (torch.optim.Optimizer): Optimizer instance.
            scheduler_config (dict): Scheduler configuration, e.g.,
                {"type": "step", "params": {"step_size": 10, "gamma": 0.1}}
        """
        self.optimizer = optimizer
        self.config = cfg
        self.scheduler = self._build_scheduler(cfg)
        self.initial_lrs = [group['lr'] for group in optimizer.param_groups]
        

    def _build_scheduler(self, scheduler_config):
        scheduler_type = self.config.SOLVER.SCHEDULER
        params = scheduler_config.get("params", {})

        if scheduler_type == "step":
            return StepLRScheduler(self.optimizer, self.config).scheduler
        elif scheduler_type == "warm_up":
            return WarmupMultiStepLR(self.optimizer, self.config)
        elif scheduler_type == "exponential":
            return ExponentialLRScheduler(**params)
        elif scheduler_type == "cosine":
            return CosineAnnealingLRScheduler(**params)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    def step(self, epoch):
        """
        Steps the current scheduler to update learning rates.
        Args:
            epoch (int): Current training epoch.
        """
        self.scheduler.step(epoch)

class DynamicLRScheduler:
    def __init__(self, optimizer, scheduler_config):
        """
        Args:
            optimizer (torch.optim.Optimizer): Optimizer instance.
            scheduler_config (dict): Scheduler configuration, e.g.,
                {"type": "step", "params": {"step_size": 10, "gamma": 0.1}}
        """
        self.optimizer = optimizer
        self.scheduler = self._build_scheduler(scheduler_config)
        self.initial_lrs = [group['lr'] for group in optimizer.param_groups]

    def _build_scheduler(self, scheduler_config):
        scheduler_type = scheduler_config.get("type", "").lower()
        params = scheduler_config.get("params", {})

        if scheduler_type == "step":
            # return torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=cfg.SOLVER.GAMMA)
            return StepLRScheduler(**params)
        elif scheduler_type == "exponential":
            return ExponentialLRScheduler(**params)
        elif scheduler_type == "cosine":
            return CosineAnnealingLRScheduler(**params)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    def step(self, epoch):
        """
        Steps the current scheduler to update learning rates.
        Args:
            epoch (int): Current training epoch.
        """
        self.scheduler.step(self.optimizer, epoch)

if __name__ == "__main__":

    learning_rates =[]
    # Example model and optimizer
    model = nn.Linear(10, 1)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Scheduler configuration (can be dynamic)
    scheduler_config = {
        "type": "step",  # Choose "step", "exponential", or "cosine"
        "params": {"step_size": 5, "gamma": 0.5}    
    }

    # scheduler_config = {
    #     "type": "exponential",  # Choose "step", "exponential", or "cosine"
    #     "params": {"gamma": 0.5}    
    # }

    # Initialize the dynamic scheduler
    dynamic_lr_scheduler = DynamicLRScheduler(optimizer, scheduler_config)

    # Simulated training loop
    for epoch in range(1, 160):
        # Simulate training step
        optimizer.zero_grad()
        loss = model(torch.randn(32, 10)).sum()
        loss.backward()
        optimizer.step()

        # Update the learning rate
        dynamic_lr_scheduler.step(epoch)

        # Print current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        print(f"Epoch {epoch}, LR: {current_lr:.6f}")

    plt.plot(learning_rates)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.show()
