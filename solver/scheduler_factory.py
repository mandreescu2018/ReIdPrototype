import torch
import math

class GeneralLRScheduler:
    def __init__(self, optimizer, strategy="step", config=None):
        """
        General learning rate scheduler supporting multiple strategies.

        Args:
            optimizer (torch.optim.Optimizer): Optimizer to adjust the learning rate for.
            strategy (str): Scheduling strategy. Options are:
                - "step": Step decay.
                - "exponential": Exponential decay.
                - "cosine": Cosine annealing.
                - "plateau": Reduce on plateau (validation-based).
            config (dict): Configuration for the scheduler strategy. Keys depend on the chosen strategy:
                - "step": {"step_size": int, "factor": float, "min_lr": float}
                - "exponential": {"gamma": float, "min_lr": float}
                - "cosine": {"T_max": int, "min_lr": float}
                - "plateau": {"patience": int, "factor": float, "min_lr": float}
        """
        self.optimizer = optimizer
        self.strategy = strategy.lower()
        self.config = config or {}
        self.last_epoch = 0
        self.best_metric = float("inf")
        self.epochs_since_improvement = 0

        # Check for required keys in the config
        if self.strategy == "step" and "step_size" not in self.config:
            raise ValueError("For 'step' strategy, 'step_size' must be provided.")
        if self.strategy not in ["step", "exponential", "cosine", "plateau"]:
            raise ValueError(f"Unsupported strategy: {self.strategy}")

    def step(self, epoch=None, metric=None):
        """
        Updates the learning rate based on the chosen strategy.

        Args:
            epoch (int, optional): Current epoch. If None, increments the internal epoch counter.
            metric (float, optional): Metric to monitor for plateau-based scheduling.
        """
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.strategy == "step":
            self._step_decay()
        elif self.strategy == "exponential":
            self._exponential_decay()
        elif self.strategy == "cosine":
            self._cosine_annealing()
        elif self.strategy == "plateau":
            if metric is None:
                raise ValueError("For 'plateau' strategy, 'metric' must be provided.")
            self._plateau(metric)

    def _step_decay(self):
        """Step decay: Reduce learning rate every `step_size` epochs."""
        step_size = self.config.get("step_size", 10)
        factor = self.config.get("factor", 0.1)
        min_lr = self.config.get("min_lr", 1e-6)

        if self.last_epoch % step_size == 0:
            for param_group in self.optimizer.param_groups:
                new_lr = max(param_group["lr"] * factor, min_lr)
                param_group["lr"] = new_lr

    def _exponential_decay(self):
        """Exponential decay: Multiply learning rate by `gamma` each epoch."""
        gamma = self.config.get("gamma", 0.9)
        min_lr = self.config.get("min_lr", 1e-6)

        for param_group in self.optimizer.param_groups:
            new_lr = max(param_group["lr"] * gamma, min_lr)
            param_group["lr"] = new_lr

    def _cosine_annealing(self):
        """Cosine annealing: Learning rate follows a cosine curve."""
        T_max = self.config.get("T_max", 50)
        min_lr = self.config.get("min_lr", 1e-6)

        for param_group in self.optimizer.param_groups:
            new_lr = min_lr + (param_group["initial_lr"] - min_lr) * (1 + math.cos(math.pi * self.last_epoch / T_max)) / 2
            param_group["lr"] = new_lr

    def _plateau(self, metric):
        """Reduce on plateau: Reduce learning rate when the monitored metric stops improving."""
        patience = self.config.get("patience", 5)
        factor = self.config.get("factor", 0.1)
        min_lr = self.config.get("min_lr", 1e-6)

        if metric < self.best_metric:
            self.best_metric = metric
            self.epochs_since_improvement = 0
        else:
            self.epochs_since_improvement += 1

        if self.epochs_since_improvement >= patience:
            for param_group in self.optimizer.param_groups:
                new_lr = max(param_group["lr"] * factor, min_lr)
                param_group["lr"] = new_lr
            self.epochs_since_improvement = 0

# ---

# ### Configuration and Usage Example

# #### Example Configurations

# 1. **Step Decay**
#    ```python
#    step_config = {"step_size": 5, "factor": 0.5, "min_lr": 1e-6}
