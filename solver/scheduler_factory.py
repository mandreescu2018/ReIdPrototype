from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR
from .warmup_lr_scheduler import WarmupMultiStepLR
from .cosine_lr import CosineLRScheduler


class SchedulerStrategy:
    """Base class for scheduler strategies."""
    def __init__(self, optimizer, cfg):
        self.optimizer = optimizer
        self.config = cfg

    @property
    def scheduler(self):
        raise NotImplementedError

class WarmupMultiStepLRScheduler(SchedulerStrategy):
    """Wrapper class for WarmupMultiStepLR scheduler."""
    @property
    def scheduler(self):
        return WarmupMultiStepLR(self.optimizer, self.config)

class ExponentialLRScheduler(SchedulerStrategy):
    """Wrapper class for PyTorch ExponentialLR scheduler."""
    @property
    def scheduler(self):
        return ExponentialLR(self.optimizer,
                             gamma=self.config.SOLVER.GAMMA)

class StepLRScheduler(SchedulerStrategy):
    """Wrapper class for PyTorch StepLR scheduler."""
    @property
    def step_size(self):
        steps = self.config.SOLVER.STEPS
        return steps[0] if isinstance(steps, (tuple, list)) else steps

    @property
    def scheduler(self):
        return StepLR(
            self.optimizer, 
            step_size=self.step_size, 
            gamma=self.config.SOLVER.GAMMA
        )

class CosineAnealingScheduler(SchedulerStrategy):
    @property
    def scheduler(self):
        return CosineAnnealingLR(self.optimizer, 
                                 T_max=self.config.SOLVER.MAX_EPOCHS, 
                                 eta_min=0.0, last_epoch=-1)
    
class CosineScheduler(SchedulerStrategy):
    """Wrapper class for CosineLRScheduler."""
    @property
    def scheduler(self):
        # type 1
        # lr_min = 0.01 * cfg.SOLVER.BASE_LR
        # warmup_lr_init = 0.001 * cfg.SOLVER.BASE_LR
        # type 2
        lr_min = 0.002 * self.config.SOLVER.BASE_LR
        warmup_lr_init = 0.01 * self.config.SOLVER.BASE_LR
        # type 3
        # lr_min = 0.001 * cfg.SOLVER.BASE_LR
        # warmup_lr_init = 0.01 * cfg.SOLVER.BASE_LR

        warmup_t = self.config.SOLVER.WARMUP_ITERS
        noise_range = None

        return CosineLRScheduler(
                self.optimizer,
                t_initial=self.config.SOLVER.MAX_EPOCHS,
                lr_min=lr_min,
                t_mul= 1.,
                decay_rate=0.1,
                warmup_lr_init=warmup_lr_init,
                warmup_t=warmup_t,
                cycle_limit=1,
                t_in_epochs=True,
                noise_range_t=noise_range,
                noise_pct= 0.67,
                noise_std= 1.,
                noise_seed=42,
            )

_scheduler_factory = {
        "step": StepLRScheduler,
        "warm_up": WarmupMultiStepLRScheduler,
        "exponential": ExponentialLRScheduler,
        "cosine": CosineScheduler,
        "cosine_annealing": CosineAnealingScheduler
    }

class LearningRateScheduler:
    """Factory class for the learning rate scheduler."""
    def __init__(self, optimizer, cfg):
        """
        Args:
            optimizer (torch.optim.Optimizer): Optimizer instance.
            cfg (dict): configuration values.                
        """
        self.optimizer = optimizer
        self.config = cfg
        self.scheduler = self._build_scheduler()
        

    def _build_scheduler(self):
        """
        Factory method to build the scheduler.
        """
        scheduler_type = self.config.SOLVER.SCHEDULER
        return _scheduler_factory[scheduler_type](self.optimizer, self.config).scheduler

        
    def load_state_dict(self, state_dict):
        """
        Loads the state of the scheduler.
        Args:
            state_dict (dict): State dictionary.
        """
        self.scheduler.load_state_dict(state_dict)

    def state_dict(self):
        """
        Returns the current state of the scheduler.
        """
        return self.scheduler.state_dict()
    
    def step(self, epoch):
        """
        Steps the current scheduler to update learning rates.
        Args:
            epoch (int): Current training epoch.
        """
        self.scheduler.step(epoch)
