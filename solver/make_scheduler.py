from .create_scheduler import create_scheduler, create_lr_scheduler

__factory = {
    'cosine': create_scheduler,
    'step_lr': create_lr_scheduler,
}
# exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
def make_scheduler(cfg, optimizer):
    return __factory[cfg.SOLVER.SCHEDULER](cfg, optimizer)
