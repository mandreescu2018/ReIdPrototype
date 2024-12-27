import torch

class OptimizerFactory:
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model = model

    def make_optimizer(self):
        params = self._get_params()
        optimizer = self._get_optimizer(params)
        return optimizer

    def _get_bias_params(self):
        lr = self.cfg.SOLVER.BASE_LR * self.cfg.SOLVER.BIAS_LR_FACTOR
        weight_decay = self.cfg.SOLVER.WEIGHT_DECAY_BIAS
        return lr, weight_decay
    
    def _get_params(self):
        params = []
        for key, value in self.model.named_parameters():
            if not value.requires_grad:
                continue
            lr = self.cfg.SOLVER.BASE_LR
            weight_decay = self.cfg.SOLVER.WEIGHT_DECAY
            if "bias" in key:
                lr, weight_decay = self._get_bias_params()
            if self.cfg.SOLVER.LARGE_FC_LR and ("classifier" in key or "arcface" in key):
                lr = self.cfg.SOLVER.BASE_LR * 2
                print('Using two times learning rate for fc')
            params.append({"params": [value], "lr": lr, "weight_decay": weight_decay})
        return params

    def _get_optimizer(self, params):
        if self.cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
            return torch.optim.SGD(params, momentum=self.cfg.SOLVER.MOMENTUM)
        elif self.cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
            return torch.optim.AdamW(params, lr=self.cfg.SOLVER.BASE_LR, weight_decay=self.cfg.SOLVER.WEIGHT_DECAY)
        elif self.cfg.SOLVER.OPTIMIZER_NAME == 'Adam':
            return torch.optim.Adam(params, lr=self.cfg.SOLVER.BASE_LR)
        else:
            return getattr(torch.optim, self.cfg.SOLVER.OPTIMIZER_NAME)(params)

