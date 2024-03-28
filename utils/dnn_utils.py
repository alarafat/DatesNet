import torch
import torch.optim as optim


class EvaluationMetric(object):
    def __init__(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.average = self.sum / self.count


def get_optimizer(model, cfg):
    assert cfg.TrainConfig.optimizer_name in ['sgd', 'adam', 'rmsprop', 'adamw'], "Optimizer option can be only between [SGD/Adam/RMSProp/AdamW]"

    optim_name = cfg.TrainConfig.optimizer_name.lower()
    if 'adam' in optim_name:
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TrainConfig.lr
        )
    elif 'adamw' in optim_name:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TrainConfig.lr,
            betas=(cfg.TrainConfig.beta1, cfg.TrainConfig.beta2),
            fused=True if cfg.device == 'cuda' else False)
    elif 'rmsprop' in optim_name:
        optimizer = optim.RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TrainConfig.lr,
            momentum=cfg.TrainConfig.momentum,
            weight_decay=cfg.TrainConfig.weight_decay,
            alpha=cfg.TrainConfig.alpha,
            centered=cfg.TrainConfig.centered
        )
    else:
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TrainConfig.lr,
            momentum=cfg.TrainConfig.momentum,
            weight_decay=cfg.TrainConfig.weight_decay,
            nesterov=cfg.TrainConfig.use_nesterov
        )

    return optimizer


def get_lr_scheduler(cfg, optimizer: torch.optim, last_epoch: int):
    assert cfg.TrainConfig.lr_scheduler_name in ['multistep', 'steplr', 'cosine'], "LR Scheduler option can be only between [multistep/steplr/cosine]"

    lr_scheduler = cfg.TrainConfig.lr_scheduler_name.lower()
    if 'cosine' in lr_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                  T_max=cfg.TrainConfig.max_iters,
                                                                  eta_min=cfg.TrainConfig.min_lr,
                                                                  last_epoch=last_epoch - 1)
    elif 'multistep' in lr_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=cfg.TrainConfig.learning_rate_steps,
                                                            gamma=cfg.TrainConfig.learning_rate_factor,
                                                            last_epoch=last_epoch - 1)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=cfg.TrainConfig.learning_rate_steps,
                                                       gamma=cfg.TrainConfig.learning_rate_factor,
                                                       last_epoch=last_epoch - 1)

    return lr_scheduler
