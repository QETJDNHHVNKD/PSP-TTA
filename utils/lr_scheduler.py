import torch
import math


class LinearWarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, warmup_start_lr=1e-5, eta_min=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
        
            return [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * self.last_epoch / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        else:

            return [
                self.eta_min + 0.5 * (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
                for base_lr in self.base_lrs
            ]
