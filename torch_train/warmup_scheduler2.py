import tensorflow.compat.v1 as tf
import torch

class WarmupScheduler2():
    def __init__(self, optimizer, warmup_steps, init_lr, final_step):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.init_lr = init_lr
        self.final_step = final_step

    def step(self, step):
        if step < self.warmup_steps:
            lr = self.init_lr * step / self.warmup_steps
        else:
            lr = self.init_lr * (1.0 - (step - self.warmup_steps) / (self.final_step-self.warmup_steps))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
