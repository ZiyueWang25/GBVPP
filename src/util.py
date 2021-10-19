import random
import os
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau, CyclicLR
from transformers import get_cosine_schedule_with_warmup


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_scheduler(optimizer, train_size, config):
    if config.scheduler == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config.factor,
                                      patience=config.patience, verbose=True)
    elif config.scheduler == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer,
                                      T_max=config.T_max,
                                      eta_min=config.min_lr, last_epoch=-1)
    elif config.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer,
                                                T_0=config.T_0,
                                                T_mult=1,
                                                eta_min=config.min_lr,
                                                last_epoch=-1)
    elif config.scheduler == 'CyclicLR':
        iter_per_ep = train_size / config.batch_size
        step_size_up = int(iter_per_ep * config.step_up_epochs)
        step_size_down = int(iter_per_ep * config.step_down_epochs)
        scheduler = CyclicLR(optimizer,
                             base_lr=config.base_lr,
                             max_lr=config.max_lr,
                             step_size_up=step_size_up,
                             step_size_down=step_size_down,
                             mode=config.mode,
                             gamma=config.cycle_decay ** (1 / (step_size_up + step_size_down)),
                             cycle_momentum=False)
    elif config.scheduler == 'cosineWithWarmUp':
        epoch_step = train_size / config.batch_size
        num_warmup_steps = int(epoch_step * config.warmup)
        num_training_steps = int(epoch_step * config.epochs)
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_training_steps)
    return scheduler


def get_param_size(p):
    nn = 1
    for s in list(p.size()):
        nn = nn * s
    return nn


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        pp += get_param_size(p)
    return pp


def get_key(path):
    f = open(path, "r")
    key = f.read().strip()
    f.close()
    return key


def class2dict(f):
    return dict((name, getattr(f, name)) for name in dir(f) if not name.startswith('__'))


def get_device(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in config.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print('Number of device:', torch.cuda.device_count())
    return device



