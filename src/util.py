import random
import os
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau, CyclicLR
from torch.optim.swa_utils import AveragedModel, SWALR
from transformers import get_cosine_schedule_with_warmup
import pickle


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_swa(model, optimizer, train_size, config):
    ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged:\
            0.4 * averaged_model_parameter + 0.6 * model_parameter
    swa_model = AveragedModel(model, avg_fn=ema_avg)
    epoch_step = train_size / config.batch_size
    swa_scheduler = SWALR(optimizer,
                          anneal_strategy="cos",
                          anneal_epochs=int(epoch_step * (config.epochs - 1)),
                          swa_lr=config.swa_lr)
    return swa_model, None


def do_swa_scheduler(step, swa_scheduler, swa_start_step):
    if (swa_scheduler is not None) and (step >= swa_start_step):
        return True
    else:
        return False

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


def save_pickle(obj, folder_path):
    pickle.dump(obj, open(folder_path, 'wb'), pickle.HIGHEST_PROTOCOL)


def load_pickle(folder_path):
    return pickle.load(open(folder_path, 'rb'))

def smart_avg(inputs, axis=1):
    """Compute the mean of the predictions if there are no outliers,
    or the median if there are outliers.

    Parameter: inputs = ndarray of shape (n_samples, n_folds)"""
    spread = inputs.max(axis=axis) - inputs.min(axis=axis) 
    spread_lim = 0.45
    print(f"Inliers:  {(spread < spread_lim).sum():7} -> compute mean")
    print(f"Outliers: {(spread >= spread_lim).sum():7} -> compute median")
    print(f"Total:    {len(inputs):7}")
    return np.where(spread < spread_lim,
                    np.mean(inputs, axis=axis),
                    np.median(inputs, axis=axis))