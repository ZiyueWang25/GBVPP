
import torch
from torch import nn


def get_loss(config):
    if config.loss_fnc == "ce":
        return cal_ce_loss
    elif config.loss_fnc == "mae":
        return cal_mae_loss
    elif config.loss_fnc == "huber":
        return lambda ytrue, ypred, weight, use_in_phase_only: cal_mae_loss(ytrue, ypred, weight, use_in_phase_only,
                                                                            config.delta)


def cal_ce_loss(y_true, y_pred, weight, use_in_phase_only):
    return nn.CrossEntropyLoss()(y_pred.reshape(-1, 950), y_true.reshape(-1))


def cal_mae_loss(y_true, y_pred, weight, use_in_phase_only, config=None):
    if use_in_phase_only:
        return (torch.abs(y_true - y_pred) * weight).sum() / weight.sum()
    else:
        return (torch.abs(y_true - y_pred)).sum() / (weight>=0).sum()


def cal_huber_loss(y_true, y_pred, weight, use_in_phase_only, delta=1):
    return nn.HuberLoss(delta=delta)(y_true, y_pred)
