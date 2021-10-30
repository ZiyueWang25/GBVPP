
import torch
from torch import nn


def get_loss(config):
    if config.loss_fnc == "ce":
        return cal_ce_loss
    elif config.loss_fnc == "ce_custom":
        return cal_ce_loss_custom
    elif config.loss_fnc == "mae":
        return cal_mae_loss
    elif config.loss_fnc == "huber":
        return lambda ytrue, ypred, weight: cal_huber_loss(ytrue, ypred, weight, config.delta)


def cal_ce_loss(y_true, y_pred, weight):
    return nn.CrossEntropyLoss()(y_pred.reshape(-1, 950), y_true.reshape(-1))

def cal_ce_loss_custom(y_true, y_pred, weight):
    y_pred = nn.Softmax(dim=-1)(y_pred.reshape(-1,950))
    y_true = y_true.reshape(-1)
    weight = weight.reshape(-1)
    probs = y_pred[torch.arange(0, y_pred.shape[0]), y_true.reshape(-1)].reshape(-1)
    probs, weights = probs[probs==probs], weight[probs==probs]
    loss = - (torch.log(probs) * weights).sum() / weights.sum()
    return loss

def cal_mae_loss(y_true, y_pred, weight):
    y_true, y_pred, weight = y_true[y_pred == y_pred], y_pred[y_pred == y_pred], weight[y_pred == y_pred]    
    return (torch.abs(y_true - y_pred) * weight).sum() / weight.sum()


def cal_huber_loss(y_true, y_pred, weight, delta=1):
    y_true, y_pred, weight = y_true[y_pred == y_pred], y_pred[y_pred == y_pred], weight[y_pred == y_pred]    
    return nn.HuberLoss(delta=delta)(y_true * weight, y_pred)
