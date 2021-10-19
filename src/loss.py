
import torch
from torch import nn


def cal_ce_loss(y_true, y_pred, weight):
    loss = nn.CrossEntropyLoss()(y_pred.reshape(-1, 950), y_true.reshape(-1), weight.reshape(-1))
    return loss


def cal_mae_loss(y_true, y_pred, weight):
    return (torch.abs(y_true - y_pred) * weight).sum() / weight.sum()