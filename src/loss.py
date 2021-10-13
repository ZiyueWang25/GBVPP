import torch


def cal_mae_loss(y_true, y_pred, weight):
    return (torch.abs(y_true - y_pred) * weight).sum() / weight.sum()