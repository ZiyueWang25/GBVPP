import torch


def loss_fnc(y_true, y_pred, weight):
    return (torch.abs(torch.flatten(y_true) - torch.flatten(y_pred)) * torch.flatten(weight)).sum() / weight.sum()