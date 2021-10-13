import numpy as np


def cal_mae_metric(y_true, y_pred, weight):
    return (np.abs(y_true - y_pred) * weight).sum() / weight.sum()