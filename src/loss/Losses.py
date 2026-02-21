# Copyright (c) Yonglan Liu
# Licensed under the MIT License.

import torch
import torch.nn.functional as F

# ===========================================
#   Losses (regression + selectivity delta)
# ===========================================
def masked_huber(pred, y, mask, delta=1.0):
    """
    huber_loss:
    MSE (L2 loss): (1/2)* e^2 when errors are small. 
                → encourages precise fitting
                → smooth gradient near zero
    MAE (L1 loss): delta * (|e| - (1/2) * delta) when errors are large
                → prevents exploding gradients
                → reduces impact of outliers
    """
    hub = F.huber_loss(pred, y, reduction="none", delta=delta)
    hub = hub * mask
    denom = mask.sum().clamp_min(1.0)
    return hub.sum() / denom


def selectivity_delta_loss(pred, y, mask, huber_delta=1.0):
    # pred,y,mask: [B,T]
    B, T = pred.shape
    loss_sum = pred.new_tensor(0.0)
    count = pred.new_tensor(0.0)

    for i in range(T):
        for j in range(T):
            if i == j:
                continue
            valid = (mask[:, i] > 0) & (mask[:, j] > 0)
            if not valid.any():
                continue
            true_delta = y[valid, i] - y[valid, j]
            pred_delta = pred[valid, i] - pred[valid, j]
            l = F.huber_loss(pred_delta, true_delta, reduction="none", delta=huber_delta)
            loss_sum += l.sum()
            count += l.numel()

    return loss_sum / count.clamp_min(1.0)


def loss_regression_plus_selectivity(pred, y, mask, lambda_delta=0.5, huber_delta=1.0):
    pt = masked_huber(pred, y, mask, delta=huber_delta)
    sel = selectivity_delta_loss(pred, y, mask, huber_delta=huber_delta)
    return pt + lambda_delta * sel
