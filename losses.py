import numpy as np
import torch
from torch import nn


def supervision(x_hat, x_gt):
    err = np.zeros(shape=17)
    dist = (x_hat - x_gt).norm(dim=-1)
    max_idx = torch.argmax(dist, dim=-1)
    for i in range(x_hat.shape[0]):
        err[max_idx[i]] += 1
    return


def discriminator_loss(score, x_hat, x_gt, m, w=50):
    batch_size = x_hat.shape[0]
    n = m.shape[1]
    mag = 10
    dist = (x_hat - x_gt).norm(dim=-1)
    target = 2 * nn.Sigmoid()(-w * dist)
    different = score - target
    for i in range(batch_size):
        different[i, m[i]] *= mag
    np_target = target.detach().cpu().numpy()
    np_score = score.detach().cpu().numpy()
    return different.abs().mean()


def generator_loss(x, x_hat):
    loss = torch.norm(x - x_hat, dim=-1).mean()

    return loss


def val_generator_loss(x, x_hat):
    loss = torch.norm((x - x_hat), dim=-1).mean()

    return loss


def prob_loss(d_prob):
    loss = -torch.log(d_prob).mean()

    return loss
