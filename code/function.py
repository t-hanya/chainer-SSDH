# -*- coding: utf-8 -*-

import chainer.functions as F


def forcing_binary_loss(h):
    """Forcing binary loss function"""
    n_batch = h.data.shape[0]
    n_units = h.data.shape[1]

    loss = -1 * F.sum((h - 0.5) ** 2) / (n_units * n_batch)
    return loss


def balancing_loss(h):
    """50% fire rate loss function"""
    n_batch = h.data.shape[0]
    n_units = h.data.shape[1]

    mean = F.sum(h, axis=1) / n_units
    loss = F.sum((mean - 0.5) ** 2) / n_batch
    return loss


def binarize(h):
    """Convert to binary code vector"""
    # 1 (value >= 0.5)
    # 0 (value < 0.5)
    b = F.floor(h + 0.5)
    return b
