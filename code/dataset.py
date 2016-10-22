# -*- coding: utf-8 -*-
import os
import random

from chainer.dataset import DatasetMixin
import numpy as np
from scipy.misc import imresize


code_dir = os.path.split(__file__)[0]
data_dir = os.path.abspath(os.path.join(code_dir, '..', 'data'))
mean_image_path = os.path.join(data_dir, 'ilsvrc_2012_mean.npy')


def resize(img, size=256):
    """Resize image"""
    img = img.transpose(1, 2, 0)  # (C, H, W) => (H, W, C)
    img = imresize(img, (size, size))
    img = img.transpose(2, 0, 1)  # (H, W, C) => (C, H, B)
    img = img.astype(np.float32)

    return img


def random_crop(img, crop_size=227):
    """Crop image randomly"""
    w, h = img.shape[2], img.shape[1]

    x0 = random.randint(0, w - crop_size)
    y0 = random.randint(0, h - crop_size)
    img = img[:, y0: y0 + crop_size, x0: x0 + crop_size]

    return img


def crop_center(img, crop_size=227):
    """Crop center of given image"""
    w, h = img.shape[2], img.shape[1]

    x0 = int((w - crop_size) / 2.)
    y0 = int((h - crop_size) / 2.)
    img = img[:, y0: y0 + crop_size, x0: x0 + crop_size]

    return img


class CIFAR10Datset(DatasetMixin):
    """CIFAR10 dataset for SSDH training"""

    def __init__(self, dataset, random=False):
        self._dataset = dataset
        self._random = random

        mean = np.load(mean_image_path).astype(np.float32)
        self._mean = mean

    def __len__(self):
        return len(self._dataset)

    def get_example(self, i):
        img, label = self._dataset[i]

        # preprocess
        img = resize(img)
        img = img - self._mean
        if self._random:
            img = random_crop(img)
            if random.random() > 0.5:
                img = img[:, :, ::-1]  # flip horizontal
        else:
            img = crop_center(img)

        return img, label
