#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Convert caffemodel file to npz format for saving time in traning preparation
"""


from chainer.functions import caffe
from chainer import serializers


ALEXNET_CAFFEMODEL = 'data/bvlc_alexnet.caffemodel'
ALEXNET_NPZ = 'data/bvlc_alexnet.npz'


if __name__ == '__main__':
    alexnet = caffe.CaffeFunction(ALEXNET_CAFFEMODEL)
    serializers.save_npz(ALEXNET_NPZ, alexnet)
