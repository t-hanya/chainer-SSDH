#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training script for SSDH model on CIFAR-10 dataset
"""


import argparse

import chainer
from chainer import serializers
from chainer import training
from chainer.training import extensions

from dataset import CIFAR10Datset
from extension import StepShift
from model import AlexNet
from model import SSDH


def check_args():
    parser = argparse.ArgumentParser(
        description='Training script of DenseNet on CIFAR-10 dataset')
    parser.add_argument('--epoch', '-e', type=int, default=32,
                        help='Number of epochs to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='Validation minibatch size')
    parser.add_argument('--resume', '-r', default='',
                        help='Initialize the trainer from given file')
    parser.add_argument('--units', '-u', type=int, default=12,
                        help='Length of output binary code')
    args = parser.parse_args()
    return args


def main():
    args = check_args()

    # prepare dataset
    c10_train, c10_test = chainer.datasets.cifar.get_cifar10()
    train = CIFAR10Datset(c10_train, random=True)
    test = CIFAR10Datset(c10_test, random=False)

    train_iter = chainer.iterators.MultiprocessIterator(train, args.batchsize)
    test_iter = chainer.iterators.MultiprocessIterator(test, args.batchsize,
                                                       repeat=False,
                                                       shuffle=False)

    # setup model
    alexnet = AlexNet()
    serializers.load_npz('data/bvlc_alexnet.npz', alexnet)

    model = SSDH(alexnet, n_units=args.units)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    eval_model = model.copy()
    eval_model.train = False

    # setup optimizer
    optimizer = chainer.optimizers.MomentumSGD(lr=0.001, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))

    # setup trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'),
                               out=args.out)

    trainer.extend(extensions.Evaluator(test_iter, eval_model,
                                        device=args.gpu))
    trainer.extend(extensions.snapshot())
    trainer.extend(extensions.snapshot_object(
                   model, 'model_{.updater.epoch}.npz'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch',
         'main/cls-loss', 'validation/main/cls-loss',
         'main/binary-loss', 'validation/main/binary-loss',
         'main/50%-loss', 'validation/main/50%-loss',
         'main/accuracy', 'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar())

    # lr_policy: "step", stepsize=25000, gamma=0.1
    shifts = [(25000, 0.0001)]
    trainer.extend(StepShift('lr', shifts, optimizer))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    # start training
    trainer.run()

if __name__ == '__main__':
    main()
