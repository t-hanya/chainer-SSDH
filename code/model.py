# -*- coding: utf-8 -*-


import chainer
import chainer.functions as F
import chainer.links as L

from function import balancing_loss
from function import binarize
from function import forcing_binary_loss


class AlexNet(chainer.Chain):

    input_size = 227

    def __init__(self):
        super(AlexNet, self).__init__(
            conv1=L.Convolution2D(None,  96, 11, stride=4),
            conv2=L.Convolution2D(None, 256,  5, pad=2),
            conv3=L.Convolution2D(None, 384,  3, pad=1),
            conv4=L.Convolution2D(None, 384,  3, pad=1),
            conv5=L.Convolution2D(None, 256,  3, pad=1),
            fc6=L.Linear(None, 4096),
            fc7=L.Linear(None, 4096),
            # fc8 is removed
        )

    def __call__(self, x, train=True):
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.dropout(F.relu(self.fc6(h)), train=train)
        h = F.dropout(F.relu(self.fc7(h)), train=train)

        return h


class SSDH(chainer.Chain):

    """Supervised Semantics-preserving Deep Hashing model

    see: https://arxiv.org/abs/1507.00101

    Attributes:
        n_units(int): Length of binary code
        n_class(int): Number of classes in training set
    """

    def __init__(self, cnn, n_units=12, n_class=10,
                 alpha=1., beta=1., gamma=1.):

        super(SSDH, self).__init__(
            cnn=cnn,
            latent=L.Linear(4096, n_units),
            fc_cls=L.Linear(n_units, n_class),
        )
        self.train = True
        self.add_persistent('alpha', alpha)
        self.add_persistent('beta', beta)
        self.add_persistent('gamma', gamma)

    def __call__(self, x, t=None):

        h = self.cnn(x, train=self.train)
        h_bin = F.sigmoid(self.latent(h))
        h_cls = self.fc_cls(h_bin)

        if not t is None:
            l_cls = self.alpha * F.softmax_cross_entropy(h_cls, t)
            l_force_bin = self.beta * forcing_binary_loss(h_bin)
            l_balance = self.gamma * balancing_loss(h_bin)

            loss = l_cls + l_force_bin + l_balance

            chainer.report({'cls-loss': l_cls,
                            'binary-loss': l_force_bin,
                            '50%-loss': l_balance,
                            'accuracy': F.accuracy(h_cls, t)}, self)
            return loss

        else:
            return binarize(h_bin)
