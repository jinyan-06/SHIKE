import math
import time

import matplotlib.pyplot as plt
import numpy
import numpy as np
import torch
import torch.nn.functional as F
from sklearn import (manifold)
from torch import nn
from torch.optim import lr_scheduler
import random
from torch.utils.data.sampler import Sampler


def set_seed(seed):
    """
    set fixed seed
    :param seed:
    :return:
    """
    print(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def Normalize(data):
    print(data)
    print(np.max(data), np.min(data))
    m = np.mean(data)
    mx = np.max(data)
    mn = np.min(data)
    new_data = (data - m) / (mx - mn)
    print(new_data)
    print(np.max(new_data), np.min(new_data))
    return new_data


def Sigmoid(data):
    # print(data)
    # print(np.max(data), np.min(data))
    data = 1.0 / (1 + np.exp(data))
    # print(data)
    # print(np.max(data), np.min(data))
    return data


# define "soft" cross-entropy with pytorch tensor operations
def softXEnt(input, target):
    logprobs = torch.nn.functional.log_softmax(input, dim=1)
    logtargets = torch.nn.functional.log_softmax(target, dim=1)
    return -(logtargets * logprobs).sum() / input.shape[0]

def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def soft_entropy(input, target, reduction='mean'):
    """ Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets, can be soft
         size_average: if false, sum is returned instead of mean

    Examples::

        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)

        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = soft_entropy(input, target)
        loss.backward()
    """
    logsoftmax = nn.LogSoftmax(dim=1)
    res = -target * logsoftmax(input)
    if reduction == 'mean':
        return torch.mean(torch.sum(res, dim=1))
    elif reduction == 'sum':
        return torch.sum(torch.sum(res, dim=1))
    else:
        return


class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class CosineAnnealingLRWarmup(lr_scheduler._LRScheduler):
    """
    Cosine Annealing with Warm Up.
    """

    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, warmup_epochs=5, base_lr=0.05, warmup_lr=0.1):
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.warmup_lr = warmup_lr
        super(CosineAnnealingLRWarmup, self).__init__(
            optimizer, last_epoch, verbose=True)

    def get_cos_lr(self):
        return [self.eta_min + (self.warmup_lr - self.eta_min) *
                (1 + math.cos(math.pi * (self.last_epoch -
                 self.warmup_epochs) / (self.T_max - self.warmup_epochs))) / 2
                / self.base_lr * base_lr
                for base_lr in self.base_lrs]

    def get_warmup_lr(self):
        return [((self.warmup_lr - self.base_lr) / (self.warmup_epochs - 1) * (self.last_epoch - 1)
                 + self.base_lr) / self.base_lr * base_lr
                for base_lr in self.base_lrs]

    def get_lr(self):
        assert self.warmup_epochs >= 2
        if self.last_epoch < self.warmup_epochs:
            return self.get_warmup_lr()
        else:
            return self.get_cos_lr()

