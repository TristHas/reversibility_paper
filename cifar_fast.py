from inspect import signature
from collections import namedtuple
import time
import numpy as np
import pandas as pd
from functools import singledispatch

import torch
from torch import nn
import torchvision

torch.backends.cudnn.benchmark = True

#####################
# utils
#####################

class Timer():
    def __init__(self):
        self.times = [time.time()]
        self.total_time = 0.0

    def __call__(self, include_in_total=True):
        self.times.append(time.time())
        delta_t = self.times[-1] - self.times[-2]
        if include_in_total:
            self.total_time += delta_t
        return delta_t

class TableLogger():
    def append(self, output):
        if not hasattr(self, 'keys'):
            self.keys = output.keys()
            print(*(f'{k:>12s}' for k in self.keys))
        filtered = [output[k] for k in self.keys]
        print(*(f'{v:12.4f}' if isinstance(v, np.float) else f'{v:12}' for v in filtered))

#####################
## data augmentation
#####################

cifar10_mean = (0.4914, 0.4822, 0.4465) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616) # equals np.std(train_set.train_data, axis=(0,1,2))/255

def normalise(x, mean=cifar10_mean, std=cifar10_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean*255
    x *= 1.0/(255*std)
    return x

def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border), (0, 0)], mode='reflect')

def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target]) 

class Crop(namedtuple('Crop', ('h', 'w'))):
    def __call__(self, x, x0, y0):
        return x[:,y0:y0+self.h,x0:x0+self.w]

    def options(self, x_shape):
        C, H, W = x_shape
        return {'x0': range(W+1-self.w), 'y0': range(H+1-self.h)}
    
    def output_shape(self, x_shape):
        C, H, W = x_shape
        return (C, self.h, self.w)
    
class FlipLR(namedtuple('FlipLR', ())):
    def __call__(self, x, choice):
        return x[:, :, ::-1].copy() if choice else x 
        
    def options(self, x_shape):
        return {'choice': [True, False]}

class Cutout(namedtuple('Cutout', ('h', 'w'))):
    def __call__(self, x, x0, y0):
        x = x.copy()
        x[:,y0:y0+self.h,x0:x0+self.w].fill(0.0)
        return x

    def options(self, x_shape):
        C, H, W = x_shape
        return {'x0': range(W+1-self.w), 'y0': range(H+1-self.h)} 
    
class Transform():
    def __init__(self, dataset, transforms):
        self.dataset, self.transforms = dataset, transforms
        self.choices = None
        
    def __len__(self):
        return len(self.dataset)
           
    def __getitem__(self, index):
        data, labels = self.dataset[index]
        for choices, f in zip(self.choices, self.transforms):
            args = {k: v[index] for (k,v) in choices.items()}
            data = f(data, **args)
        return data, labels
    
    def set_random_choices(self):
        self.choices = []
        x_shape = self.dataset[0][0].shape
        N = len(self)
        for t in self.transforms:
            options = t.options(x_shape)
            x_shape = t.output_shape(x_shape) if hasattr(t, 'output_shape') else x_shape
            self.choices.append({k:np.random.choice(v, size=N) for (k,v) in options.items()})

union = lambda *dicts: {k: v for d in dicts for (k, v) in d.items()}

#####################
## training utils
#####################

@singledispatch
def cat(*xs):
    raise NotImplementedError
    
@singledispatch
def to_numpy(x):
    raise NotImplementedError

class PiecewiseLinear(namedtuple('PiecewiseLinear', ('knots', 'vals'))):
    def __call__(self, t):
        return np.interp([t], self.knots, self.vals)[0]

class StatsLogger():
    def __init__(self, keys):
        self._stats = {k:[] for k in keys}

    def append(self, output):
        for k,v in self._stats.items():
            v.append(output[k].detach())
    
    def stats(self, key):
        return cat(*self._stats[key])
        
    def mean(self, key):
        return np.mean(to_numpy(self.stats(key)), dtype=np.float)



#####################
## dataset
#####################

def cifar10(root):
    train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=True)
    return {
        'train': {'data': train_set.train_data, 'labels': train_set.train_labels},
        'test': {'data': test_set.test_data, 'labels': test_set.test_labels}
    }

#####################
## data loading
#####################

trainable_params = lambda model:filter(lambda p: p.requires_grad, model.parameters())

class TorchOptimiser():
    def __init__(self, weights, optimizer, step_number=0, **opt_params):
        self.weights = weights
        self.step_number = step_number
        self.opt_params = opt_params
        self._opt = optimizer(weights, **self.param_values())
    
    def param_values(self):
        return {k: v(self.step_number) if callable(v) else v for k,v in self.opt_params.items()}
    
    def step(self):
        self.step_number += 1
        self._opt.param_groups[0].update(**self.param_values())
        self._opt.step()

    def __repr__(self):
        return repr(self._opt)
        
def SGD(weights, lr=0, momentum=0, weight_decay=0, dampening=0, nesterov=False):
    return TorchOptimiser(weights, torch.optim.SGD, lr=lr, momentum=momentum, 
                          weight_decay=weight_decay, dampening=dampening, 
                          nesterov=nesterov)