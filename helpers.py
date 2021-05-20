from cifar_fast import (PiecewiseLinear, Crop, FlipLR, Cutout, Transform, normalise, pad, transpose, Timer,
                        union, TableLogger, StatsLogger, SGD, cifar10, trainable_params)
from torch import nn
import torch

class Batches():
    def __init__(self, dataset, batch_size, shuffle, set_random_choices=False, num_workers=0, drop_last=False, device="cuda:0"):
        self.device = device
        self.dataset = dataset
        self.batch_size = batch_size
        self.set_random_choices = set_random_choices
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=shuffle, drop_last=drop_last
        )
    
    def __iter__(self):
        if self.set_random_choices:
            self.dataset.set_random_choices() 
        return ({'input': x.to(self.device), 'target': y.to(self.device).long()} for (x,y) in self.dataloader)
    
    def __len__(self): 
        return len(self.dataloader)

def train_epoch(model, train_batches, test_batches, optimizer_step, timer, test_time_in_total=True):
    train_stats, train_time = run_batches(model, train_batches, True, optimizer_step), timer()
    test_stats, test_time = run_batches(model, test_batches, False), timer(test_time_in_total)
    return { 
        'train time': train_time, 'train loss': train_stats.mean('loss'), 'train acc': train_stats.mean('correct'), 
        'test time': test_time, 'test loss': test_stats.mean('loss'), 'test acc': test_stats.mean('correct'),
        'total time': timer.total_time, 
    }

def train(model, optimizer, train_batches, test_batches, epochs, 
          loggers=(), test_time_in_total=True, timer=None):  
    timer = timer or Timer()
    for epoch in range(epochs):
        epoch_stats = train_epoch(model, train_batches, test_batches, optimizer.step, timer, test_time_in_total=test_time_in_total) 
        summary = union({'epoch': epoch+1, 'lr': optimizer.param_values()['lr']*train_batches.batch_size}, epoch_stats)
        for logger in loggers:
            logger.append(summary)    
    return summary

def run_batches(model, batches, training, optimizer_step=None, stats=None):
    stats = stats or StatsLogger(('loss', 'correct'))
    model.train(training)   
    for batch in batches:
        inp = batch["input"]
        target = batch["target"]
        output = model(inp)
        output = {"loss":loss(output, target), "correct":acc(output, target)}
        stats.append(output) 
        if training:
            output['loss'].sum().backward()
            optimizer_step()
            model.zero_grad() 
    return stats

celoss = nn.CrossEntropyLoss(reduce=False)

def acc(out, target):
    return out.max(dim = 1)[1] == target

def loss(out, target):
    return celoss(out, target)

def conv_bn(c_in, c_out, bn_weight_init=1.0, **kw):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False), 
        batch_norm(c_out, bn_weight_init=bn_weight_init, **kw), 
        nn.ReLU(True)
    )

class Residual(nn.Module):
    def __init__(self, c, **kw):
        super().__init__()
        self.module=nn.Sequential(conv_bn(c, c, **kw), conv_bn(c, c, **kw))
        
    def forward(self, x):
        return x + self.module(x)
    
class Net(nn.Module):
    def __init__(self, w=0.125, features=[64, 128, 256, 512], pool=2):
        super().__init__()
        self.prep = conv_bn(3, features[0])
        self.layer1=nn.Sequential(conv_bn(features[0], features[1]), nn.MaxPool2d(pool), Residual(features[1]))
        self.layer2=nn.Sequential(conv_bn(features[1], features[2]), nn.MaxPool2d(pool))
        self.layer3=nn.Sequential(conv_bn(features[2], features[3]), nn.MaxPool2d(pool), Residual(features[3]))
        self.pool  = nn.MaxPool2d(4)
        self.linear= nn.Linear(features[3], 10, bias=False)
        self.w=w
        
    def forward(self, x):
        x=self.prep(x)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.pool(x)
        x = x.view(x.size(0), x.size(1))
        x=self.linear(x)
        return x*self.w

