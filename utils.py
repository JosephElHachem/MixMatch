import os
import torch
import numpy as np

def get_duration(t0, t1):
    hours, rem = divmod(t1 - t0, 3600)
    minutes, seconds = divmod(rem, 60)
    current_time = "{:0>2d}:{:0>2d}:{:0>2d}".format(int(hours), int(minutes), int(seconds))
    return current_time

def cross_entropy(pred, soft_targets, one_hot=True):
    if not one_hot:
        one_hot_targets = torch.zeros_like(pred)
        one_hot_targets[np.arange(pred.shape[0]), soft_targets] = 1
        soft_targets = one_hot_targets
    logsoftmax = torch.nn.LogSoftmax(dim=1)
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), axis=1))

def make_path(path):
    print(path)
    print('current dir:', os.getcwd())
    if path is not None and not os.path.isdir(path):
        os.makedirs(path)
