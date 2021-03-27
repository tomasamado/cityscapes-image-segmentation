#!/usr/bin/env python
# coding: utf-8

import numpy as np 
import torch.nn as nn

def normal_init(m, mean=0, std=1):
    """Initialize the weights of a module using a normal distribution 

    Args:
        m (nn.Module) - Module to be initialize
        mean (number) - Mean of the normal distribution.
        std (number) - Standard deviation of the normal distribution.

    Ref: https://discuss.pytorch.org/t/pytorch-how-to-initialize-weights/81511/2
    """
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()

def count_parameters(model, only_trainable=False):
    """ Count the number of parameters of a model.
    
        Args:
            model (torch.nn.Module) - Model
            only_trainable (bool) - Determines if only the trainable 
                parameters are counted or not. Default: False
        
        Returns:
            (int) - Count
    """
    if only_trainable:
        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        count = sum(p.numel() for p in model.parameters())
    return count