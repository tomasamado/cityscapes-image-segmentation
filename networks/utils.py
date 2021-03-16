#!/usr/bin/env python
# coding: utf-8

import numpy as np 
import torch.nn as nn

def normal_init(m, mean=0, std=1):
    """Initialize the weights of a module using a normal distribution 

    Parameters:
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