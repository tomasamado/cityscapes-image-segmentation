#!/usr/bin/env python
# coding: utf-8
import os
import torch.nn as nn
import torch.nn.functional as F
import torch


class RConv(nn.Module):
    """ Recurrent convolutional block. """
    
    def __init__(self, in_channels, out_channels):
        
        """ Initialize an instance of RConv. 

            Parameters:
                in_channels: number of channels from the input
                out_channels: desired number of channels to output
        """
        super(RConv, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding = 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding = 1)
        
    def forward(self, x):
        
        x_f = self.conv1(x)        
        h1 = self.conv2(x_f)   
        h2 = self.conv2(h1) + x_f 
        h3 = self.conv2(h2) + x_f      
        
        return h3

class RRConv(nn.Module):
    """ Recurrent residual convolutional block. """
    
    def __init__(self, in_channels, out_channels):
        
         """ Initialize an instance of RConv. 

            Parameters:
                in_channels: number of channels from the input
                out_channels: desired number of channels to output
        """
        super(RRConv, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding = 1)
        self.rconv1 = RConv(in_channels, out_channels)
        self.rconv2 = RConv(out_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):   
        
        #Identity convolution to add later
        iden = self.conv1(x)
        
        x = F.relu(self.bn1(self.rconv1(x)))
        x = F.relu(self.bn1(self.rconv2(x)))
        
        #Original input is added
        x = x + iden
        x = F.relu(x)
        
        return x       
