#!/usr/bin/env python
# coding: utf-8

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from .r2unet import Downsample, Upsample
from .r2unet.rrcnblock import RRConv
from .hanet import HANet


class R2UnetHANet(nn.Module):
    def __init__(self):
        super(R2UnetHANet, self).__init__()
        
        #Conv layer
        
        self.downsample1 = Downsample(3,64)
        self.downsample2 = Downsample(64,128)
        self.downsample3 = Downsample(128,256)
        self.downsample4 = Downsample(256,512)
        
        self.upsample1 = Upsample(1024,512)
        self.upsample2 = Upsample(512,256)
        self.upsample3 = Upsample(256,128)
        self.hanet = HANet((128,128,256), (64,256,512))
        self.upsample4 = Upsample(128,64)     
        
        self.rrcu1 = RRConv(512, 1024)
        self.conv1 = nn.Conv2d(64, 19, 1)

        

    def forward(self, x):

        x, x_crop1 = self.downsample1(x) 
        x, x_crop2 = self.downsample2(x)   
        x, x_crop3 = self.downsample3(x)    
        x, x_crop4 = self.downsample4(x)
        
        x = self.rrcu1(x)
        
        x = self.upsample1(x, x_crop4)
        x = self.upsample2(x, x_crop3)
        x = self.upsample3(x, x_crop2)
        A = self.hanet(x)
        #print("A = ", A.size())
        #print("before = ", x.size())
        x = self.upsample4(x, x_crop1)
        #print("after = ", x.size())
        x = A * x
        x = F.relu(self.conv1(x))
        
        return x
