#!/usr/bin/env python
# coding: utf-8

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from .r2unet import Downsample, Upsample
from .r2unet.rrcnblock import RRConv
from .hanet import HANet


# +
class R2Unet16HANet(nn.Module):

    def __init__(self, hanet_layers=1):
        assert hanet_layers in [1,2,3], "invalid number of HANet layers"
        
        super(R2Unet16HANet, self).__init__()
        
        self.hanet_layers = hanet_layers
        self.downsample1 = Downsample(3,16)
        self.downsample2 = Downsample(16, 32)
        self.downsample3 = Downsample(32,64)
        self.downsample4 = Downsample(64,128)
        
        self.upsample1 = Upsample(256,128)
       
        self.hanet_L1 = HANet((128,32,64), (64,64,128)) 
        self.upsample2 = Upsample(128,64)
        if hanet_layers >= 2:
            self.hanet_L2 = HANet((64,64,128), (32,128,256))
        self.upsample3 = Upsample(64,32)
        if hanet_layers == 3:
            self.hanet_L3 = HANet((32,128,256), (16,256,512))
        self.upsample4 = Upsample(32,16)   
        
        self.rrcu1 = RRConv(128, 256)
        self.conv1 = nn.Conv2d(16, 19, 1)

    def forward(self, x):

        x, x_crop1 = self.downsample1(x) 
        x, x_crop2 = self.downsample2(x)   
        x, x_crop3 = self.downsample3(x)    
        x, x_crop4 = self.downsample4(x)
        
        x = self.rrcu1(x)
        
        x = self.upsample1(x, x_crop4)   
        
        A1 = self.hanet_L1(x)
        x = self.upsample2(x, x_crop3)
        x = A1 * x
            
        if self.hanet_layers >= 2:
            A2 = self.hanet_L2(x)
            x = self.upsample3(x, x_crop2)
            x = A2 * x
        else:
            x = self.upsample3(x, x_crop2)
            
        if self.hanet_layers == 3:
            A3 = self.hanet_L3(x)        
            x = self.upsample4(x, x_crop1)
            x = A3 * x
        else:
            x = self.upsample4(x, x_crop1)
        
        x = F.relu(self.conv1(x))
        
        return x

    
class R2Unet64HANet(nn.Module):
    
    def __init__(self, hanet_layers=1):
        assert hanet_layers in [1,2,3], "invalid number of HANet layers"
        
        super(R2Unet64HANet, self).__init__()
        
        self.hanet_layers = hanet_layers
        self.downsample1 = Downsample(3,64)
        self.downsample2 = Downsample(64,128)
        self.downsample3 = Downsample(128,256)
        self.downsample4 = Downsample(256,512)
        
        self.upsample1 = Upsample(1024,512)
        self.hanet_L1 = HANet((512,32,64), (256,64,128)) 
        self.upsample2 = Upsample(512,256) 
        
        if hanet_layers >= 2:
            self.hanet_L2 = HANet((256,64,128), (128,128,256))
            
        self.upsample3 = Upsample(256,128)
        if hanet_layers == 3:
            self.hanet_L3 = HANet((128,128,256), (64,256,512))
            
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
        
        A1 = self.hanet_L1(x)
        x = self.upsample2(x, x_crop3)
        x = A1 * x
            
        if self.hanet_layers >= 2:
            A2 = self.hanet_L2(x)
            x = self.upsample3(x, x_crop2)
            x = A2 * x
        else:
            x = self.upsample3(x, x_crop2)
            
        if self.hanet_layers == 3:
            A3 = self.hanet_L3(x)        
            x = self.upsample4(x, x_crop1)
            x = A3 * x
        else:
            x = self.upsample4(x, x_crop1)
       
        x = F.relu(self.conv1(x))
        
        return x
