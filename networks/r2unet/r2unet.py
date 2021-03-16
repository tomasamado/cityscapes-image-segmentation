#!/usr/bin/env python
# coding: utf-8

# In[379]:


import os
import torch.nn as nn
import torch.nn.functional as F
import torch  
from .rrcnblock import RRConv

class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()
        
        self.rrcu = RRConv(in_channel, out_channel)
        self.maxpool = nn.MaxPool2d(2)
        
    def forward(self, x):
        
        x = self.rrcu(x)
        x_up = x
        x = self.maxpool(x)
        
        return x, x_up
    
class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        
        self.convtrans1 = nn.ConvTranspose2d(in_channel, out_channel, 3, stride = 2, padding = 1, output_padding = 1)
        self.rrcu = RRConv(in_channel, out_channel)
        
    def forward(self, x, x_crop):
        
        x = F.relu(self.convtrans1(x))
        x = torch.cat((x_crop, x), dim = 1)
        x = self.rrcu(x)
           
        return x
    
class R2UNet(nn.Module):
    def __init__(self):
        super(R2UNet, self).__init__()
        
        #Conv layer
        
        self.downsample1 = Downsample(3,64)
        self.downsample2 = Downsample(64,128)
        self.downsample3 = Downsample(128,256)
        self.downsample4 = Downsample(256,512)
        self.downsample5 = Downsample(512,1024)
        
        self.upsample1 = Upsample(1024,512)
        self.upsample2 = Upsample(512,256)
        self.upsample3 = Upsample(256,128)
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
        x = self.upsample4(x, x_crop1)
        
        x = F.relu(self.conv1(x))
        
        return x

