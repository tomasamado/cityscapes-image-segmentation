#!/usr/bin/env python
# coding: utf-8

from .r2unet import Downsample, Upsample
from .hanet import HANet


class R2UnetHANet()
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
