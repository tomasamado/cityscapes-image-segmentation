#!/usr/bin/env python
# coding: utf-8
"""
    segnet module
    
    Based on `SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image 
    Segmentation <http://https://https://arxiv.org/abs/1511.00561>`_
"""

import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models 
from utils import normal_init

def load_pretrained_vg166_bn():
    """ Loads the pretrained VGG-16 with Batch Normalization model """
    if os.path.isfile('models/vgg16_bn/hub/checkpoints/vgg16_bn-6c64b313.pth'):
        checkpoint = torch.load('models/vgg16_bn/hub/checkpoints/vgg16_bn-6c64b313.pth')
        vgg16_bn = models.vgg16_bn()
        vgg16_bn.load_state_dict(checkpoint)

    else: # Download model if it doesn't exist
        os.environ['TORCH_HOME'] = 'models/vgg16_bn'
        vgg16_bn = models.vgg16_bn(pretrained=True, progress=False)

    return vgg16_bn

class conv2DBatchNormRelu(nn.Module):
    """ Conv2d - Batch Normalization - ReLU module 
        
        Ref: https://www.programmersought.com/article/28751510513/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                                    dilation=1,    bias=True, batchnorm=True):
        """ Initialize a conv2DBatchNormRelu module 

            Parameters: 
                in_channels (int) – Number of channels in the input image
                out_channels (int) – Number of channels produced by the convolution
                kernel_size (int or tuple) – Size of the convolving kernel
                stride (int or tuple, optional) – Stride of the convolution. Default: 1
                padding (int or tuple, optional) – Zero-padding added to both sides of 
                    the input. Default: 0
                dilation (int or tuple, optional) – Spacing between kernel elements. 
                    Default: 1
                bias (bool, optional) – If True, adds a learnable bias to the output. 
                    Default: True
                batchnorm (bool, optional) – If True, adds a torch.nn.BatchNorm2d(out_channels) 
                    layer to the module. Default: True
        """
        super(conv2DBatchNormRelu, self).__init__()
        if batchnorm:
                self.block = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, 
                                                padding=padding, bias=bias, dilation=dilation),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                )
        else:
                self.block = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, 
                                                padding=padding, bias=bias, dilation=dilation),
                        nn.ReLU(inplace=True)
                )
    
    def forward(self, inputs):
        """ Compute a forward pass.
            Parameters: 
                inputs (torch.Tensor) - Inputs
            
            Returns:
                outputs (torch.Tensor) - Output of the module
        """
        outputs = self.block(inputs)
        return outputs


class segnetEncoderBlock2(nn.Module):
    """Segnet-Basic Encoder Block module"""

    def __init__(self, in_channels, out_channels):
        """ Create an instance of segnetEncoderBlock.

            Parameters: 
                in_channels (int): number of input channels.
                out_channels (int): number of output channels.
        """
        super(segnetEncoderBlock2, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_channels, out_channels, 
                                            kernel_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(out_channels, out_channels, 
                                            kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, inputs):
        """ Encode the inputs    

            Parameters: 
                inputs (torch.Tensor): inputs to be encoded.

            Returns:
                (torch.Tensor, torch.Tensor, int): encoded input, indices of 
                the MaxPooling operation and output's size
        """
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        output_size = outputs.size()
        outputs, indices = self.maxpool(outputs)
        return outputs, indices, output_size


class segnetEncoderBlock3(nn.Module):
    """Segnet-Basic Encoder Block module"""

    def __init__(self, in_channels, out_channels):
        """ Create an instance of segnetEncoderBlock.

            Parameters: 
                in_channels (int) - Number of input channels.
                out_channels (int) - Number of output channels.
        """
        super(segnetEncoderBlock3, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_channels, out_channels, 
                                            kernel_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(out_channels, out_channels, 
                                            kernel_size=3, stride=1, padding=1)
        self.conv3 = conv2DBatchNormRelu(out_channels, out_channels, 
                                            kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, inputs):
        """ Encode the inputs    

            Parameters: 
                inputs (torch.Tensor) - Inputs to be encoded.

            Returns:
                (torch.Tensor, torch.Tensor, int) - Encoded input, indices of 
                the MaxPooling operation and output's size
        """
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        output_size = outputs.size()
        outputs, indices = self.maxpool(outputs)
        return outputs, indices, output_size


class segnetDecoderBlock2(nn.Module):
    """Segnet-Basic Decoder Block module"""

    def __init__(self, in_channels, out_channels):
        """ Create an instance of segnetDecoderBlock2.

            Parameters: 
                in_channels (int): number of input channels.
                out_channels (int): number of output channels.
        """
        super(segnetDecoderBlock2, self).__init__()
        self.unpool = nn.MaxUnpool2d(2,2)
        self.conv1 = conv2DBatchNormRelu(in_channels, out_channels, 
                            kernel_size=3, stride=1, bias=False, padding=1)
        self.conv2 = conv2DBatchNormRelu(out_channels, out_channels, 
                            kernel_size=3, stride=1, bias=False, padding=1)

    def forward(self, inputs, indices, output_size):
        """ Decode the input. 
            
            Parameters: 
                inputs (torch.Tensor): inputs to be decoded. 
                indices (torch.Tensor): saved indices of the MaxPooling operation.
                output_size (int): the desired output size.
            
            Returns: 
                (torch.Tensor): decoded input.
        """
        outputs = self.unpool(inputs, indices=indices, output_size=output_size)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs


class segnetDecoderBlock3(nn.Module):
    """Segnet-Basic Decoder Block module"""

    def __init__(self, in_channels, out_channels):
        """ Create an instance of segnetDecoderBlock.

            Parameters: 
                in_channels (int): number of input channels.
                out_channels (int): number of output channels.
        """
        super(segnetDecoderBlock3, self).__init__()
        self.unpool = nn.MaxUnpool2d(2,2)
        self.conv1 = conv2DBatchNormRelu(in_channels, out_channels, 
                        kernel_size=3, stride=1, bias=False, padding=1)
        self.conv2 = conv2DBatchNormRelu(out_channels, out_channels, 
                        kernel_size=3, stride=1,bias=False,    padding=1)
        self.conv3 = conv2DBatchNormRelu(out_channels, out_channels, 
                        kernel_size=3, stride=1,bias=False,    padding=1)

    def forward(self, inputs, indices, output_size):
        """ Decode the input. 
            
            Parameters: 
                inputs (torch.Tensor): inputs to be decoded. 
                indices (torch.Tensor): saved indices of the MaxPooling operation.
                output_size (int): the desired output size.
            
            Returns: 
                (torch.Tensor): decoded input.
        """
        outputs = self.unpool(inputs, indices=indices, output_size=output_size)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs


class Segnet(nn.Module):
    """SegNet Class"""

    def __init__(self, in_channels, out_channels, debug=False):
        """Initialize an instance of SegNet

            Parameters:
                in_channels (int): number of input channels 
                out_channels (int): number of output channels
                vgg16_bn (torch.model): pretrained VGG-16 (with Batch Normalization) model
        """
        super(Segnet, self).__init__()
        self.in_channels = in_channels 
        self.out_channels = out_channels 
        self.debug = debug     

        # Encoder (VGG16 without Classifier)
        self.enc_block00 = segnetEncoderBlock2(self.encoder_dims('block00', 'in'), 
                                                self.encoder_dims('block00', 'out'))
        self.enc_block01 = segnetEncoderBlock2(self.encoder_dims('block01', 'in'), 
                                                self.encoder_dims('block01', 'out'))
        self.enc_block02 = segnetEncoderBlock3(self.encoder_dims('block02', 'in'), 
                                                self.encoder_dims('block02', 'out'))
        self.enc_block03 = segnetEncoderBlock3(self.encoder_dims('block03', 'in'), 
                                                self.encoder_dims('block03', 'out'))
        self.enc_block04 = segnetEncoderBlock3(self.encoder_dims('block04', 'in'), 
                                                self.encoder_dims('block04', 'out'))
                               
        # Initialize the encoder's weights
        self._load_encoder_weights()

        # Decoder 
        self.dec_block04 = segnetDecoderBlock3(self.decoder_dims('block04','in'), 
                                                self.decoder_dims('block04','out'))
        self.dec_block03 = segnetDecoderBlock3(self.decoder_dims('block03','in'), 
                                                self.decoder_dims('block03','out'))
        self.dec_block02 = segnetDecoderBlock3(self.decoder_dims('block02','in'), 
                                                self.decoder_dims('block02','out'))
        self.dec_block01 = segnetDecoderBlock2(self.decoder_dims('block01','in'), 
                                                self.decoder_dims('block01','out'))
        self.dec_block00 = segnetDecoderBlock2(self.decoder_dims('block00','in'), 
                                                self.decoder_dims('block00','out'))

        # Initialize the decoder's weights
        normal_init(self.dec_block04)
        normal_init(self.dec_block03)
        normal_init(self.dec_block02)
        normal_init(self.dec_block01)
        normal_init(self.dec_block00)

        # Softmax
        self.softmax = nn.Softmax(dim=1)
         
    def forward(self, x):
        """Compute a forward pass 

            Parameters: 
                x (torch.Tensor): input image(s).

            Returns: 
                (torch.Tensor): pixel-level classification
        """ 
        # Encoder
        enc00, indices00, output_size00 = self.enc_block00(x)
        enc01, indices01, output_size01 = self.enc_block01(enc00)
        enc02, indices02, output_size02 = self.enc_block02(enc01)
        enc03, indices03, output_size03 = self.enc_block03(enc02)
        enc04, indices04, output_size04 = self.enc_block04(enc03)

        # Decoder 
        dec04 = self.dec_block04(enc04, indices04, output_size04)
        dec03 = self.dec_block03(dec04, indices03, output_size03)
        dec02 = self.dec_block02(dec03, indices02, output_size02)
        dec01 = self.dec_block01(dec02, indices01, output_size01)
        dec00 = self.dec_block00(dec01, indices00, output_size00) 

        # Pixel-level classification 
        output = self.softmax(dec00)
        return output
    
    def debug(self):
        """Activate debug mode"""
        self.debug = True 

    def no_debug(self):
        """Deactivate debug mode"""
        self.debug = False

    def debug(self, debug):
        """Configure debug mode """
        self.debug = debug

    def encoder_dims(self, block, io):
        """ Obtain the encoder dimensions based on the input dimensions

            Parameters: 
                block (string): encoder block 
                io (string): 'in' or 'out'

            Returns: 
                (int): input/output dimensions of the corresponding encoder block
        """
        encoder_dimensions = {
            'block00': { 'in': self.in_channels,'out': 64 },
            'block01': { 'in': 64,    'out': 128 },
            'block02': { 'in': 128, 'out': 256 },
            'block03': { 'in': 256, 'out': 512 },
            'block04': { 'in': 512, 'out': 512 }
        }
        return encoder_dimensions[block][io]

    def decoder_dims(self, block, io): 
        """ Obtain the decoder's dimensions based on the output dimensions 

            Parameters: 
                block (string): decoder block 
                io (string): 'in' or 'out'

            Returns: 
                (int): input/output dimensions of the corresponding decoder block
        """
        decoder_dimensions = {
            'block04': { 'in': 512, 'out': 512 },
            'block03': { 'in': 512, 'out': 256 },
            'block02': { 'in': 256, 'out': 128 },
            'block01': { 'in': 128, 'out': 64 },
            'block00': { 'in':    64, 'out': self.out_channels }
        }
        return decoder_dimensions[block][io]
        
    def _load_encoder_weights(self):
        """
            Load the corresponding weights of the train VGG16 model into the encoder.
        """ 
        # Load pretrained model
        vgg16_bn = load_pretrained_vg166_bn()

        # Encoder block00
        assert self.enc_block00.conv1.block[0].weight.size() == vgg16_bn.features[0].weight.size() 
        assert self.enc_block00.conv1.block[0].bias.size() == vgg16_bn.features[0].bias.size() 
        assert self.enc_block00.conv1.block[1].weight.size() == vgg16_bn.features[1].weight.size() 
        assert self.enc_block00.conv1.block[1].bias.size() == vgg16_bn.features[1].bias.size() 
        assert self.enc_block00.conv2.block[0].weight.size() == vgg16_bn.features[3].weight.size() 
        assert self.enc_block00.conv2.block[0].bias.size() == vgg16_bn.features[3].bias.size() 
        assert self.enc_block00.conv2.block[1].weight.size() == vgg16_bn.features[4].weight.size() 
        assert self.enc_block00.conv2.block[1].bias.size() == vgg16_bn.features[4].bias.size()

        self.enc_block00.conv1.block[0].weight.size() == vgg16_bn.features[0].weight.size() 
        self.enc_block00.conv1.block[0].bias.size() == vgg16_bn.features[0].bias.size() 
        self.enc_block00.conv1.block[1].weight.size() == vgg16_bn.features[1].weight.size() 
        self.enc_block00.conv1.block[1].bias.size() == vgg16_bn.features[1].bias.size() 
        self.enc_block00.conv2.block[0].weight.size() == vgg16_bn.features[3].weight.size() 
        self.enc_block00.conv2.block[0].bias.size() == vgg16_bn.features[3].bias.size() 
        self.enc_block00.conv2.block[1].weight.size() == vgg16_bn.features[4].weight.size() 
        self.enc_block00.conv2.block[1].bias.size() == vgg16_bn.features[4].bias.size()

        # Encoder block01
        assert self.enc_block01.conv1.block[0].weight.size() == vgg16_bn.features[7].weight.size() 
        assert self.enc_block01.conv1.block[0].bias.size() == vgg16_bn.features[7].bias.size() 
        assert self.enc_block01.conv1.block[1].weight.size() == vgg16_bn.features[8].weight.size() 
        assert self.enc_block01.conv1.block[1].bias.size() == vgg16_bn.features[8].bias.size() 
        assert self.enc_block01.conv2.block[0].weight.size() == vgg16_bn.features[10].weight.size() 
        assert self.enc_block01.conv2.block[0].bias.size() == vgg16_bn.features[10].bias.size() 
        assert self.enc_block01.conv2.block[1].weight.size() == vgg16_bn.features[11].weight.size() 
        assert self.enc_block01.conv2.block[1].bias.size() == vgg16_bn.features[11].bias.size()
        
        self.enc_block01.conv1.block[0].weight.size() == vgg16_bn.features[7].weight.size() 
        self.enc_block01.conv1.block[0].bias.size() == vgg16_bn.features[7].bias.size() 
        self.enc_block01.conv1.block[1].weight.size() == vgg16_bn.features[8].weight.size() 
        self.enc_block01.conv1.block[1].bias.size() == vgg16_bn.features[8].bias.size() 
        self.enc_block01.conv2.block[0].weight.size() == vgg16_bn.features[10].weight.size() 
        self.enc_block01.conv2.block[0].bias.size() == vgg16_bn.features[10].bias.size() 
        self.enc_block01.conv2.block[1].weight.size() == vgg16_bn.features[11].weight.size() 
        self.enc_block01.conv2.block[1].bias.size() == vgg16_bn.features[11].bias.size()

        # Encoder block02 
        assert self.enc_block02.conv1.block[0].weight.size() == vgg16_bn.features[14].weight.size() 
        assert self.enc_block02.conv1.block[0].bias.size() == vgg16_bn.features[14].bias.size() 
        assert self.enc_block02.conv1.block[1].weight.size() == vgg16_bn.features[15].weight.size() 
        assert self.enc_block02.conv1.block[1].bias.size() == vgg16_bn.features[15].bias.size() 
        assert self.enc_block02.conv2.block[0].weight.size() == vgg16_bn.features[17].weight.size() 
        assert self.enc_block02.conv2.block[0].bias.size() == vgg16_bn.features[17].bias.size() 
        assert self.enc_block02.conv2.block[1].weight.size() == vgg16_bn.features[18].weight.size() 
        assert self.enc_block02.conv2.block[1].bias.size() == vgg16_bn.features[18].bias.size()
        assert self.enc_block02.conv3.block[0].weight.size() == vgg16_bn.features[20].weight.size() 
        assert self.enc_block02.conv3.block[0].bias.size() == vgg16_bn.features[20].bias.size() 
        assert self.enc_block02.conv3.block[1].weight.size() == vgg16_bn.features[21].weight.size() 
        assert self.enc_block02.conv3.block[1].bias.size() == vgg16_bn.features[21].bias.size()

        self.enc_block02.conv1.block[0].weight.size() == vgg16_bn.features[14].weight.size() 
        self.enc_block02.conv1.block[0].bias.size() == vgg16_bn.features[14].bias.size() 
        self.enc_block02.conv1.block[1].weight.size() == vgg16_bn.features[15].weight.size() 
        self.enc_block02.conv1.block[1].bias.size() == vgg16_bn.features[15].bias.size() 
        self.enc_block02.conv2.block[0].weight.size() == vgg16_bn.features[17].weight.size() 
        self.enc_block02.conv2.block[0].bias.size() == vgg16_bn.features[17].bias.size() 
        self.enc_block02.conv2.block[1].weight.size() == vgg16_bn.features[18].weight.size() 
        self.enc_block02.conv2.block[1].bias.size() == vgg16_bn.features[18].bias.size()
        self.enc_block02.conv3.block[0].weight.size() == vgg16_bn.features[20].weight.size() 
        self.enc_block02.conv3.block[0].bias.size() == vgg16_bn.features[20].bias.size() 
        self.enc_block02.conv3.block[1].weight.size() == vgg16_bn.features[21].weight.size() 
        self.enc_block02.conv3.block[1].bias.size() == vgg16_bn.features[21].bias.size()

        # Encoder block03
        assert self.enc_block03.conv1.block[0].weight.size() == vgg16_bn.features[24].weight.size() 
        assert self.enc_block03.conv1.block[0].bias.size() == vgg16_bn.features[24].bias.size() 
        assert self.enc_block03.conv1.block[1].weight.size() == vgg16_bn.features[25].weight.size() 
        assert self.enc_block03.conv1.block[1].bias.size() == vgg16_bn.features[25].bias.size() 
        assert self.enc_block03.conv2.block[0].weight.size() == vgg16_bn.features[27].weight.size() 
        assert self.enc_block03.conv2.block[0].bias.size() == vgg16_bn.features[27].bias.size() 
        assert self.enc_block03.conv2.block[1].weight.size() == vgg16_bn.features[28].weight.size() 
        assert self.enc_block03.conv2.block[1].bias.size() == vgg16_bn.features[28].bias.size()
        assert self.enc_block03.conv3.block[0].weight.size() == vgg16_bn.features[30].weight.size() 
        assert self.enc_block03.conv3.block[0].bias.size() == vgg16_bn.features[30].bias.size() 
        assert self.enc_block03.conv3.block[1].weight.size() == vgg16_bn.features[31].weight.size() 
        assert self.enc_block03.conv3.block[1].bias.size() == vgg16_bn.features[31].bias.size()
        
        self.enc_block03.conv1.block[0].weight.size() == vgg16_bn.features[24].weight.size() 
        self.enc_block03.conv1.block[0].bias.size() == vgg16_bn.features[24].bias.size() 
        self.enc_block03.conv1.block[1].weight.size() == vgg16_bn.features[25].weight.size() 
        self.enc_block03.conv1.block[1].bias.size() == vgg16_bn.features[25].bias.size() 
        self.enc_block03.conv2.block[0].weight.size() == vgg16_bn.features[27].weight.size() 
        self.enc_block03.conv2.block[0].bias.size() == vgg16_bn.features[27].bias.size() 
        self.enc_block03.conv2.block[1].weight.size() == vgg16_bn.features[28].weight.size() 
        self.enc_block03.conv2.block[1].bias.size() == vgg16_bn.features[28].bias.size()
        self.enc_block03.conv3.block[0].weight.size() == vgg16_bn.features[30].weight.size() 
        self.enc_block03.conv3.block[0].bias.size() == vgg16_bn.features[30].bias.size() 
        self.enc_block03.conv3.block[1].weight.size() == vgg16_bn.features[31].weight.size() 
        self.enc_block03.conv3.block[1].bias.size() == vgg16_bn.features[31].bias.size()
        
        # Encoder block04
        assert self.enc_block04.conv1.block[0].weight.size() == vgg16_bn.features[34].weight.size() 
        assert self.enc_block04.conv1.block[0].bias.size() == vgg16_bn.features[34].bias.size() 
        assert self.enc_block04.conv1.block[1].weight.size() == vgg16_bn.features[35].weight.size() 
        assert self.enc_block04.conv1.block[1].bias.size() == vgg16_bn.features[35].bias.size() 
        assert self.enc_block04.conv2.block[0].weight.size() == vgg16_bn.features[37].weight.size() 
        assert self.enc_block04.conv2.block[0].bias.size() == vgg16_bn.features[37].bias.size() 
        assert self.enc_block04.conv2.block[1].weight.size() == vgg16_bn.features[38].weight.size() 
        assert self.enc_block04.conv2.block[1].bias.size() == vgg16_bn.features[38].bias.size()
        assert self.enc_block04.conv3.block[0].weight.size() == vgg16_bn.features[40].weight.size() 
        assert self.enc_block04.conv3.block[0].bias.size() == vgg16_bn.features[40].bias.size() 
        assert self.enc_block04.conv3.block[1].weight.size() == vgg16_bn.features[41].weight.size() 
        assert self.enc_block04.conv3.block[1].bias.size() == vgg16_bn.features[41].bias.size()
        

        self.enc_block04.conv1.block[0].weight.size() == vgg16_bn.features[34].weight.size() 
        self.enc_block04.conv1.block[0].bias.size() == vgg16_bn.features[34].bias.size() 
        self.enc_block04.conv1.block[1].weight.size() == vgg16_bn.features[35].weight.size() 
        self.enc_block04.conv1.block[1].bias.size() == vgg16_bn.features[35].bias.size() 
        self.enc_block04.conv2.block[0].weight.size() == vgg16_bn.features[37].weight.size() 
        self.enc_block04.conv2.block[0].bias.size() == vgg16_bn.features[37].bias.size() 
        self.enc_block04.conv2.block[1].weight.size() == vgg16_bn.features[38].weight.size() 
        self.enc_block04.conv2.block[1].bias.size() == vgg16_bn.features[38].bias.size()
        self.enc_block04.conv3.block[0].weight.size() == vgg16_bn.features[40].weight.size() 
        self.enc_block04.conv3.block[0].bias.size() == vgg16_bn.features[40].bias.size() 
        self.enc_block04.conv3.block[1].weight.size() == vgg16_bn.features[41].weight.size() 
        self.enc_block04.conv3.block[1].bias.size() == vgg16_bn.features[41].bias.size()
        
        if self.debug: 
            print("VGG-16 weights loaded")
