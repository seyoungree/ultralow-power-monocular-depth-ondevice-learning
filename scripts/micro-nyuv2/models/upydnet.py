'''
Copyright (C) 2024 University of Bologna

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

'''
Authors: Davide Nadalini (d.nadalini@unibo.it)
SOURCES: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9422776
'''

import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


''' MODULES '''
# First block of uPyD-Net
class ShallowEncoder(nn.Module):

    def __init__(self, in_ch, cout0, cout1, cout2):
        
        super(ShallowEncoder, self).__init__()

        self.conv0 = nn.Conv2d(in_channels = in_ch, out_channels = cout0,  kernel_size = 3, padding = 1, stride = 1)
        self.conv1 = nn.Conv2d(in_channels = cout0, out_channels = cout0,  kernel_size = 3, padding = 1, stride = 1)
        self.conv2 = nn.Conv2d(in_channels = cout0, out_channels = cout1,  kernel_size = 3, padding = 1, stride = 2)
        self.conv3 = nn.Conv2d(in_channels = cout1, out_channels = cout1,  kernel_size = 3, padding = 1, stride = 1)
        self.conv4 = nn.Conv2d(in_channels = cout1, out_channels = cout2,  kernel_size = 3, padding = 1, stride = 2)
        self.conv5 = nn.Conv2d(in_channels = cout2, out_channels = cout2,  kernel_size = 3, padding = 1, stride = 1)
        self.leakyrelu = nn.LeakyReLU(negative_slope = 0.125)

    def forward(self, x):
        y0 = self.leakyrelu(self.conv0(x))
        y0 = self.leakyrelu(self.conv1(y0))
        pyd0 = y0
        y1 = self.leakyrelu(self.conv2(y0))
        y1 = self.leakyrelu(self.conv3(y1))
        pyd1 = y1
        y2 = self.leakyrelu(self.conv4(y1))
        y2 = self.leakyrelu(self.conv5(y2))
        pyd2 = y2
        return pyd0, pyd1, pyd2

# Decoders of uPyd-Net
class PDecoder(nn.Module):

    def __init__(self, in_ch, mid_ch, out_ch):

        super(PDecoder, self).__init__()

        self.conv0 = nn.Conv2d(in_channels = in_ch,  out_channels = mid_ch, kernel_size = 3, padding = 1, stride = 1)
        self.conv1 = nn.Conv2d(in_channels = mid_ch, out_channels = mid_ch, kernel_size = 3, padding = 1, stride = 1)
        self.conv2 = nn.Conv2d(in_channels = mid_ch, out_channels = out_ch, kernel_size = 3, padding = 1, stride = 1)
        self.leakyrelu = nn.LeakyReLU(negative_slope = 0.125)

    def forward(self, x):
        y = self.leakyrelu(self.conv0(x))
        y = self.leakyrelu(self.conv1(y))
        y = self.conv2(y)
        return y

    
# Transposed convolution (https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html)
# Size of output (same for Wo) + example: 
# Ho = (Hi-1)*str - 2*pad + (ker-1) + outpad + 1
# 32 = (16-1)*2   - 0*1   + (2-1)   + 0      + 1 = 
#    = 30         - 0     + 1       + 0      + 1 
class Upsampler(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(Upsampler, self).__init__()
        self.tconv = nn.ConvTranspose2d(
            in_channels    = in_ch,
            out_channels   = out_ch,
            kernel_size    = 2,
            stride         = 2,
            padding        = 0,
            output_padding = 0,
            groups         = 1
        )

    def forward(self, x):
        y = self.tconv(x)
        return y




''' MODELS '''

# Standard uPyD-Net
class uPydNet(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(uPydNet, self).__init__()    
        
        self.encoder  = ShallowEncoder(in_ch, 8, 16, 32)
        self.decoder0 = PDecoder(32, 32, 32)
        self.decoder1 = PDecoder(48, 32, 32)
        self.decoder2 = PDecoder(40, 32, out_ch)
        self.ups0     = Upsampler(32, 32)
        self.ups1     = Upsampler(32, 32)
        self.relu     = nn.ReLU()

    def forward(self, x):
        pyd0, pyd1, pyd2 = self.encoder(x)
        dc0     = self.ups0(self.decoder0(pyd2))
        concat0 = torch.cat((pyd1, dc0), 1)     # Concat on channels
        dc1     = self.ups1(self.decoder1(concat0))
        concat1 = torch.cat((pyd0, dc1), 1)     # Concat on channels
        dc2     = self.decoder2(concat1)
        dc2     = self.relu(dc2)
        return dc2
    

# uPyD-net large
class uPydNet_L(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(uPydNet_L, self).__init__()    
        
        self.encoder  = ShallowEncoder(in_ch, 32, 64, 96)
        self.decoder0 = PDecoder(96, 96, 96)
        self.decoder1 = PDecoder(160, 96, 96)
        self.decoder2 = PDecoder(128, 96, out_ch)
        self.ups0     = Upsampler(96, 96)
        self.ups1     = Upsampler(96, 96)
        self.relu     = nn.ReLU()

    def forward(self, x):
        pyd0, pyd1, pyd2 = self.encoder(x)
        dc0     = self.ups0(self.decoder0(pyd2))
        concat0 = torch.cat((pyd1, dc0), 1)     # Concat on channels
        dc1     = self.ups1(self.decoder1(concat0))
        concat1 = torch.cat((pyd0, dc1), 1)     # Concat on channels
        dc2     = self.decoder2(concat1)
        dc2     = self.relu(dc2)
        return dc2
    

# uPyD-Net targeting 8x8 depth map from a 320x320 input
class uPydNet_8x8(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(uPydNet_8x8, self).__init__()    
        
        self.encoder  = ShallowEncoder(in_ch, 8, 16, 32)
        self.decoder0 = PDecoder(32, 32, 32)
        self.decoder1 = PDecoder(48, 32, 32)
        self.decoder2 = PDecoder(40, 32, out_ch)
        self.ups0     = Upsampler(32, 32)
        self.ups1     = Upsampler(32, 32)
        self.outpool  = nn.AvgPool2d(kernel_size = 40, stride = 40)
        self.relu     = nn.ReLU()

    def forward(self, x):
        pyd0, pyd1, pyd2 = self.encoder(x)
        dc0     = self.ups0(self.decoder0(pyd2))
        concat0 = torch.cat((pyd1, dc0), 1)     # Concat on channels
        dc1     = self.ups1(self.decoder1(concat0))
        concat1 = torch.cat((pyd0, dc1), 1)     # Concat on channels
        dc2     = self.decoder2(concat1)
        dc2     = self.relu(dc2)
        out8x8  = self.outpool(dc2)
        return out8x8






''' TEST MODEL '''
if __name__ == '__main__':

    # Test modules
    x_enc = torch.ones(1,  1, 16, 16)
    x_dec = torch.ones(1, 32, 16, 16)
    x_ups = torch.ones(1,  1, 16, 16)

    print("\n")
    print(f"Size of x_enc is {x_enc.size()}")
    print(f"Size of x_dec is {x_dec.size()}")
    print(f"Size of x_ups is {x_ups.size()}")
    print("\n")

    enc = ShallowEncoder(1, 8, 16, 32)
    dec = PDecoder(32, 32, 32)
    ups = Upsampler(1, 1)

    with torch.no_grad():
        y_enc0, y_enc1, y_enc2 = enc(x_enc)
        print(f"Sizes of y_enc are {y_enc0.size()}, {y_enc1.size()}, {y_enc2.size()}")
        y_dec = dec(x_dec)
        print(f"Size  of y_dec  is {y_dec.size()}")
        y_ups = ups(x_ups)
        print(f"Size  of y_ups  is {y_ups.size()}")

    print("\n")


    # Test complete model
    x_upyd  = torch.ones(1, 1, 64, 64)
    print(f"\nSize of x_upyd is {x_upyd.size()}")
    upydnet = uPydNet(1, 1)
    y_upyd  = upydnet(x_upyd)
    print(f"Size of y_upyd is {y_upyd.size()}\n")

    # Test downsampling model
    x_dwn  = torch.ones(1, 1, 320, 320)
    print(f"\nSize of x_dwn is {x_dwn.size()}")
    upydnet_8x8 = uPydNet_8x8(1, 1)
    y_dwn  = upydnet_8x8(x_dwn)
    print(f"Size of y_dwn is {y_dwn.size()}\n")
