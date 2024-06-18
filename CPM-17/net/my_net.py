# -*- coding: UTF-8 -*-

import torch.nn as nn
import torch

from net.unet import UNet
from net.unet_light import UNet_Light3
from net.unet_dcn import UNet_Light3_dcn

class MyNet(nn.Module):
    def __init__(self, net_select, is_ellipse, output_channels=2):
        super().__init__()
        self.is_ellipse = is_ellipse
        if is_ellipse:
            input_channels = 4 # need assistance of ellipses
        else:
            input_channels = 3 # original RGB input images

        if net_select in ['unet-transpose', 'unet-bilinear']:
            bilinear = {'unet-transpose':False, 'unet-bilinear':True}[net_select]
            self.mynet = UNet(input_channels, output_channels, bilinear)
        elif net_select in ['unet-light3']:
            self.mynet = {"unet-light3":UNet_Light3(input_channels, output_channels),}[net_select]
        elif net_select in ["unet-dcn"]:
            self.mynet = {"unet-dcn":UNet_Light3_dcn(input_channels,output_channels),}[net_select]
        else:
            raise Exception ("Error: please choose another -net_select-.")
        print('--> net_select:{}'.format(net_select))
    def forward(self, x, ellipse):
        if self.is_ellipse:
            ellipse = torch.unsqueeze(ellipse,1) # N*1*x*x
            x = torch.cat((x,ellipse),1) # N*4*x*x
            x = self.mynet(x)
        else:
            x = self.mynet(x)
        return x

