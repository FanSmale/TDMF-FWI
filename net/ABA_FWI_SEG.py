# -*- coding: utf-8 -*-
"""
Created on 2024/07/16

@author: XUQIONG (xuqiong@swpu.edu.cn)

"""

################################################
########        DESIGN   NETWORK        ########
################################################

import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from wtconv import *

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )

    def forward(self, x):
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        y = self.fc(avg_pool) + self.fc(max_pool)
        y = torch.sigmoid(y)
        return x * y


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, 3, padding=1, bias=False)

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_pool, max_pool], dim=1)
        y = self.conv(y)
        y = torch.sigmoid(y)
        return x * y


class CBAMModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAMModule, self).__init__()
        # self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        #  x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetConv2, self).__init__()
        # Kernel size: 3*3, Stride: 1, Padding: 1
        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True), )
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True), )
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       nn.ReLU(inplace=True), )
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       nn.ReLU(inplace=True), )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class unetDown(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetDown, self).__init__()
        self.conv = unetConv2(in_size, out_size, is_batchnorm)
        self.down = nn.MaxPool2d(2, 2, ceil_mode=True)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        outputs = self.down(outputs)
        return outputs


class unetDownWt(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetDownWt, self).__init__()
        self.conv = unetConv2(in_size, out_size, is_batchnorm)
        self.wt = WTConv2d(out_size, out_size)
        self.down = nn.MaxPool2d(2, 2, ceil_mode=True)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        outputs = self.wt(outputs)
        outputs = self.down(outputs)
        return outputs


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size, out_size, True)
        # Transposed convolution
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset1 = (outputs2.size()[2] - inputs1.size()[2])
        offset2 = (outputs2.size()[3] - inputs1.size()[3])
        padding = [offset2 // 2, (offset2 + 1) // 2, offset1 // 2, (offset1 + 1) // 2]
        # Skip and concatenate
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


class ABA_FWI_SEG(nn.Module):
    def __init__(self, n_classes, in_channels, is_deconv, is_batchnorm):
        super(ABA_FWI_SEG, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.n_classes = n_classes

        filters = [64, 128, 256, 512, 1024]

        self.down1 = unetDownWt(self.in_channels, filters[0], self.is_batchnorm)
        self.down2 = unetDownWt(filters[0], filters[1], self.is_batchnorm)
        self.down3 = unetDownWt(filters[1], filters[2], self.is_batchnorm)
        self.down4 = unetDownWt(filters[2], filters[3], self.is_batchnorm)
        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)
        self.up4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up1 = unetUp(filters[1], filters[0], self.is_deconv)
        self.final = nn.Conv2d(filters[0], self.n_classes, 1)
        self.sa1 = SpatialAttention()
        self.sa2 = SpatialAttention()
        self.sa3 = SpatialAttention()
        self.sa4 = SpatialAttention()

    def forward(self, inputs, label_dsp_dim):
        down1 = self.down1(inputs)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        center = self.center(down4)
        up4 = self.up4(down4, center)
        up4 = self.sa4(up4)
        up3 = self.up3(down3, up4)
        up3 = self.sa3(up3)
        up2 = self.up2(down2, up3)
        up2 = self.sa2(up2)
        up1 = self.up1(down1, up2)
        up1 = self.sa1(up1)
        up1 = up1[:, :, 1:1 + label_dsp_dim[0], 1:1 + label_dsp_dim[1]].contiguous()

        return self.final(up1)

    # Initialization of Parameters
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()


if __name__ == '__main__':
    x = torch.randn(10, 29, 400, 301)  # 创建一个形状为(1, 10)的随机Tensor

    model = ABAWT_FWI_SEG(1,29,True,True)
    #
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"Parameter: {name}, Size: {param.size()}")

    #在网络中进行正向传播
    output = model(x,[200,301])
    print(output)  # (none, 42, 42, 1)