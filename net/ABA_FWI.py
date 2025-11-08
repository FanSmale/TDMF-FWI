# -*- coding: utf-8 -*-
"""
Created on 2024/7/16 20:11

@author: XUQIONG  (xuqiong@swpu.edu.cn)

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.show import *
from wtconv import *

NORM_LAYERS = {'bn': nn.BatchNorm2d, 'in': nn.InstanceNorm2d, 'ln': nn.LayerNorm}


class ConvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn', relu_slop=0.2, dropout=None):
        super(ConvBlock, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride,
                            padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.PReLU())
        if dropout:
            layers.append(nn.Dropout2d(0.8))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ConvBlock_Tanh(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn'):
        super(ConvBlock_Tanh, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride,
                            padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=2, stride=2, padding=0, output_padding=0, norm='bn'):
        super(DeconvBlock, self).__init__()
        layers = [nn.ConvTranspose2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride,
                                     padding=padding, output_padding=output_padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.PReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


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


class SpatialAttention1(nn.Module):
    def __init__(self):
        super(SpatialAttention1, self).__init__()
        self.conv = nn.Conv2d(2, 1, 3, padding=1, bias=False)

    def forward(self, x):

        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_pool, max_pool], dim=1)
        y = self.conv(y)
        y = torch.sigmoid(y)
        # import matplotlib.pyplot as plt
        # plt.imshow(x.detach().cpu().numpy()[102][15][:][:])
        # plt.axis('off')  # 关闭坐标轴
        # plt.show()
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


class ABA_FWI(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, **kwargs):
        super(ABA_FWI, self).__init__()
        self.convblock1 = ConvBlock(5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = ConvBlock(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = ConvBlock(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = ConvBlock(dim3, dim3, stride=2, padding=(9, 6))
        self.convblock5_2 = ConvBlock(dim3, dim3)
        self.convblock6_1 = ConvBlock(dim3, dim4, stride=2)
        self.convblock6_2 = ConvBlock(dim4, dim4)
        self.convblock7_1 = ConvBlock(dim4, dim4, stride=2)
        self.convblock7_2 = ConvBlock(dim4, dim4)
        self.convblock8_1 = ConvBlock(dim4, dim5, stride=2)
        self.convblock8_2 = ConvBlock(dim5, dim5)

        self.deconv2_1 = DeconvBlock(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock(dim4, dim4)
        self.deconv3_1 = DeconvBlock(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = ConvBlock(dim3, dim3)
        self.deconv4_1 = DeconvBlock(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = ConvBlock(dim2, dim2)
        self.deconv5_1 = DeconvBlock(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = ConvBlock(dim1, dim1)
        self.deconv6 = ConvBlock_Tanh(dim1, 1)

        self.CBAM_model1 = CBAMModule(256)
        self.CBAM_model2 = CBAMModule(128)
        self.CBAM_model3 = CBAMModule(64)
        self.CBAM_model4 = CBAMModule(32)

        self.change_channel0 = nn.Conv2d(256, 128, kernel_size=1)
        self.change_channel1 = nn.Conv2d(512, 256, kernel_size=1)
        self.change_channel2 = nn.Conv2d(256, 128, kernel_size=1)

        self.wt0 = WTConv2d(128,128)
        self.wt1 = WTConv2d(128,128)
        self.wt2 = WTConv2d(256,256)

    def forward(self, x):
        # Encoder Part
        x0 = self.convblock1(x)  # (None, 32, 500, 70)
        x0 = self.convblock2_1(x0)  # (None, 64, 250, 70)
        x0 = self.convblock2_2(x0)  # (None, 64, 250, 70)
        x0 = self.convblock3_1(x0)  # (None, 64, 125, 70)
        x0 = self.convblock3_2(x0)  # (None, 64, 125, 70)

        x1 = self.convblock4_1(x0)  # (None, 128, 63, 70)
        x2 = self.convblock4_2(x1)  # (None, 128, 63, 70)
        x2_wt = self.wt0(x2)

        x3 = self.convblock5_1(x2_wt)  # (None, 128, 40, 40)
        x4 = self.convblock5_2(x3)  # (None, 128, 40, 40)
        x4_wt = self.wt1(x4)

        x5 = self.convblock6_1(x4_wt)  # (None, 256, 20, 20)
        x6 = self.convblock6_2(x5)  # (None, 256, 20, 20)
        x6_wt = self.wt2(x6)

        x7 = self.convblock7_1(x6_wt)  # (None, 256, 10, 10)
        x8 = self.convblock7_2(x7)  # (None, 256, 10, 10)

        x9 = self.convblock8_1(x8)  # (None, 512, 5, 5)
        x10 = self.convblock8_2(x9)  # (None, 512, 5, 5)

        # Decoder Part Vmodel
        y0 = self.deconv2_1(x10)  # (None, 256, 10, 10)
        y0_concat = torch.cat((x8, y0), dim=1)
        y0_concat = self.change_channel1(y0_concat)
        y1 = self.deconv2_2(y0_concat)  # (None, 256, 10, 10)
        y1_ca = self.CBAM_model1(y1)

        y2 = self.deconv3_1(y1_ca)  # (None, 128, 20, 20)
        x6_wt = self.change_channel0(x6_wt)
        y2_concat = torch.cat((x6_wt, y2), dim=1)
        y2_concat = self.change_channel2(y2_concat)
        y3 = self.deconv3_2(y2_concat)  # (None, 128, 20, 20)
        y3_ca = self.CBAM_model2(y3)

        y4 = self.deconv4_1(y3_ca)  # (None, 64, 40, 40)
        y5 = self.deconv4_2(y4)  # (None, 64, 40, 40)
        y5_ca = self.CBAM_model3(y5)

        y6 = self.deconv5_1(y5_ca)  # (None, 32, 80, 80)
        y7 = self.deconv5_2(y6)  # (None, 32, 80, 80)
        y7_ca = self.CBAM_model4(y7)

        # pain_openfwi_velocity_model(y7_ca[0,0,:,:].cpu().detach().numpy())
        y8 = F.pad(y7_ca, [-5, -5, -5, -5], mode="constant", value=0)  # (None, 32, 70, 70) 125, 100
        # pain_openfwi_velocity_model(y8[0,0,:,:].cpu().detach().numpy())
        y9 = self.deconv6(y8)  # (None, 1, 70, 70)

        return y9


class ABA_FWI_100(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, **kwargs):
        super(ABA_FWI_100, self).__init__()
        self.convblock1 = ConvBlock(7, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = ConvBlock(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = ConvBlock(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = ConvBlock(dim3, dim3, stride=2, padding=(10, 7))
        self.convblock5_2 = ConvBlock(dim3, dim3)
        self.convblock6_1 = ConvBlock(dim3, dim4, stride=2)
        self.convblock6_2 = ConvBlock(dim4, dim4)
        self.convblock7_1 = ConvBlock(dim4, dim4, stride=2)
        self.convblock7_2 = ConvBlock(dim4, dim4)
        self.convblock8_1 = ConvBlock(dim4, dim5, stride=2)
        self.convblock8_2 = ConvBlock(dim5, dim5)

        self.deconv2_1 = DeconvBlock(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock(dim4, dim4)
        self.deconv3_1 = DeconvBlock(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = ConvBlock(dim3, dim3)
        self.deconv4_1 = DeconvBlock(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = ConvBlock(dim2, dim2)
        self.deconv5_1 = DeconvBlock(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = ConvBlock(dim1, dim1)
        self.deconv6 = ConvBlock_Tanh(dim1, 1)

        self.CBAM_model1 = CBAMModule(256)
        self.CBAM_model2 = CBAMModule(128)
        self.CBAM_model3 = CBAMModule(64)
        self.CBAM_model4 = CBAMModule(32)

        self.change_channel0 = nn.Conv2d(256, 128, kernel_size=1)
        self.change_channel1 = nn.Conv2d(512, 256, kernel_size=1)
        self.change_channel2 = nn.Conv2d(256, 128, kernel_size=1)

        self.wt0 = WTConv2d(128,128)
        self.wt1 = WTConv2d(128,128)
        self.wt2 = WTConv2d(256,256)

    def forward(self, x):
        # Encoder Part
        x0 = self.convblock1(x)  # (None, 32, 750, 100)
        x0 = self.convblock2_1(x0)  # (None, 64, 375, 100)
        x0 = self.convblock2_2(x0)  # (None, 64, 375, 100)
        x0 = self.convblock3_1(x0)  # (None, 64, 188, 100)
        x0 = self.convblock3_2(x0)  # (None, 64, 188, 100)

        x1 = self.convblock4_1(x0)  # (None, 128, 94, 100)
        x2 = self.convblock4_2(x1)  # (None, 128, 94, 100)
        x2_wt = self.wt0(x2)

        x3 = self.convblock5_1(x2_wt)  # (None, 128, 56, 56)
        x4 = self.convblock5_2(x3)  # (None, 128, 56, 56)
        x4_wt = self.wt1(x4)

        x5 = self.convblock6_1(x4_wt)  # (None, 256, 28, 28)
        x6 = self.convblock6_2(x5)  # (None, 256, 28, 28)
        x6_wt = self.wt2(x6)

        x7 = self.convblock7_1(x6_wt)  # (None, 256, 14, 14)
        x8 = self.convblock7_2(x7)  # (None, 256, 14, 14)

        x9 = self.convblock8_1(x8)  # (None, 512, 7, 7)
        x10 = self.convblock8_2(x9)  # (None, 512, 7, 7)

        # Decoder Part Vmodel
        y0 = self.deconv2_1(x10)  # (None, 256, 14, 14)
        y0_concat = torch.cat((x8, y0), dim=1)
        y0_concat = self.change_channel1(y0_concat)
        y1 = self.deconv2_2(y0_concat)  # (None, 256, 14, 14)
        y1_ca = self.CBAM_model1(y1)

        y2 = self.deconv3_1(y1_ca)  # (None, 128, 28, 28)
        x6_wt = self.change_channel0(x6_wt)
        y2_concat = torch.cat((x6_wt, y2), dim=1)
        y2_concat = self.change_channel2(y2_concat)
        y3 = self.deconv3_2(y2_concat)  # (None, 128, 28, 28)
        y3_ca = self.CBAM_model2(y3)

        y4 = self.deconv4_1(y3_ca)  # (None, 64, 56, 56)
        y5 = self.deconv4_2(y4)  # (None, 64, 56, 56)
        y5_ca = self.CBAM_model3(y5)

        y6 = self.deconv5_1(y5_ca)  # (None, 32, 112, 112)
        y7 = self.deconv5_2(y6)  # (None, 32, 112, 112)
        y7_ca = self.CBAM_model4(y7)

        # pain_openfwi_velocity_model(y7_ca[0,0,:,:].cpu().detach().numpy())
        y8 = F.pad(y7_ca, [-6, -6, -6, -6], mode="constant", value=0)  # (None, 32, 100, 100) 125, 100
        # pain_openfwi_velocity_model(y8[0,0,:,:].cpu().detach().numpy())
        y9 = self.deconv6(y8)  # (None, 1, 100, 100)

        return y9


class ABA_Loss(nn.Module):
    """
    The ablation experiment.
    Add skip connections into InversionNet.
    """
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, **kwargs):
        super(ABA_Loss, self).__init__()
        self.convblock1 = ConvBlock(5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = ConvBlock(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = ConvBlock(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = ConvBlock(dim3, dim3, stride=2, padding=(9, 6))
        self.convblock5_2 = ConvBlock(dim3, dim3)
        self.convblock6_1 = ConvBlock(dim3, dim4, stride=2)
        self.convblock6_2 = ConvBlock(dim4, dim4)
        self.convblock7_1 = ConvBlock(dim4, dim4, stride=2)
        self.convblock7_2 = ConvBlock(dim4, dim4)
        self.convblock8_1 = ConvBlock(dim4, dim5, stride=2)
        self.convblock8_2 = ConvBlock(dim5, dim5)

        self.deconv2_1 = DeconvBlock(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock(dim4, dim4)
        self.deconv3_1 = DeconvBlock(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = ConvBlock(dim3, dim3)
        self.deconv4_1 = DeconvBlock(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = ConvBlock(dim2, dim2)
        self.deconv5_1 = DeconvBlock(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = ConvBlock(dim1, dim1)
        self.deconv6 = ConvBlock_Tanh(dim1, 1)

        self.change_channel0 = nn.Conv2d(256, 128, kernel_size=1)
        self.change_channel1 = nn.Conv2d(512, 256, kernel_size=1)
        self.change_channel2 = nn.Conv2d(256, 128, kernel_size=1)

    def forward(self, x):
        # Encoder Part
        x = self.convblock1(x)  # (None, 32, 500, 70)
        x = self.convblock2_1(x)  # (None, 64, 250, 70)
        x = self.convblock2_2(x)  # (None, 64, 250, 70)
        x = self.convblock3_1(x)  # (None, 64, 125, 70)
        x = self.convblock3_2(x)  # (None, 64, 125, 70)
        x = self.convblock4_1(x)  # (None, 128, 63, 70)
        x = self.convblock4_2(x)  # (None, 128, 63, 70)
        x = self.convblock5_1(x)  # (None, 128, 40, 40)
        x = self.convblock5_2(x)  # (None, 128, 40, 40)

        x = self.convblock6_1(x)  # (None, 256, 20, 20)
        x = self.convblock6_2(x)  # (None, 256, 20, 20)

        x1 = x  # (None, 64, 20, 20)

        x = self.convblock7_1(x)  # (None, 256, 10, 10)
        x = self.convblock7_2(x)  # (None, 256, 10, 10)

        x2 = x  # (None, 64, 20, 20)

        x = self.convblock8_1(x)  # (None, 512, 5, 5)
        x = self.convblock8_2(x)  # (None, 512, 5, 5)

        # Decoder Part Vmodel
        y = self.deconv2_1(x)  # (None, 256, 10, 10)
        y_concat = torch.cat((x2, y), dim=1)
        y_concat = self.change_channel1(y_concat)
        y1 = self.deconv2_2(y_concat)  # (None, 256, 10, 10)

        y2 = self.deconv3_1(y1)  # (None, 128, 20, 20)
        x1 = self.change_channel0(x1)
        y2_concat = torch.cat((x1, y2), dim=1)
        y2_concat = self.change_channel2(y2_concat)
        y3 = self.deconv3_2(y2_concat)  # (None, 128, 20, 20)

        y4 = self.deconv4_1(y3)  # (None, 64, 40, 40)
        y5 = self.deconv4_2(y4)  # (None, 64, 40, 40)

        y6 = self.deconv5_1(y5)  # (None, 32, 80, 80)
        y7 = self.deconv5_2(y6)  # (None, 32, 80, 80)

        y8 = F.pad(y7, [-5, -5, -5, -5], mode="constant", value=0)  # (None, 32, 70, 70) 125, 100
        # pain_openfwi_velocity_model(y8[0,0,:,:].cpu().detach().numpy())
        y9 = self.deconv6(y8)  # (None, 1, 70, 70)

        return y9


class ABA_Loss_conv(nn.Module):
    """
    The ablation experiment.   将 WTconv 和 SAM 退化为一般的卷积模块
    Add skip connections into InversionNet.
    """
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, **kwargs):
        super(ABA_Loss_conv, self).__init__()
        self.convblock1 = ConvBlock(5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = ConvBlock(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = ConvBlock(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = ConvBlock(dim3, dim3, stride=2, padding=(9, 6))
        self.convblock5_2 = ConvBlock(dim3, dim3)
        self.convblock6_1 = ConvBlock(dim3, dim4, stride=2)
        self.convblock6_2 = ConvBlock(dim4, dim4)
        self.convblock7_1 = ConvBlock(dim4, dim4, stride=2)
        self.convblock7_2 = ConvBlock(dim4, dim4)
        self.convblock8_1 = ConvBlock(dim4, dim5, stride=2)
        self.convblock8_2 = ConvBlock(dim5, dim5)

        self.deconv2_1 = DeconvBlock(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock(dim4, dim4)
        self.deconv3_1 = DeconvBlock(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = ConvBlock(dim3, dim3)
        self.deconv4_1 = DeconvBlock(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = ConvBlock(dim2, dim2)
        self.deconv5_1 = DeconvBlock(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = ConvBlock(dim1, dim1)
        self.deconv6 = ConvBlock_Tanh(dim1, 1)

        self.CBAM_model1 = ConvBlock(256, 256, kernel_size=3, stride=1, padding=1)
        self.CBAM_model2 = ConvBlock(128, 128, kernel_size=3, stride=1, padding=1)
        self.CBAM_model3 = ConvBlock(64, 64, kernel_size=3, stride=1, padding=1)
        self.CBAM_model4 = ConvBlock(32, 32, kernel_size=3, stride=1, padding=1)

        self.change_channel0 = nn.Conv2d(256, 128, kernel_size=1)
        self.change_channel1 = nn.Conv2d(512, 256, kernel_size=1)
        self.change_channel2 = nn.Conv2d(256, 128, kernel_size=1)

        self.wt0 = ConvBlock(128, 128, kernel_size=3, stride=1, padding=1)
        self.wt1 = ConvBlock(128, 128, kernel_size=3, stride=1, padding=1)
        self.wt2 = ConvBlock(256, 256, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Encoder Part
        x0 = self.convblock1(x)  # (None, 32, 500, 70)
        x0 = self.convblock2_1(x0)  # (None, 64, 250, 70)
        x0 = self.convblock2_2(x0)  # (None, 64, 250, 70)
        x0 = self.convblock3_1(x0)  # (None, 64, 125, 70)
        x0 = self.convblock3_2(x0)  # (None, 64, 125, 70)

        x1 = self.convblock4_1(x0)  # (None, 128, 63, 70)
        x2 = self.convblock4_2(x1)  # (None, 128, 63, 70)
        x2_wt = self.wt0(x2)

        x3 = self.convblock5_1(x2_wt)  # (None, 128, 40, 40)
        x4 = self.convblock5_2(x3)  # (None, 128, 40, 40)
        x4_wt = self.wt1(x4)

        x5 = self.convblock6_1(x4_wt)  # (None, 256, 20, 20)
        x6 = self.convblock6_2(x5)  # (None, 256, 20, 20)
        x6_wt = self.wt2(x6)

        x7 = self.convblock7_1(x6_wt)  # (None, 256, 10, 10)
        x8 = self.convblock7_2(x7)  # (None, 256, 10, 10)

        x9 = self.convblock8_1(x8)  # (None, 512, 5, 5)
        x10 = self.convblock8_2(x9)  # (None, 512, 5, 5)

        # Decoder Part Vmodel
        y0 = self.deconv2_1(x10)  # (None, 256, 10, 10)
        y0_concat = torch.cat((x8, y0), dim=1)
        y0_concat = self.change_channel1(y0_concat)
        y1 = self.deconv2_2(y0_concat)  # (None, 256, 10, 10)
        y1_ca = self.CBAM_model1(y1)

        y2 = self.deconv3_1(y1_ca)  # (None, 128, 20, 20)
        x6_wt = self.change_channel0(x6_wt)
        y2_concat = torch.cat((x6_wt, y2), dim=1)
        y2_concat = self.change_channel2(y2_concat)
        y3 = self.deconv3_2(y2_concat)  # (None, 128, 20, 20)
        y3_ca = self.CBAM_model2(y3)

        y4 = self.deconv4_1(y3_ca)  # (None, 64, 40, 40)
        y5 = self.deconv4_2(y4)  # (None, 64, 40, 40)
        y5_ca = self.CBAM_model3(y5)

        y6 = self.deconv5_1(y5_ca)  # (None, 32, 80, 80)
        y7 = self.deconv5_2(y6)  # (None, 32, 80, 80)
        y7_ca = self.CBAM_model4(y7)

        # pain_openfwi_velocity_model(y7_ca[0,0,:,:].cpu().detach().numpy())
        y8 = F.pad(y7_ca, [-5, -5, -5, -5], mode="constant", value=0)  # (None, 32, 70, 70) 125, 100
        # pain_openfwi_velocity_model(y8[0,0,:,:].cpu().detach().numpy())
        y9 = self.deconv6(y8)  # (None, 1, 70, 70)

        return y5_ca, y7_ca, y9

        # return y9


class ABA_Loss_conv_only(nn.Module):
    """
    The ablation experiment.   将 WTconv 和 SAM 退化为一般的卷积模块
    Add skip connections into InversionNet.
    """
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, **kwargs):
        super(ABA_Loss_conv_only, self).__init__()
        self.convblock1 = ConvBlock(5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = ConvBlock(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = ConvBlock(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = ConvBlock(dim3, dim3, stride=2, padding=(9, 6))
        self.convblock5_2 = ConvBlock(dim3, dim3)
        self.convblock6_1 = ConvBlock(dim3, dim4, stride=2)
        self.convblock6_2 = ConvBlock(dim4, dim4)
        self.convblock7_1 = ConvBlock(dim4, dim4, stride=2)
        self.convblock7_2 = ConvBlock(dim4, dim4)
        self.convblock8_1 = ConvBlock(dim4, dim5, stride=2)
        self.convblock8_2 = ConvBlock(dim5, dim5)

        self.deconv2_1 = DeconvBlock(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock(dim4, dim4)
        self.deconv3_1 = DeconvBlock(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = ConvBlock(dim3, dim3)
        self.deconv4_1 = DeconvBlock(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = ConvBlock(dim2, dim2)
        self.deconv5_1 = DeconvBlock(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = ConvBlock(dim1, dim1)
        self.deconv6 = ConvBlock_Tanh(dim1, 1)

        self.CBAM_model1 = nn.Conv2d(256, 256, kernel_size=1)
        self.CBAM_model2 = nn.Conv2d(128, 128, kernel_size=1)
        self.CBAM_model3 = nn.Conv2d(64, 64, kernel_size=1)
        self.CBAM_model4 = nn.Conv2d(32, 32, kernel_size=1)

        self.change_channel0 = nn.Conv2d(256, 128, kernel_size=1)
        self.change_channel1 = nn.Conv2d(512, 256, kernel_size=1)
        self.change_channel2 = nn.Conv2d(256, 128, kernel_size=1)

        self.wt0 = nn.Conv2d(128, 128, kernel_size=1)
        self.wt1 = nn.Conv2d(128, 128, kernel_size=1)
        self.wt2 = nn.Conv2d(256, 256, kernel_size=1)

    def forward(self, x):
        # Encoder Part
        x0 = self.convblock1(x)  # (None, 32, 500, 70)
        x0 = self.convblock2_1(x0)  # (None, 64, 250, 70)
        x0 = self.convblock2_2(x0)  # (None, 64, 250, 70)
        x0 = self.convblock3_1(x0)  # (None, 64, 125, 70)
        x0 = self.convblock3_2(x0)  # (None, 64, 125, 70)

        x1 = self.convblock4_1(x0)  # (None, 128, 63, 70)
        x2 = self.convblock4_2(x1)  # (None, 128, 63, 70)
        x2_wt = self.wt0(x2)

        x3 = self.convblock5_1(x2_wt)  # (None, 128, 40, 40)
        x4 = self.convblock5_2(x3)  # (None, 128, 40, 40)
        x4_wt = self.wt1(x4)

        x5 = self.convblock6_1(x4_wt)  # (None, 256, 20, 20)
        x6 = self.convblock6_2(x5)  # (None, 256, 20, 20)
        x6_wt = self.wt2(x6)

        x7 = self.convblock7_1(x6_wt)  # (None, 256, 10, 10)
        x8 = self.convblock7_2(x7)  # (None, 256, 10, 10)

        x9 = self.convblock8_1(x8)  # (None, 512, 5, 5)
        x10 = self.convblock8_2(x9)  # (None, 512, 5, 5)

        # Decoder Part Vmodel
        y0 = self.deconv2_1(x10)  # (None, 256, 10, 10)
        y0_concat = torch.cat((x8, y0), dim=1)
        y0_concat = self.change_channel1(y0_concat)
        y1 = self.deconv2_2(y0_concat)  # (None, 256, 10, 10)
        y1_ca = self.CBAM_model1(y1)

        y2 = self.deconv3_1(y1_ca)  # (None, 128, 20, 20)
        x6_wt = self.change_channel0(x6_wt)
        y2_concat = torch.cat((x6_wt, y2), dim=1)
        y2_concat = self.change_channel2(y2_concat)
        y3 = self.deconv3_2(y2_concat)  # (None, 128, 20, 20)
        y3_ca = self.CBAM_model2(y3)

        y4 = self.deconv4_1(y3_ca)  # (None, 64, 40, 40)
        y5 = self.deconv4_2(y4)  # (None, 64, 40, 40)
        y5_ca = self.CBAM_model3(y5)

        y6 = self.deconv5_1(y5_ca)  # (None, 32, 80, 80)
        y7 = self.deconv5_2(y6)  # (None, 32, 80, 80)
        y7_ca = self.CBAM_model4(y7)

        # pain_openfwi_velocity_model(y7_ca[0,0,:,:].cpu().detach().numpy())
        y8 = F.pad(y7_ca, [-5, -5, -5, -5], mode="constant", value=0)  # (None, 32, 70, 70) 125, 100
        # pain_openfwi_velocity_model(y8[0,0,:,:].cpu().detach().numpy())
        y9 = self.deconv6(y8)  # (None, 1, 70, 70)

        return y9


if __name__ == '__main__':
    input = torch.rand((5, 7, 1500, 100))

    model = ABA_FWI_100()
    output = model(input)
    print('ok')
