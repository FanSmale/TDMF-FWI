# -*- coding: utf-8 -*-
"""
Created on 2025/9/16 22:11

@author: XUQIONG  (xuqiong@swpu.edu.cn)

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from func.show import *
from wtconv import *
import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_, DropPath

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


class TemporalConv(nn.Module):
    def __init__(self, in_channels, width_in, width_out):
        super(TemporalConv, self).__init__()
        self.depthwise_conv1 = ConvBlock(width_in, width_in, kernel_size=(1, 7), stride=(1, 2), padding=(0, 3))
        self.depthwise_conv2 = ConvBlock(width_in, width_in)
        self.pointwise_conv1 = ConvBlock(width_in, width_out, kernel_size=1, padding=0)
        self.pointwise_conv2 = ConvBlock(in_channels, in_channels * 2, kernel_size=1, padding=0)


    def forward(self, x):
        x0 = x.permute(0, 3, 1, 2)           # (none, 151, 58, 200)       (none, 35, 10, 500)
        x1 = self.depthwise_conv1(x0)        # (none, 151, 58, 100)        (none, 35, 10, 250)
        x2 = self.depthwise_conv2(x1)       # (none, 151, 58, 100)         (none, 35, 10, 500)

        x3 = self.pointwise_conv1(x2)        # (none, 76, 58, 100)        (none, 18, 10, 250)
        x4 = x3.permute(0, 2, 3, 1)          # (none, 58, 100, 76)        (none, 10, 250, 18)
        y = self.pointwise_conv2(x4)         # (none, 10, 500, 35)       (none, 20, 250, 18)
        return y

# X = torch.zeros((1, 5, 1000, 70))
# model = TemporalConv(5, 70, 5)
# out = model(X)

class SpatioConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpatioConv, self).__init__()
        self.pre_dim_reducer1 = ConvBlock(in_channels, out_channels, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.pre_dim_reducer2 = ConvBlock(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):
        dimred0 = x
        dimred1 = self.pre_dim_reducer1(dimred0)     # (none, 8, 280, 70)
        dimred2 = self.pre_dim_reducer2(dimred1)     # (none, 8, 280, 70)

        return dimred2

# SpatioConv(in_channels=5)
# SpatioConv(in_channels=10)
# SpatioConv(in_channels=20)
# SpatioConv(in_channels=40)
# SpatioConv(in_channels=80)

class unetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        '''
        Downsampling Unit
        (The purple arrow shown in Fig 1 of the paper)

        :param in_size:      Number of channels of input
        :param out_size:     Number of channels of output
        :param is_batchnorm: Whether to use BN
        :param activ_fuc:    Activation function
        '''
        super(unetDown, self).__init__()
        self.wtConv = WTConv2d(in_channels, in_channels)
        self.conv_1 = ConvBlock(in_channels, out_channels)
        self.conv_2 = ConvBlock(out_channels, out_channels, stride=2)
        self.conv = ConvBlock(out_channels * 2, out_channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, inputs1, inputs2):
        skip_output = self.wtConv(inputs2)
        outputs2_0 = self.conv_1(skip_output)
        outputs2_1 = self.conv_2(outputs2_0)
        outputs = torch.cat([inputs1, outputs2_1], 1)
        outputs = self.conv(outputs)
        outputs = self.spatial_attention(outputs)
        return outputs


class unetUp(nn.Module):
    def __init__(self, in_channels, out_channels, output_lim):
        '''
        Upsampling Unit
        (The yellow arrow shown in Fig 1 of the paper)

        :param in_size:      Number of channels of input
        :param out_size:     Number of channels of output
        :param is_deconv:    Whether to use deconvolution
        :param activ_fuc:    Activation function
        '''
        super(unetUp, self).__init__()
        self.output_lim = output_lim
        self.deconv = DeconvBlock(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.conv = ConvBlock(out_channels*2, out_channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, input1, input2):
        input2 = self.deconv(input2)
        input2 = F.interpolate(input2, size=self.output_lim, mode='bilinear', align_corners=False)
        input = torch.cat([input1, input2], 1)
        input = self.conv(input)
        output = self.spatial_attention(input)

        return output
# unetUp(512, 256, output_lim=[9, 9])
# unetUp(256, 128, output_lim=[18, 18])
# unetUp(128, 64, output_lim=[35, 35])
# unetUp(64, 32, output_lim=[70, 70])
class unetResUp(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, output_lim):
        '''
        Upsampling Unit
        (The yellow arrow shown in Fig 1 of the paper)

        :param in_size:      Number of channels of input
        :param out_size:     Number of channels of output
        :param is_deconv:    Whether to use deconvolution
        :param activ_fuc:    Activation function
        '''
        super(unetResUp, self).__init__()
        self.output_lim = output_lim
        self.deconv = DeconvBlock(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.conv = ConvBlock(out_channels*2+middle_channels, out_channels)

    def forward(self, input1, input2, input3):
        # input1 上采样 input2 时间残差 input3 速度模型
        input2 = self.deconv(input2)
        input2 = F.interpolate(input2, size=self.output_lim, mode='bilinear', align_corners=False)
        input1 = F.interpolate(input1, size=self.output_lim, mode='bilinear', align_corners=False)
        input = torch.cat([input1, input2, input3], 1)
        output = self.conv(input)
        return output

class unetTransUp(nn.Module):
    def __init__(self, in_channels, out_channels, output_lim):
        '''
        Upsampling Unit
        (The yellow arrow shown in Fig 1 of the paper)

        :param in_size:      Number of channels of input
        :param out_size:     Number of channels of output
        :param is_deconv:    Whether to use deconvolution
        :param activ_fuc:    Activation function
        '''
        super(unetTransUp, self).__init__()
        self.output_lim = output_lim
        self.deconv = DeconvBlock(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.conv = ConvBlock(out_channels*2, out_channels)

    def forward(self, input1, input2):
        # input1 上采样 input2 时间残差 input3 速度模型
        input2 = self.deconv(input2)
        input2 = F.interpolate(input2, size=self.output_lim, mode='bilinear', align_corners=False)
        input = torch.cat([input1, input2], 1)
        output = self.conv(input)
        return output

class WMSA(nn.Module):
    def __init__(self, input_dim=5, output_dim=5, head_dim=5, window_size=7, ratio=15, type='W'):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim // head_dim
        self.window_size = window_size
        self.ratio = ratio
        self.type = type
        self.embedding_layer = nn.Linear(self.input_dim, 3 * self.input_dim, bias=True)

        self.relative_position_params = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), self.n_heads)
        )
        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(
            self.relative_position_params.view(2 * window_size - 1, 2 * window_size - 1, self.n_heads)
            .transpose(1, 2).transpose(0, 1)
        )

    def generate_mask(self, h, w, p, shift):
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

    def forward(self, x):  # x.shape: (400, 305, 29)
        if self.type != 'W':
            x = torch.roll(x, shifts=(-(self.window_size // 2), -(self.window_size // 2)), dims=(1, 2))
        x = rearrange(x, 'DataSplite (w1 p1) (w2 p2) c -> DataSplite w1 w2 p1 p2 c',
                      p1=self.window_size, p2=self.window_size)  # (1, 80, 61, 5, 5, 29)

        h_windows, w_windows = x.size(1), x.size(2)

        x = rearrange(x, 'DataSplite w1 w2 p1 p2 c -> DataSplite (w1 w2) (p1 p2) c',
                      p1=self.window_size, p2=self.window_size)  # (1, 4880, 25, 29)

        qkv = self.embedding_layer(x)  # (1, 4880, 25, 87)
        q, k, v = rearrange(qkv, 'DataSplite nw np (threeh c) -> threeh DataSplite nw np c',
                            c=self.head_dim).chunk(3, dim=0)   # (1, 4880, 25, 29)

        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale   # (1, 10, 4880, 25, 25)
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')

        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size // 2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)  # (1, 10, 4880, 25, 29)
        output = rearrange(output, 'h DataSplite w p c -> DataSplite w p (h c)')  # (1, 4880, 25, 29)
        output = self.linear(output)      # (1, 4880, 25, 29)
        output = rearrange(output, 'DataSplite (c w1 w2) (p1 p2) c1 -> DataSplite (w1 p1) (w2 p2) (c c1)',
                           c=self.ratio, w2=w_windows, p1=self.window_size)    # (1, 200, 305, 58)

        if self.type != 'W':
            output = torch.roll(output, shifts=(self.window_size // 2, self.window_size // 2), dims=(1, 2))
        return output

    def relative_embedding(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size - 1
        return self.relative_position_params[:, relation[:, :, 0].long(), relation[:, :, 1].long()]

class Block(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, ratio, type='W', input_resolution=None):
        """ SwinTransformer Block"""
        super(Block, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.ratio = ratio
        self.type = type
        if input_resolution <= window_size:
            self.type = 'W'

        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.ratio, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim * self.ratio)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * self.ratio, 4 * input_dim * self.ratio),
            nn.ReLU(),
            nn.Linear(4 * input_dim * self.ratio, output_dim * self.ratio),
        )
    def forward(self, x):    # 400, 305, 29
        # x1 = self.ln1(x)     # 1050, 70, 5
        # x2 = self.msa(x1)    # 1, 70, 70, 75
        # x3 = self.drop_path(x2)
        x = self.drop_path(self.msa(self.ln1(x)))
        x = self.drop_path(self.mlp(self.ln2(x)))
        return x

class TransBlock(nn.Module):
    def __init__(self, trans_dim=1, head_dim=1, window_size=7, drop_path=0.1, ratio=15, type='W', input_resolution=1):
        """  TransResBlock"""
        super(TransBlock, self).__init__()
        self.trans_dim = trans_dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.drop_path = drop_path
        self.ratio = ratio
        self.type = type
        self.input_resolution = input_resolution

        assert self.type in ['W', 'SW']
        if self.input_resolution <= self.window_size:
            self.type = 'W'

        self.trans_block = Block(self.trans_dim, self.trans_dim, self.head_dim, self.window_size, self.drop_path, self.ratio, self.type, self.input_resolution)
        self.conv1_1 = nn.Conv2d(self.trans_dim * self.ratio, self.trans_dim * self.ratio, 1, 1, 0, bias=True)

    def forward(self, x):
        _, c, h, w = x.shape
        pad_dim = ((w // 5) + 1) * 5
        x_padded = F.pad(x, (0, pad_dim - w, 0, 0))  # (left, right, top, bottom)
        trans_x = Rearrange('DataSplite c h w -> DataSplite h w c')(x_padded)   # 1, 400, 305, 29
        trans_x = self.trans_block(trans_x)  # 1, 70, 70, 75
        trans_x = Rearrange('DataSplite h w c -> DataSplite c h w')(trans_x)  # 1, 15, 70, 70
        output = self.conv1_1(trans_x)
        return output


class TransResBlock(nn.Module):
    def __init__(self, conv_dim, trans_dim=1, head_dim=1, window_size=7, drop_path=0.1, ratio=15, type='W', input_resolution=1):
        """  TransResBlock"""
        super(TransResBlock, self).__init__()
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.drop_path = drop_path
        self.ratio = ratio
        self.type = type
        self.input_resolution = input_resolution

        assert self.type in ['W', 'SW']
        if self.input_resolution <= self.window_size:
            self.type = 'W'

        self.trans_block = Block(self.trans_dim, self.trans_dim, self.head_dim, self.window_size, self.drop_path, self.ratio, self.type, self.input_resolution)
        self.conv1_1 = nn.Conv2d(self.trans_dim * self.ratio, self.trans_dim * self.ratio, 1, 1, 0, bias=True)
        self.conv1_2 = ConvBlock(self.trans_dim * self.ratio + self.conv_dim, self.conv_dim)

    def forward(self, x, y):
        _, c, h, w = x.shape
        if self.window_size == 3:
            pad_dim_w = (w // 3) * 3
            pad_dim_h = (h // 3) * 3
            x_padded = F.interpolate(x, size=(pad_dim_h, pad_dim_w), mode="bilinear")
        else:
            pad_dim = ((w // 5) + 1) * 5
            x_padded = F.pad(x, (0, pad_dim - w, 0, 0))
        trans_x = Rearrange('DataSplite c h w -> DataSplite h w c')(x_padded)   # 1, 38, 19, 464
        trans_x = self.trans_block(trans_x)  # 1, 70, 70, 75
        trans_x = Rearrange('DataSplite h w c -> DataSplite c h w')(trans_x)  # 1, 15, 70, 70
        _, _, h1, w1 = y.shape
        trans_x = F.interpolate(trans_x, size=(h1, w1), mode="bilinear")
        res = self.conv1_1(trans_x)
        output = torch.cat((y, res), dim=1)
        output = self.conv1_2(output)
        return output

class TDFS_FWI_SEG(nn.Module):
    def __init__(self):
        super(TDFS_FWI_SEG, self).__init__()
        self.temporalDown2 = TemporalConv(in_channels=29, width_in=301, width_out=151)
        self.temporalDown3 = TemporalConv(in_channels=58, width_in=151, width_out=76)
        self.temporalDown4 = TemporalConv(in_channels=116, width_in=76, width_out=38)
        self.temporalDown5 = TemporalConv(in_channels=232, width_in=38, width_out=19)

        self.temporal2spatio1 = SpatioConv(in_channels=29, out_channels=64)
        self.temporal2spatio2 = SpatioConv(in_channels=58, out_channels=128)
        self.temporal2spatio3 = SpatioConv(in_channels=116, out_channels=256)
        self.temporal2spatio4 = SpatioConv(in_channels=232, out_channels=512)
        self.temporal2spatio5 = SpatioConv(in_channels=464, out_channels=1024)

        self.spatioDown2 = unetDown(in_channels=64, out_channels=128)
        self.spatioDown3 = unetDown(in_channels=128, out_channels=256)
        self.spatioDown4 = unetDown(in_channels=256, out_channels=512)
        self.spatioDown5 = unetDown(in_channels=512, out_channels=1024)

        self.spatioUp4 = unetUp(in_channels=1024, out_channels=512, output_lim=[25, 38])
        self.spatioUp3 = unetUp(in_channels=512, out_channels=256, output_lim=[50, 76])
        self.spatioUp2 = unetUp(in_channels=256, out_channels=128, output_lim=[100, 151])
        self.spatioUp1 = unetUp(in_channels=128, out_channels=64, output_lim=[200, 301])

        self.trans1 = TransBlock(trans_dim=29, head_dim=29, window_size=5, drop_path=0.1, ratio=2, type='W', input_resolution=70)
        self.trans2 = TransBlock(trans_dim=58, head_dim=29, window_size=5, drop_path=0.1, ratio=2, type='W',
                                 input_resolution=35)
        self.trans3 = TransBlock(trans_dim=116, head_dim=29, window_size=5, drop_path=0.1, ratio=2, type='W',
                                 input_resolution=18)
        self.trans4 = TransBlock(trans_dim=232, head_dim=29, window_size=5, drop_path=0.1, ratio=2, type='W',
                                 input_resolution=9)
        self.trans5 = TransResBlock(conv_dim=1024, trans_dim=464, head_dim=29, window_size=3, drop_path=0.1, ratio=2,
                                    type='W', input_resolution=5)

        self.resUp4 = unetResUp(middle_channels=464, in_channels=1024, out_channels=512, output_lim=[25, 38])
        self.resUp3 = unetResUp(middle_channels=232, in_channels=512, out_channels=256, output_lim=[50, 76])
        self.resUp2 = unetResUp(middle_channels=116, in_channels=256, out_channels=128, output_lim=[100, 151])
        self.resUp1 = unetResUp(middle_channels=58, in_channels=128, out_channels=64, output_lim=[200, 301])

        self.last_layer = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        # Encoder Part
        temporalDown2 = self.temporalDown2(x)                    # (none, 58, 200, 151)
        temporalDown3 = self.temporalDown3(temporalDown2)        # (none, 116, 100, 76)
        temporalDown4 = self.temporalDown4(temporalDown3)        # (none, 232, 50, 38)
        temporalDown5 = self.temporalDown5(temporalDown4)        # (none, 464, 25, 19)

        temporal2spatio1 = self.temporal2spatio1(x)                    # (none, 64, 200, 301)
        temporal2spatio2 = self.temporal2spatio2(temporalDown2)        # (none, 128, 100, 151)
        temporal2spatio3 = self.temporal2spatio3(temporalDown3)        # (none, 256, 50, 76)
        temporal2spatio4 = self.temporal2spatio4(temporalDown4)        # (none, 512, 25, 38)
        temporal2spatio5 = self.temporal2spatio5(temporalDown5)        # (none, 1024, 13, 19)

        spatioDown2 = self.spatioDown2(temporal2spatio2, temporal2spatio1)    # (none, 128, 100, 151)
        spatioDown3 = self.spatioDown3(temporal2spatio3, spatioDown2)         # (none, 256, 50, 76)
        spatioDown4 = self.spatioDown4(temporal2spatio4, spatioDown3)         # (none, 512, 25, 38)
        spatioDown5 = self.spatioDown5(temporal2spatio5, spatioDown4)         # (none, 1024, 13, 19)

        spatioUp4 = self.spatioUp4(spatioDown4, spatioDown5)     # (none, 512, 25, 38)
        spatioUp3 = self.spatioUp3(spatioDown3, spatioUp4)       # (none, 256, 50, 76)
        spatioUp2 = self.spatioUp2(spatioDown2, spatioUp3)       # (none, 128, 100, 151)
        spatioUp1 = self.spatioUp1(temporal2spatio1, spatioUp2)  # (none, 64, 200, 301)

        trans1 = self.trans1(x)  # (none, 58, 200, 305)
        trans2 = self.trans2(temporalDown2)  # (none, 116, 100, 155)
        trans3 = self.trans3(temporalDown3)  # (none, 232, 50, 80)
        trans4 = self.trans4(temporalDown4)    # (none, 464, 25, 40)
        trans5 = self.trans5(temporalDown5, spatioDown5)  # (none, 1024, 13, 19)

        resUp4 = self.resUp4(trans4, trans5, spatioUp4)    # (none, 512, 25, 38)
        resUp3 = self.resUp3(trans3, resUp4, spatioUp3)    # (none, 256, 50, 76)
        resUp2 = self.resUp2(trans2, resUp3, spatioUp2)    # (none, 128, 100, 151)
        resUp1 = self.resUp1(trans1, resUp2, spatioUp1)    # (none, 64, 200, 301)

        resUp1 = F.interpolate(resUp1, size=(201, 301), mode="bilinear")
        outputs = self.last_layer(resUp1)
        return outputs

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

    model = TDFS_FWI_SEG()
    #
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"Parameter: {name}, Size: {param.size()}")

    #在网络中进行正向传播
    output = model(x)
    print(output.shape)  # (none, 42, 42, 1)