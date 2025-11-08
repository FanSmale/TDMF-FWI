# -*- coding: utf-8 -*-
"""
Created on 2025/10/16 20:18

@author: XUQIONG  (xuqiong@swpu.edu.cn)

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.show import *
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
        x0 = x.permute(0, 3, 1, 2)           # (none, 70, 5, 1000)       (none, 35, 10, 500)
        x1 = self.depthwise_conv1(x0)        # (none, 70, 5, 500)        (none, 35, 10, 250)
        x2 = self.depthwise_conv2(x1)       # (none, 70, 5, 500)         (none, 35, 10, 500)

        x3 = self.pointwise_conv1(x2)        # (none, 35, 5, 500)        (none, 18, 10, 250)
        x4 = x3.permute(0, 2, 3, 1)          # (none, 5, 500, 35)        (none, 10, 250, 18)
        y = self.pointwise_conv2(x4)         # (none, 10, 500, 35)       (none, 20, 250, 18)
        return y

class SpatioConv(nn.Module):
    def __init__(self, in_channels):
        super(SpatioConv, self).__init__()
        ratio = in_channels // 5
        self.pre_dim_reducer1 = ConvBlock(in_channels, 8*ratio, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.pre_dim_reducer2 = ConvBlock(8*ratio, 8*ratio, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.pre_dim_reducer3 = ConvBlock(8*ratio, 16*ratio, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.pre_dim_reducer4 = ConvBlock(16*ratio, 16*ratio, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.pre_dim_reducer5 = ConvBlock(16*ratio, 32*ratio, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.pre_dim_reducer6 = ConvBlock(32*ratio, 32*ratio, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):

        width = x.shape[3]
        new_size = [width * 8, width]
        dimred0 = F.interpolate(x, size=new_size, mode='bilinear', align_corners=False)   # (none, 5, 560, 70)

        dimred1 = self.pre_dim_reducer1(dimred0)     # (none, 8, 280, 70)
        dimred2 = self.pre_dim_reducer2(dimred1)     # (none, 8, 280, 70)
        dimred3 = self.pre_dim_reducer3(dimred2)     # (none, 16, 140, 70)
        dimred4 = self.pre_dim_reducer4(dimred3)     # (none, 16, 140, 70)
        dimred5 = self.pre_dim_reducer5(dimred4)     # (none, 32, 70, 70)
        dimred6 = self.pre_dim_reducer6(dimred5)     # (none, 32, 70, 70)

        return dimred6

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
        # input2 （32, 70, 70） input1 (64, 35, 35)
        skip_output = self.wtConv(inputs2)   #（32, 70, 70）
        outputs2_0 = self.conv_1(skip_output)  #（64, 70, 70）
        outputs2_1 = self.conv_2(outputs2_0)  #（64, 35, 35）
        outputs = torch.cat([inputs1, outputs2_1], 1)  #（128, 35, 35）
        outputs = self.conv(outputs)  #（64, 35, 35）
        outputs = self.spatial_attention(outputs)  #（64, 35, 35）
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
         # input1 (256, 9, 9)  input2 (512, 5, 5)
        input2 = self.deconv(input2)   # input2 (256, 10, 10)
        input2 = F.interpolate(input2, size=self.output_lim, mode='bilinear', align_corners=False)   # input2 (256, 9, 9)
        input = torch.cat([input1, input2], 1)   # （512， 9， 9）
        input = self.conv(input) #  (256, 9, 9)
        output = self.spatial_attention(input) #  (256, 9, 9)

        return output

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

class E_TDM_FWI(nn.Module):
    '''
    E_TDC_FWI = ABA-FWI + E_TDC
    '''
    def __init__(self):
        super(E_TDM_FWI, self).__init__()
        self.temporalDown2 = TemporalConv(in_channels=5, width_in=70, width_out=35)
        self.temporalDown3 = TemporalConv(in_channels=10, width_in=35, width_out=18)
        self.temporalDown4 = TemporalConv(in_channels=20, width_in=18, width_out=9)
        self.temporalDown5 = TemporalConv(in_channels=40, width_in=9, width_out=5)

        self.temporal2spatio1 = SpatioConv(in_channels=5)
        self.temporal2spatio2 = SpatioConv(in_channels=10)
        self.temporal2spatio3 = SpatioConv(in_channels=20)
        self.temporal2spatio4 = SpatioConv(in_channels=40)
        self.temporal2spatio5 = SpatioConv(in_channels=80)

        self.spatioDown2 = unetDown(in_channels=32, out_channels=64)
        self.spatioDown3 = unetDown(in_channels=64, out_channels=128)
        self.spatioDown4 = unetDown(in_channels=128, out_channels=256)
        self.spatioDown5 = unetDown(in_channels=256, out_channels=512)

        self.spatioUp4 = unetUp(in_channels=512, out_channels=256, output_lim=[9, 9])
        self.spatioUp3 = unetUp(in_channels=256, out_channels=128, output_lim=[18, 18])
        self.spatioUp2 = unetUp(in_channels=128, out_channels=64, output_lim=[35, 35])
        self.spatioUp1 = unetUp(in_channels=64, out_channels=32, output_lim=[70, 70])

        self.last_layer = ConvBlock_Tanh(32, 1)

    def forward(self, x):
        # Encoder Part
        temporalDown2 = self.temporalDown2(x)                    # (none, 10, 500, 35)
        temporalDown3 = self.temporalDown3(temporalDown2)        # (none, 20, 250, 18)
        temporalDown4 = self.temporalDown4(temporalDown3)        # (none, 40, 125, 9)
        temporalDown5 = self.temporalDown5(temporalDown4)        # (none, 80, 63, 5)

        temporal2spatio1 = self.temporal2spatio1(x)                    # (none, 32, 70, 70)
        temporal2spatio2 = self.temporal2spatio2(temporalDown2)        # (none, 64, 35, 35)
        temporal2spatio3 = self.temporal2spatio3(temporalDown3)        # (none, 128, 18, 18)
        temporal2spatio4 = self.temporal2spatio4(temporalDown4)        # (none, 256, 9, 9)
        temporal2spatio5 = self.temporal2spatio5(temporalDown5)        # (none, 512, 5, 5)

        spatioDown2 = self.spatioDown2(temporal2spatio2, temporal2spatio1)    # (none, 64, 35, 35)
        spatioDown3 = self.spatioDown3(temporal2spatio3, spatioDown2)         # (none, 128, 18, 18)
        spatioDown4 = self.spatioDown4(temporal2spatio4, spatioDown3)         # (none, 256, 9, 9)
        spatioDown5 = self.spatioDown5(temporal2spatio5, spatioDown4)         # (none, 512, 5, 5)

        spatioUp4 = self.spatioUp4(spatioDown4, spatioDown5)     # (none, 256, 9, 9)
        spatioUp3 = self.spatioUp3(spatioDown3, spatioUp4)       # (none, 128, 18, 18)
        spatioUp2 = self.spatioUp2(spatioDown2, spatioUp3)       # (none, 64, 35, 35)
        spatioUp1 = self.spatioUp1(temporal2spatio1, spatioUp2)  # (none, 32, 70, 70)

        outputs = self.last_layer(spatioUp1)
        return outputs

class WMSA_G(nn.Module):
    def __init__(self, input_dim=5, output_dim=5, head_dim=5, window_size=7, ratio=15, type='W'):
        super(WMSA_G, self).__init__()
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
        self.global_bias = self._create_global_attention_bias()

    def _create_global_attention_bias(self):
        """
        创建一个高斯型先验偏置，鼓励注意力集中在窗口中心区域
        返回: (1, 1, W*W, W*W) 的固定 buffer
        """
        W = self.window_size
        # 1. 构建相对坐标差矩阵 (W, W)
        coords = torch.arange(W).view(-1, 1) - torch.arange(W).view(1, -1)  # (W, W)
        # 2. 高斯核：σ = W / 3
        sigma = W / 3.0
        gauss_kernel = torch.exp(-coords ** 2 / (2 * sigma ** 2))  # (W, W)
        # 3. 展平为 (W*W, W*W)，对每个位置对 (i,j) 赋值为高斯相似度
        # 注意：这里我们假设展平顺序是 row-major: (0,0), (0,1), ..., (W-1,W-1)
        gauss_kernel_flat = gauss_kernel.view(W * W, 1) @ gauss_kernel.view(1, W * W)  # 外积？不，我们要的是基于距离
        # 更准确的做法：基于每对 token 的欧氏距离构建先验
        coords_flat = torch.stack([torch.arange(W).repeat(W), torch.arange(W).repeat(W, 1).t().contiguous().view(-1)],
                                  dim=1)  # (W*W, 2)
        dist_sq = torch.sum((coords_flat[:, None, :] - coords_flat[None, :, :]) ** 2, dim=-1)  # (W*W, W*W)
        prior = torch.exp(-dist_sq / (2 * sigma ** 2))  # (W*W, W*W)
        # 4. 扩展维度并注册为 buffer
        prior = prior.unsqueeze(0).unsqueeze(0)  # (1, 1, W*W, W*W)
        self.register_buffer('global_attention_prior', prior)
        return prior

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

    def forward(self, x):  # x.shape: (1000, 70, 5)
        if self.type != 'W':
            x = torch.roll(x, shifts=(-(self.window_size // 2), -(self.window_size // 2)), dims=(1, 2))
        x = rearrange(x, 'DataSplite (w1 p1) (w2 p2) c -> DataSplite w1 w2 p1 p2 c',
                      p1=self.window_size, p2=self.window_size)  # (1, 150, 10, 7, 7, 5)

        h_windows, w_windows = x.size(1), x.size(2)

        x = rearrange(x, 'DataSplite w1 w2 p1 p2 c -> DataSplite (w1 w2) (p1 p2) c',
                      p1=self.window_size, p2=self.window_size)  # (1, 1500, 49, 5)

        qkv = self.embedding_layer(x)  # (1, 1500, 49, 5)
        q, k, v = rearrange(qkv, 'DataSplite nw np (threeh c) -> threeh DataSplite nw np c',
                            c=self.head_dim).chunk(3, dim=0)   # (1, 1, 1500, 49, 5)

        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')

        sim = sim + self.global_attention_prior

        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size // 2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h DataSplite w p c -> DataSplite w p (h c)')
        output = self.linear(output)      # (1, 1500, 49, 5)
        output = rearrange(output, 'DataSplite (c w1 w2) (p1 p2) c1 -> DataSplite (w1 p1) (w2 p2) (c c1)',
                           c=self.ratio, w2=w_windows, p1=self.window_size)    # (1, 70, 70, 75)

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
        self.msa = WMSA_G(input_dim, input_dim, head_dim, window_size, self.ratio, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim * self.ratio)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * self.ratio, 4 * input_dim * self.ratio),
            nn.ReLU(),
            nn.Linear(4 * input_dim * self.ratio, output_dim * self.ratio),
        )
    def forward(self, x):    # 1050, 70, 5
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
        pad_dim = ((h + w - 1) // w) * w
        x_padded = F.pad(x, (0, 0, 0, pad_dim-h))  # (left, right, top, bottom)
        trans_x = Rearrange('DataSplite c h w -> DataSplite h w c')(x_padded)   # 1, 5, 1050, 70
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
        pad_dim = ((h + w - 1) // w) * w
        x_padded = F.pad(x, (0, 0, 0, pad_dim-h))  # (left, right, top, bottom)
        trans_x = Rearrange('DataSplite c h w -> DataSplite h w c')(x_padded)   # 1, 5, 1000, 70
        trans_x = self.trans_block(trans_x)  # 1, 70, 70, 75
        trans_x = Rearrange('DataSplite h w c -> DataSplite c h w')(trans_x)  # 1, 15, 70, 70
        res = self.conv1_1(trans_x)
        output = torch.cat((y, res), dim=1)
        output = self.conv1_2(output)
        return output

class TDMF_FWI(nn.Module):
    def __init__(self):
        super(TDMF_FWI, self).__init__()
        self.temporalDown2 = TemporalConv(in_channels=5, width_in=70, width_out=35)
        self.temporalDown3 = TemporalConv(in_channels=10, width_in=35, width_out=18)
        self.temporalDown4 = TemporalConv(in_channels=20, width_in=18, width_out=9)
        self.temporalDown5 = TemporalConv(in_channels=40, width_in=9, width_out=5)

        self.temporal2spatio1 = SpatioConv(in_channels=5)
        self.temporal2spatio2 = SpatioConv(in_channels=10)
        self.temporal2spatio3 = SpatioConv(in_channels=20)
        self.temporal2spatio4 = SpatioConv(in_channels=40)
        self.temporal2spatio5 = SpatioConv(in_channels=80)

        self.spatioDown2 = unetDown(in_channels=32, out_channels=64)
        self.spatioDown3 = unetDown(in_channels=64, out_channels=128)
        self.spatioDown4 = unetDown(in_channels=128, out_channels=256)
        self.spatioDown5 = unetDown(in_channels=256, out_channels=512)

        self.spatioUp4 = unetUp(in_channels=512, out_channels=256, output_lim=[9, 9])
        self.spatioUp3 = unetUp(in_channels=256, out_channels=128, output_lim=[18, 18])
        self.spatioUp2 = unetUp(in_channels=128, out_channels=64, output_lim=[35, 35])
        self.spatioUp1 = unetUp(in_channels=64, out_channels=32, output_lim=[70, 70])

        self.trans5 = TransResBlock(conv_dim=512, trans_dim=80, head_dim=20, window_size=5, drop_path=0.1, ratio=13, type='W', input_resolution=5)
        self.trans4 = TransBlock(trans_dim=40, head_dim=10, window_size=3, drop_path=0.1, ratio=14, type='W', input_resolution=9)
        self.trans3 = TransBlock(trans_dim=20, head_dim=10, window_size=3, drop_path=0.1, ratio=14, type='W', input_resolution=18)
        self.trans2 = TransBlock(trans_dim=10, head_dim=5, window_size=5, drop_path=0.1, ratio=15, type='W', input_resolution=35)
        self.trans1 = TransBlock(trans_dim=5, head_dim=5, window_size=7, drop_path=0.1, ratio=15, type='W', input_resolution=70)

        self.resUp4 = unetResUp(in_channels=512, middle_channels=40 * 14, out_channels=256, output_lim=[9, 9])
        self.resUp3 = unetResUp(in_channels=256, middle_channels=20 * 14, out_channels=128, output_lim=[18, 18])
        self.resUp2 = unetResUp(in_channels=128, middle_channels=10 * 15, out_channels=64, output_lim=[35, 35])
        self.resUp1 = unetResUp(in_channels=64, middle_channels=5 * 15, out_channels=32, output_lim=[70, 70])

        self.last_layer = ConvBlock_Tanh(32, 1)

    def forward(self, x):
        # Encoder Part
        temporalDown2 = self.temporalDown2(x)                    # (none, 10, 500, 35)
        temporalDown3 = self.temporalDown3(temporalDown2)        # (none, 20, 250, 18)
        temporalDown4 = self.temporalDown4(temporalDown3)        # (none, 40, 125, 9)
        temporalDown5 = self.temporalDown5(temporalDown4)        # (none, 80, 63, 5)

        temporal2spatio1 = self.temporal2spatio1(x)                    # (none, 32, 70, 70)
        temporal2spatio2 = self.temporal2spatio2(temporalDown2)        # (none, 64, 35, 35)
        temporal2spatio3 = self.temporal2spatio3(temporalDown3)        # (none, 128, 18, 18)
        temporal2spatio4 = self.temporal2spatio4(temporalDown4)        # (none, 256, 9, 9)
        temporal2spatio5 = self.temporal2spatio5(temporalDown5)        # (none, 512, 5, 5)

        spatioDown2 = self.spatioDown2(temporal2spatio2, temporal2spatio1)    # (none, 64, 35, 35)
        spatioDown3 = self.spatioDown3(temporal2spatio3, spatioDown2)         # (none, 128, 18, 18)
        spatioDown4 = self.spatioDown4(temporal2spatio4, spatioDown3)         # (none, 256, 9, 9)
        spatioDown5 = self.spatioDown5(temporal2spatio5, spatioDown4)         # (none, 512, 5, 5)

        spatioUp4 = self.spatioUp4(spatioDown4, spatioDown5)     # (none, 256, 9, 9)
        spatioUp3 = self.spatioUp3(spatioDown3, spatioUp4)       # (none, 128, 18, 18)
        spatioUp2 = self.spatioUp2(spatioDown2, spatioUp3)       # (none, 64, 35, 35)
        spatioUp1 = self.spatioUp1(temporal2spatio1, spatioUp2)  # (none, 32, 70, 70)

        trans5 = self.trans5(temporalDown5, spatioDown5)   # (none, 512, 5, 5)
        trans4 = self.trans4(temporalDown4)    # (none, 560, 9, 9)
        trans3 = self.trans3(temporalDown3)    # (none, 280, 18, 18)
        trans2 = self.trans2(temporalDown2)    # (none, 150, 35, 35)
        trans1 = self.trans1(x)                # (none, 75, 70, 70)

        resUp4 = self.resUp4(trans4, trans5, spatioUp4)    # (none, 256, 9, 9)
        resUp3 = self.resUp3(trans3, resUp4, spatioUp3)    # (none, 128, 18, 18)
        resUp2 = self.resUp2(trans2, resUp3, spatioUp2)    # (none, 64, 35, 35)
        resUp1 = self.resUp1(trans1, resUp2, spatioUp1)    # (none, 32, 70, 70)

        outputs = self.last_layer(resUp1)

        # for feature maps
        # return spatioUp1, trans1, spatioUp2, trans2, spatioUp3, trans3, spatioUp4, trans4,
        # return spatioDown5, spatioUp4, spatioUp3, spatioUp2, spatioUp1, trans5, trans4, trans3, trans2, trans1
        # return spatioDown5, spatioUp4, spatioUp3, spatioUp2, spatioUp1, trans5, trans4, trans3, trans2, trans1, trans5, resUp4, resUp3, resUp2, resUp1, temporal2spatio5, temporal2spatio4, temporal2spatio3, temporal2spatio2, temporal2spatio1
        # return temporalDown5, temporalDown4, temporalDown3, temporalDown2, x
        return  outputs

class I_DTC(nn.Module):
    def __init__(self):
        super(I_DTC, self).__init__()
        self.temporalDown2 = TemporalConv(in_channels=5, width_in=70, width_out=35)
        self.temporalDown3 = TemporalConv(in_channels=10, width_in=35, width_out=18)
        self.temporalDown4 = TemporalConv(in_channels=20, width_in=18, width_out=9)
        self.temporalDown5 = TemporalConv(in_channels=40, width_in=9, width_out=5)

        self.trans5 = TransBlock(trans_dim=80, head_dim=20, window_size=5, drop_path=0.1, ratio=13, type='W', input_resolution=5)
        self.trans4 = TransBlock(trans_dim=40, head_dim=10, window_size=3, drop_path=0.1, ratio=14, type='W', input_resolution=9)
        self.trans3 = TransBlock(trans_dim=20, head_dim=10, window_size=3, drop_path=0.1, ratio=14, type='W', input_resolution=18)
        self.trans2 = TransBlock(trans_dim=10, head_dim=5, window_size=5, drop_path=0.1, ratio=15, type='W', input_resolution=35)
        self.trans1 = TransBlock(trans_dim=5, head_dim=5, window_size=7, drop_path=0.1, ratio=15, type='W', input_resolution=70)

        self.resUp4 = unetTransUp(in_channels=80 * 13, out_channels=40 * 14, output_lim=[9, 9])
        self.resUp3 = unetTransUp(in_channels=40 * 14, out_channels=20 * 14, output_lim=[18, 18])
        self.resUp2 = unetTransUp(in_channels=20 * 14, out_channels=10 * 15, output_lim=[35, 35])
        self.resUp1 = unetTransUp(in_channels=10 * 15, out_channels=5 * 15, output_lim=[70, 70])

        self.last_layer = ConvBlock_Tanh(5 * 15, 1)

    def forward(self, x):
        # Encoder Part
        temporalDown2 = self.temporalDown2(x)                    # (none, 10, 500, 35)
        temporalDown3 = self.temporalDown3(temporalDown2)        # (none, 20, 250, 18)
        temporalDown4 = self.temporalDown4(temporalDown3)        # (none, 40, 125, 9)
        temporalDown5 = self.temporalDown5(temporalDown4)        # (none, 80, 63, 5)

        trans5 = self.trans5(temporalDown5)   # (none, 1040, 5, 5)
        trans4 = self.trans4(temporalDown4)    # (none, 560, 9, 9)
        trans3 = self.trans3(temporalDown3)    # (none, 280, 18, 18)
        trans2 = self.trans2(temporalDown2)    # (none, 150, 35, 35)
        trans1 = self.trans1(x)                # (none, 75, 70, 70)

        resUp4 = self.resUp4(trans4, trans5)    # (none, 560, 9, 9)
        resUp3 = self.resUp3(trans3, resUp4)    # (none, 280, 18, 18)
        resUp2 = self.resUp2(trans2, resUp3)    # (none, 150, 35, 35)
        resUp1 = self.resUp1(trans1, resUp2)    # (none, 75, 70, 70)

        outputs = self.last_layer(resUp1)

        return outputs

if __name__ == '__main__':
    # 假设的 70×70 索引表
    x = torch.randn(1, 5, 1000, 70)  # 可以是一些预定义的值，或者根据实际情况进行学习
    y = torch.randn(32, 1, 70, 70)

    # 实例化网络模型
    model = TDMF_FWI()

    # 前向传播
    output = model(x)
    print(output.shape)  # 输出形状应为 (32, 70, 70)
