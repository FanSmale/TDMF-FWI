import numpy as np
import matplotlib.pyplot as plt
import scipy.io

import warnings
warnings.filterwarnings('ignore')

import time
import scipy.ndimage
import scipy.io
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import deepwave
import matplotlib
import torch
from data.show import *

import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def createSR(num_shots, num_sources_per_shot, num_receivers_per_shot, num_dims, source_spacing, receiver_spacing,
             source_depth, receiver_depth):
    """
        Create arrays containing the source and receiver locations
        Args:
            num_shots: nunmber of shots
            num_sources_per_shot: number of sources per shot
            num_receivers_per_shot： number of receivers per shot
            num_dims: dimension of velocity model
        return:
            x_s: Source locations [num_shots, num_sources_per_shot, num_dimensions]
            x_r: Receiver locations [num_shots, num_receivers_per_shot, num_dimensions]
    """
    x_s = torch.zeros(num_shots, num_sources_per_shot, num_dims)
    if source_depth != 0:
        x_s[:, 0, 0] = source_depth
    x_s[:, 0, 1] = torch.arange(1, num_shots + 1).float() * source_spacing
    x_r = torch.zeros(num_shots, num_receivers_per_shot, num_dims)
    if receiver_depth != 0:
        x_r[:, :, 0] = receiver_depth
    x_r[0, :, 1] = torch.arange(0, num_receivers_per_shot).float() * receiver_spacing
    x_r[:, :, 1] = x_r[0, :, 1].repeat(num_shots, 1)

    return x_s, x_r

# ================= 配置区域 =================
# 1. 模拟您的 1*70 数据 (请在此处替换为您真实的井数据)
# 假设这是速度数据 (单位: m/s)，例如从 1500 m/s 逐渐增加到 3000 m/s

dir = "E:/Code/DST_FWI_base_5.0/test_result/CurveVelB/"
TDMF_DIR = dir +"DST-FWI_TrainSize24000_Epoch160_BatchSize20_LR0.0001_TestResults.mat"
# CurveFaultB [44]10
# CurveFaultA [85]20
# CurveVelB [80]30
# CurveVelA [91]5
# i = 240 for marmousi
i = 80
j = 30
dz = 10  # 网格间距 10 米
pd_model = scipy.io.loadmat(TDMF_DIR)['Prediction'][i,:]
gt_model = scipy.io.loadmat(TDMF_DIR)['GT'][i,:]
pain_openfwi_velocity_model(pd_model)
pain_openfwi_velocity_model(gt_model)

peak_freq = 15.0
peak_source_time = 1 / peak_freq
dx = 10
dt = 0.001
nz = 70
ny = 70
nt = int(1 / dt)
num_dims = 2
num_shots = 5
num_sources_per_shot = 1
num_receivers_per_shot = ny
source_spacing = np.floor(ny / (num_shots+1))                         # space intervel between neigbouring sources
receiver_spacing = np.ceil(ny / (num_receivers_per_shot +1))

device = torch.device('cuda:0')


x_s, x_r = createSR(num_shots, num_sources_per_shot, num_receivers_per_shot, num_dims, source_spacing, receiver_spacing,source_depth=0,receiver_depth=0)
x_s, x_r = x_s.to(device), x_r.to(device)

## load or create initial source amplitude for inversion ###
source_amplitudes_true = (
    deepwave.wavelets.ricker(peak_freq, nt, dt, 1/peak_freq)
    .repeat(num_shots, num_sources_per_shot, 1)
    .to(device)
)
source_amplitudes = (
    deepwave.wavelets.ricker(peak_freq, nt, dt, 1/peak_freq)
)
source_amplitudes = torch.tensor(source_amplitudes).to(device)

out = deepwave.scalar(torch.from_numpy(pd_model).float().to(device), dx, dt,
                       source_amplitudes_true.to(device),
                       x_s.to(device),
                       x_r.to(device),
                      )
receiver_amplitudes_true = out[-1]

out1 = deepwave.scalar(torch.from_numpy(gt_model).float().to(device), dx, dt,
                       source_amplitudes_true.to(device),
                       x_s.to(device),
                       x_r.to(device),
                      )
receiver_amplitudes_pred = out1[-1]
i = 4
# pain_openfwi_seismic_data(receiver_amplitudes_true[i].detach().cpu().numpy().T)
# pain_openfwi_seismic_data(receiver_amplitudes_pred[i].detach().cpu().numpy().T)
# RES = receiver_amplitudes_true[i] - receiver_amplitudes_pred[i]
# pain_openfwi_seismic_data(RES.detach().cpu().numpy().T)

vmin, vmax = torch.quantile(
    receiver_amplitudes_true[0],
    torch.tensor([0.05, 0.95]).to(device),
)
plt.figure()
_, ax = plt.subplots(1, 3, figsize=(10.5, 7), sharey=True)
ax[0].imshow(
    receiver_amplitudes_true[2].cpu().T,
    aspect="auto",
    cmap="gray",
    vmin=vmin,
    vmax=vmax,
)
receiver_amplitudes_model = receiver_amplitudes_pred.detach().cpu().numpy()
ax[1].imshow(
    receiver_amplitudes_model[2].T,
    aspect="auto",
    cmap="gray",
    vmin=vmin,
    vmax=vmax,
)
ax[2].imshow(
    receiver_amplitudes_true[2].cpu().T - receiver_amplitudes_model[2].T,
    aspect="auto",
    cmap="gray",
    vmin=vmin,
    vmax=vmax,
)
ax[0].set_xlabel("Channel")
ax[0].set_ylabel("Time Sample")
ax[1].set_xlabel("Shot")
plt.tight_layout()
plt.show()
plt.savefig("example_forward_model.jpg")










