# -*- coding: utf-8 -*-
"""
Created on 2025/10/17 11:15

@author: XUQIONG  (xuqiong@swpu.edu.cn)

"""
# -*- coding: utf-8 -*-

import torch
import numpy as np
import torch.nn as nn
from math import log10
from torch.autograd import Variable
import torch.autograd as autograd
import pandas as pd
import os
import cv2

from scipy.ndimage import uniform_filter
import matplotlib.pyplot as plt
import scipy.io
from skimage.metrics import structural_similarity as ssim
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PathConfig import *
import torch.nn.functional as F

import lpips
lp = lpips.LPIPS(net='alex', version="0.1")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def turn(GT):
    dim = GT.shape
    for j in range(0, dim[1]):
        for i in range(0, dim[0] // 2):
            temp = GT[i, j]
            GT[i, j] = GT[dim[0] - 1 - i, j]
            GT[dim[0] - 1 - i, j] = temp
    return GT


def PSNR(target, prediction):
    psnr = 20 * log10(abs(target.max()) / np.sqrt(np.sum((target - prediction) ** 2) / prediction.size))
    return psnr


def SSIM(target, prediction):
    return ssim(target, prediction, data_range=target.max() - target.min(), multichannel=True)  #True（默认值），则假定图像为多通道图像，SSIM将在每个通道上计算并返回通道之间的平均值。


def MSE(prediction, target):
    prediction = Variable(torch.from_numpy(prediction))
    target = Variable(torch.from_numpy(target))
    criterion = nn.MSELoss(reduction='mean')
    MSE = criterion(prediction, target)
    return MSE.item()


def MAE(prediction, target):
    prediction = Variable(torch.from_numpy(prediction))
    target = Variable(torch.from_numpy(target))
    criterion = nn.L1Loss(reduction='mean')
    mae = criterion(prediction, target)
    return mae.item()

def local_MSE(prediction, target, edge):
    temp = (prediction - target) ** 2 * edge
    # 计算矩阵中元素为 1 的个数
    count = np.sum(edge == 1)
    mse = np.sum(temp) / count
    return mse

def local_MAE(prediction, target, edge):
    temp = np.abs(prediction - target) * edge
    # 计算矩阵中元素为 1 的个数
    count = np.sum(edge == 1)
    mae = np.sum(temp)/count
    return mae

def _uqi_single(GT,P,ws):
    N = ws**2
    window = np.ones((ws,ws))

    GT_sq = GT*GT
    P_sq = P*P
    GT_P = GT*P

    GT_sum = uniform_filter(GT, ws)
    P_sum =  uniform_filter(P, ws)
    GT_sq_sum = uniform_filter(GT_sq, ws)
    P_sq_sum = uniform_filter(P_sq, ws)
    GT_P_sum = uniform_filter(GT_P, ws)

    GT_P_sum_mul = GT_sum*P_sum
    GT_P_sum_sq_sum_mul = GT_sum*GT_sum + P_sum*P_sum
    numerator = 4*(N*GT_P_sum - GT_P_sum_mul)*GT_P_sum_mul
    denominator1 = N*(GT_sq_sum + P_sq_sum) - GT_P_sum_sq_sum_mul
    denominator = denominator1*GT_P_sum_sq_sum_mul

    q_map = np.ones(denominator.shape)
    index = np.logical_and((denominator1 == 0) , (GT_P_sum_sq_sum_mul != 0))
    q_map[index] = 2*GT_P_sum_mul[index]/GT_P_sum_sq_sum_mul[index]
    index = (denominator != 0)
    q_map[index] = numerator[index]/denominator[index]

    s = int(np.round(ws/2))
    return np.mean(q_map[s:-s,s:-s])

def UQI(GT,P,ws=8):
    if len(GT.shape) == 2:
        GT = GT[:, :, np.newaxis]
        P = P[:, :, np.newaxis]

    GT = GT.astype(np.float64)
    P = P.astype(np.float64)
    return np.mean([_uqi_single(GT[:,:,i],P[:,:,i],ws) for i in range(GT.shape[2])])


def LPIPS(GT, P):
    '''

    :param GT: numpy
    :param P: numpy
    :return:
    '''
    GT_tensor = torch.from_numpy(GT)
    P_tensor = torch.from_numpy(P)
    return lp.forward(GT_tensor, P_tensor).item()


def extract_contours(para_image):
    '''
    Use Canny to extract contour features

    :param image:       Velocity model (numpy)
    :return:            Binary contour structure of the velocity model (numpy)  0\1 image
    '''

    image = para_image

    norm_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_image_to_255 = norm_image * 255
    norm_image_to_255 = norm_image_to_255.astype(np.uint8)
    canny = cv2.Canny(norm_image_to_255, 10, 15)
    bool_canny = np.clip(canny, 0, 1)
    return bool_canny


def dilate_tv(loss_out_w):

    # 创建膨胀的内核（kernel）
    kernel = torch.ones((1, 1, 3, 3), dtype=torch.float).to('cuda')  # 适用于多通道的 3x3 内核
    loss_out_w = loss_out_w.to(torch.float)
    # 使用卷积进行膨胀操作
    dilated_tensor = F.conv2d(loss_out_w, kernel,
                              padding=1, stride=1)
    result = torch.zeros_like(dilated_tensor)
    result[dilated_tensor != 0 ] = 1
    return result

def SaveTrainResults(loss, SavePath, ModelName, font2, font3):
    fig, ax = plt.subplots()
    plt.plot(loss[1:], linewidth=2)
    ax.set_xlabel('Num. of epochs', font2)
    ax.set_ylabel('MSE Loss', font2)
    ax.set_title('Training', font3)
    ax.set_xlim([1, 10])
    ax.set_xticks([i for i in range(0, Epochs+1, 20)])
    ax.set_xticklabels((str(i) for i in range(0, Epochs+1, 20)))
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(12)
    ax.grid(linestyle='dashed', linewidth=0.5)

    plt.savefig(SavePath + ModelName + 'TrainLoss.png', transparent=True)
    data = {}
    data['loss'] = loss
    scipy.io.savemat(SavePath + ModelName + 'TrainLoss.mat', data)
    plt.show()
    plt.close()


def SaveTrainValidResults(train_loss, val_loss, SavePath, ModelName, font2, font3):
    fig, ax = plt.subplots()
    plt.plot(train_loss[1:], label='Training')
    plt.plot(val_loss[1:], label='Validation')  # 绘制 y1，并添加标签
    ax.set_xlabel('Num. of epochs', font2)
    ax.set_ylabel('Loss', font2)
    ax.set_title('Training and validation Loss', font3)
    ax.set_xlim([1, 10])
    ax.set_xticks([i for i in range(0, Epochs+1, 20)])
    ax.set_xticklabels((str(i) for i in range(0, Epochs+1, 20)))
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(12)
    ax.grid(linestyle='dashed', linewidth=0.5)

    plt.savefig(SavePath + ModelName + 'TrainLoss.png', transparent=True)
    data = {'train_loss': train_loss, 'val_loss': val_loss}
    scipy.io.savemat(SavePath + ModelName + 'TrainValidLoss.mat', data)
    plt.show()
    plt.close()


def SaveTrainValidGANResults(train_loss_g, train_loss_d, val_loss, SavePath, ModelName, font2, font3):
    fig, ax = plt.subplots()
    plt.plot(train_loss_g[1:], label='Training_g')
    plt.plot(train_loss_d[1:], label='Training_d')
    plt.plot(val_loss[1:], label='Validation')  # 绘制 y1，并添加标签
    plt.legend()
    ax.set_xlabel('Num. of epochs', font2)
    ax.set_ylabel('Loss', font2)
    ax.set_title('Training and validation Loss', font3)
    ax.set_xlim([1, 10])
    ax.set_xticks([i for i in range(0, Epochs + 1, 20)])
    ax.set_xticklabels((str(i) for i in range(0, Epochs + 1, 20)))
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(12)
    ax.grid(linestyle='dashed', linewidth=0.5)

    plt.savefig(SavePath + ModelName + 'TrainLoss.png', transparent=True)
    data = {}
    data['train_loss_d'] = train_loss_g
    data['train_loss_d'] = train_loss_d
    data['val_loss'] = val_loss
    scipy.io.savemat(SavePath + ModelName + 'TrainValidLoss.mat', data)
    plt.show()
    plt.close()


def SaveTestResults(TotPSNR, TotSSIM, ToMSE, ToMAE, ToUQI, ToLPIPS, ToBMSE, ToBMAE, Prediction, GT, SavePath):
    data = {}
    data['TotPSNR'] = TotPSNR
    data['TotSSIM'] = TotSSIM
    data['ToMSE'] = ToMSE
    data['ToMAE'] = ToMAE
    data['ToUQI'] = ToUQI
    data['ToLPIPS'] = ToLPIPS
    data['ToBMSE'] = ToBMSE
    data['ToBMAE'] = ToBMAE
    data['GT'] = GT
    data['Prediction'] = Prediction
    print('TotPSNR: {}, TotSSIM: {},ToMSE: {}, ToMAE: {},ToUQI: {}, ToLPIPS: {}'.format(
        np.mean(TotPSNR), np.mean(TotSSIM), np.mean(ToMSE), np.mean(ToMAE), np.mean(ToUQI), np.mean(ToLPIPS), np.mean(ToBMSE), np.mean(ToBMAE)))

    file_path = SavePath + 'save_result.xlsx'
    df = pd.read_excel(file_path)

    Test_data = {
        'ModelName': ModelName,
        'TotPSNR': np.mean(TotPSNR),
        'TotSSIM': np.mean(TotSSIM),
        'ToMSE': np.mean(ToMSE),
        'ToMAE': np.mean(ToMAE),
        'ToUQI': np.mean(ToUQI),
        'ToLPIPS': np.mean(ToLPIPS),
        'ToBMSE': np.mean(ToBMSE),
        'ToBMAE': np.mean(ToBMAE),
    }
    Test_data = pd.DataFrame(Test_data, index=[0])
    df = pd.concat([df, Test_data], ignore_index=True)

    # df = df.append(Test_data, ignore_index=True)
    df.to_excel(file_path, index=False)

    scipy.io.savemat(SavePath + ModelName + '_TestResults.mat', data)

def SaveSelectedTestResults(Prediction_N, GT_N, Model, Path):
    data = {}
    data['Prediction_N'] = Prediction_N
    data['GT_N'] = GT_N
    scipy.io.savemat(Path + Model + '_TestResults.mat', data)