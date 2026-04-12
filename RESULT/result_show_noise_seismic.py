# -*- coding: utf-8 -*-
"""
Created on 2023/10/18 17:23

@author: XUQIONG  (xuqiong@swpu.edu.cn)

"""
import numpy as np
from data.show import *
from scipy.io import savemat, loadmat
import os
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib as mpl
import numpy as np
import cv2
mpl.use('TkAgg')


if __name__ == '__main__':
    data_path = 'D:/Data/Marmousi/marmousi_70_70/test_data/seismic/seismic1.npy'
    #
    # D:/Xu-Qiong/Data/FlatFaultA/train_data/seismic/seismic2.npy
    seismic_data = np.load(data_path, mmap_mode='r')
    noise_mean = 0
    noise_std = 0.1
    noise = np.random.normal(noise_mean, noise_std, seismic_data.shape)
    seismic_noise_data = seismic_data + noise
    pain_openfwi_seismic_data1(seismic_data[170, 2, :, :])
    pain_openfwi_seismic_data1(seismic_noise_data[0, 2, :, :])
