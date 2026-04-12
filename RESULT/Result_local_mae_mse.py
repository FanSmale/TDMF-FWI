
# -*- coding: utf-8 -*-
"""
Created on 2023/10/25 9:24

@author: XUQIONG  (xuqiong@swpu.edu.cn)

"""
################################################
########        IMPORT LIBARIES         ########
################################################
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import time
from PathConfig import *
from model.InversionNet import *
from model.FCNVMB import *
from model.IAEDN import *
from data.data import *
from data.show import *
from data.loss import *
################################################
########         LOAD    NETWORK        ########
################################################

cuda_available = torch.cuda.is_available()
device = torch.device('cuda' if cuda_available else 'cpu')


model_file = train_result_dir + PreModelname
net = IAEDN()   #InversionNet  IAEDN  FCNVMB
net.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))

################################################
########    LOADING TESTING DATA       ########
################################################

print('***************** 正在加载测试数据集 *****************')

# 输出当前工作目录
dataset_dir = Data_path

testSet = Dataset_openfwi4(dataset_dir, TestSize, 1, "seismic", "test")   # 11 for test

test_loader = DataLoader(testSet, batch_size=TestBatchSize, shuffle=False)

################################################
########            TESTING             ########
################################################

print()
print('*******************************************')
print('*******************************************')
print('                  开始测试                  ')
print('*******************************************')
print('*******************************************')
print()

# Initialization
since = time.time()

Total_PSNR = np.zeros((1, TestSize), dtype=float)
Total_SSIM = np.zeros((1, TestSize), dtype=float)
Total_MSE = np.zeros((1, TestSize), dtype=float)
Total_MAE = np.zeros((1, TestSize), dtype=float)
Total_UQI = np.zeros((1, TestSize), dtype=float)
Total_LPIPS = np.zeros((1, TestSize), dtype=float)


Prediction = np.zeros((TestSize, ModelDim[0], ModelDim[1]), dtype=float)
GT = np.zeros((TestSize, ModelDim[0], ModelDim[1]), dtype=float)
Prediction_N = np.zeros((3, ModelDim[0], ModelDim[1]), dtype=float)
GT_N = np.zeros((3, ModelDim[0], ModelDim[1]), dtype=float)

total = 0

for i, (seismic_datas, vmodels, edges) in enumerate(test_loader):
    # Predictions
    net.eval()
    net.to(device)
    vmodels = vmodels[0].to(device)
    seismic_datas = seismic_datas[0].to(device)
    edges = edges[0].to(device)
    edges = dilate_tv(edges)
    # Forward prediction
    outputs = net(seismic_datas)

    # 转换参数的输入类型为 float
    # outputs = outputs.to(torch.float32)
    # vmodels = vmodels.to(torch.float32)

    outputs = outputs.data.cpu().numpy()
    outputs = np.where(outputs > 0.0, outputs, 0.0)

    gts = vmodels.data.cpu().numpy()
    edges = edges.data.cpu().numpy()


    # Calculate the PSNR, SSIM
    for k in range(TestBatchSize):
        pd = outputs[k, :, :, :].reshape(ModelDim[0], ModelDim[1])
        gt = gts[k, :, :, :].reshape(ModelDim[0], ModelDim[1])
        edge = edges[k, :, :, :].reshape(ModelDim[0], ModelDim[1])

        mse = local_MSE(pd, gt, edge)
        mae = local_MAE(pd, gt, edge)

        Total_MSE[0, total] = mse
        Total_MAE[0, total] = mae

        total = total + 1

        print('The %d testing MSE:  %.4f, MAE:  %.4f' % (total,mse, mae))

print('local_MSE: %.6f, local_MAE: %.6f'% (np.mean(Total_MSE), np.mean(Total_MAE)))

