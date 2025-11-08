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
from net.InversionNet import *
from net.ABA_FWI import *
from data.data import *
from data.show import *
################################################
########         LOAD    NETWORK        ########
################################################

cuda_available = torch.cuda.is_available()
device = torch.device('cuda' if cuda_available else 'cpu')


model_file = train_result_dir + PreModelname
net = ABA_FWI()   # InversionNet  ABA_FWI  FCNVMB
net.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))

################################################
########    LOADING TESTING DATA       ########
################################################

print('***************** 正在加载测试数据集 *****************')

# 输出当前工作目录
vmodel_path = 'D:/Data/Marmousi/vmodels/vmodel_mar_big_117_567_75.npy'
seismic_path = 'D:/Data/Marmousi/seismic/seismic_mar_big_117_567_75.npy'   # vmodel_mar_smal_100_310_26   big_117_567_75
TestBatchSize = 75
TestSize = 75

testSet = Dataset_mar_test(seismic_path, vmodel_path, TestSize)   # 11 for test

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

for i, (seismic_datas, vmodels, vmodel_max_min) in enumerate(test_loader):
    # Predictions
    net.eval()
    net.to(device)
    vmodels = vmodels[0].to(device)
    seismic_datas = seismic_datas[0].to(device)
    vmodel_max_min = vmodel_max_min[0].to(device)

    # Forward prediction
    outputs = net(seismic_datas)

    # 转换参数的输入类型为 float
    # outputs = outputs.to(torch.float32)
    # vmodels = vmodels.to(torch.float32)

    outputs = outputs.data.cpu().numpy()
    outputs = np.where(outputs > 0.0, outputs, 0.0)

    gts = vmodels.data.cpu().numpy()
    vmodel_max_min = vmodel_max_min.data.cpu().numpy()

    # Calculate the PSNR, SSIM
    for k in range(TestBatchSize):
        pd = outputs[k, :, :, :].reshape(ModelDim[0], ModelDim[1])
        gt = gts[k, :, :].reshape(ModelDim[0], ModelDim[1])
        vmax = vmodel_max_min[k, 0]
        vmin = vmodel_max_min[k, 1]
        if total in [60, 61, 62, 63, 64, 65]:    #FlatFaultA[46, 60, 2] CurveFaultA[0, 5466, 5868]  FlatVelA[0, 2, 5][22, 35, 44] CurveVelA[0,2,3]
            pd_N = pd * (vmax - vmin) + vmin
            gt_N = gt * (vmax - vmin) + vmin
            PlotComparison_openfwi_velocity_model(pd_N, gt_N)
            # pain_openfwi_velocity_model(pd_N)
            # pain_openfwi_velocity_model(gt_N)
            # pain_openfwi_velocity_model(pd_N - gt_N)

        Prediction[i * TestBatchSize + k, :, :] = pd
        GT[i * TestBatchSize + k, :, :] = gt

        psnr = PSNR(gt, pd)
        ssim = SSIM(gt, pd)
        mse = MSE(pd, gt)
        mae = MAE(pd, gt)
        uqi = UQI(pd, gt)
        lpips = LPIPS(pd, gt)

        Total_PSNR[0, total] = psnr
        Total_SSIM[0, total] = ssim
        Total_MSE[0, total] = mse
        Total_MAE[0, total] = mae
        Total_UQI[0, total] = uqi
        Total_LPIPS[0, total] = lpips

        total = total + 1

        print('The %d testing psnr: %.2f, SSIM: %.4f, MSE:  %.4f, MAE:  %.4f, UQI:  %.4f, LPIPS: %.4f' % (total, psnr, ssim, mse, mae, uqi, lpips))

SaveTestResults(Total_PSNR, Total_SSIM, Total_MSE, Total_MAE, Total_UQI, Total_LPIPS,
                Prediction, GT, test_result_dir)

