# -*- coding: utf-8 -*-
"""
Created on 2023/10/25 9:24

@author: XUQIONG  (xuqiong@swpu.edu.cn)

"""
################################################
########        IMPORT LIBARIES         ########
################################################
import os

from net.DDNet70 import DDNet70Model
from net.DST_FWI import DST_FWI

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import time
from PathConfig import *
from net.InversionNet import *
from net.FCNVMB import *
from net.ABA_FWI import *
from net.DST_FWI import *
from net.DST_FWI_GUASS import *
from data.data import *
from data.show import *
################################################
########         LOAD    NETWORK        ########
################################################

cuda_available = torch.cuda.is_available()
device = torch.device('cuda' if cuda_available else 'cpu')
NoiseFlag = False  #True False
DataSet = 'CurveVelB/'  # CurveVelA  FlatFaultA  FlatVelA   CurveFaultA   marmousi_70_70
model = 'DST_FWI_G_L1_tanh' # InversionNet  VelocityGAN  IAEDN_WTUU   Noise_IAEDN_WTUU
PreModelname = 'DST_FWI_G_L1_tanh_TrainSize24000_Epoch200_BatchSize10_LR0.0001_epoch130.pkl'

model_file = 'E:/Code/DST_FWI_4.0/train_result/'+ DataSet + PreModelname
net = DST_FWI_G_L1_tanh()   #InversionNet  IAEDN_WTUU Inversion_U  DDNet70Model

model_param = torch.load(model_file, map_location=torch.device('cpu'))
new_model_param = {}

# Iterate over the model parameters
for key, value in model_param.items():
    # Replace 'module.' in the key if it exists, otherwise keep the key unchanged
    if 'module.' in key:
        new_key = key.replace('module.', '')
    else:
        new_key = key

    # Assign the value to the new key in the new dictionary
    new_model_param[new_key] = value

# Load the new parameters into the model
net.load_state_dict(new_model_param)

# CurveVelA
# DST_FWI_G_L1_tanh_TrainSize24000_Epoch200_BatchSize10_LR0.0001_epoch160.pkl
# DST_FWI_G_L1_tanh_TrainSize24000_Epoch200_BatchSize10_LR0.0001_epoch165.pkl

# CurveFaultA
# DST_FWI_G_L1_tanh_TrainSize48000_Epoch160_BatchSize10_LR0.0001_epoch135.pkl
# DST_FWI_G_L1_tanh_noise_TrainSize48000_Epoch160_BatchSize20_LR0.0001_epoch130.pkl

# CurveVelB
# DST_FWI_G_L1_tanh_TrainSize24000_Epoch200_BatchSize10_LR0.0001_epoch130.pkl
# DST_FWI_G_L1_tanh_noise_TrainSize24000_Epoch160_BatchSize20_LR0.0001_epoch130.pkl

# CurveFaultB
# DST_FWI_G_L1_tanh_TrainSize48000_Epoch150_BatchSize20_LR0.0001_epoch80.pkl
# DST_FWI_G_L1_tanh_noise_TrainSize48000_Epoch160_BatchSize20_LR0.0001_epoch85.pkl

# marmousi_70_70
# DST_FWI_G_L1_tanh_TrainSize30926_Epoch160_BatchSize10_LR0.0001_epoch145.pkl

# net.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))

# CurveVelA
# CurveVelA_ABA-FWI.pkl
# CurveVelA_DDNet70.pkl
# CurveVelA_VelocityGAN.pkl
# CurveVelA_InversionNet.pkl
# DST_FWI_trainSize24000_Epoch200_BatchSize20_LR0.0001_epoch200.pkl
# DST_FWI_conv_TrainSize24000_Epoch200_BatchSize20_LR0.0001_epoch180.pkl
# DST_FWI_trans_TrainSize24000_Epoch200_BatchSize20_LR0.0001_epoch180.pkl
# Noise_DST_FWI_TrainSize24000_Epoch200_BatchSize20_LR0.0001_epoch180.pkl

#  CurveFaultA
# CurveFaultA_ABA-FWI.pkl
# CurveFaultA_DDNet70.pkl
# CurveFaultA_VelocityGAN.pkl
# CurveFaultA_InversionNet.pkl
# ABA-FWI_TrainSize48000_Epoch160_BatchSize20_LR0.0001_epoch105.pkl
# DST_FWI_TrainSize48000_Epoch200_BatchSize20_LR0.0001_epoch190.pkl
# DST_FWI_conv_TV_TrainSize48000_Epoch200_BatchSize20_LR0.0001_epoch150.pkl
# DST_FWI_G_TrainSize48000_Epoch200_BatchSize10_LR0.0001_epoch140.pkl


# CurveVelB
# ABA_FWI_TrainSize24000_Epoch200_BatchSize20_LR0.0001_epoch200.pkl
# DST_FWI_trainSize24000_Epoch200_BatchSize20_LR0.0001_epoch200.pkl
# DST_FWI_conv_TrainSize24000_Epoch200_BatchSize20_LR0.0001_epoch200.pkl
# InversionNet_CurEpo150.pkl
# CurveVelB_CLStage3_TrSize24000_AllEpo130_CurEpo130.pkl
# VelocityGAN_TrainSize24000_Epoch500_BatchSize20_LR0.0001_epoch460.pkl
# Noise_DST_FWI_TrainSize24000_Epoch200_BatchSize20_LR0.0001_epoch160.pkl
# DST_FWI_G_TrainSize24000_Epoch200_BatchSize10_LR0.0001_epoch150.pkl
# DST_FWI_g_l1_TrainSize24000_Epoch200_BatchSize10_LR0.0001_epoch200.pkl
# DST_FWI_g_l5_TrainSize24000_Epoch200_BatchSize10_LR0.0001_epoch200.pkl

# CurveFaultB
# ABA_FWI_TrainSize48000_Epoch200_BatchSize20_LR0.0001_epoch180.pkl
# DST_FWI_conv_TrainSize48000_Epoch200_BatchSize20_LR0.0001_epoch200.pkl
# DST_FWI_TrainSize48000_Epoch200_BatchSize20_LR0.0001_epoch200.pkl
# InversionNet_CurEpo150.pkl
# CurveFaultB_CLStage3_TrSize48000_AllEpo130_CurEpo130.pkl
# VelocityGAN_TrainSize48000_Epoch500_BatchSize20_LR0.0001_epoch400.pkl
# Noise_DST_FWI_TrainSize48000_Epoch200_BatchSize20_LR0.0001_epoch90.pkl

path =  'E:/Paper/DST-FWI/实验结果/OpenFWI_RESULTS/' + DataSet
Data_path = 'F:/Data/OpenFWI/' + DataSet    #  D:/Data/OpenFWI/  D:/Data/Marmousi/
################################################
########    LOADING TESTING DATA       ########
################################################

print('***************** 正在加载测试数据集 *****************')
TestSize = 500
TestBatchSize = 5
# 输出当前工作目录
dataset_dir = Data_path

testSet = Dataset_openfwi_test(dataset_dir, TestSize, 1, "seismic", "test")   # 11 for test  # dataset_dir, TestSize, 1, "seismic", "test"

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
Prediction_N = np.zeros((6, ModelDim[0], ModelDim[1]), dtype=float)
GT_N = np.zeros((6, ModelDim[0], ModelDim[1]), dtype=float)

total = 0

for i, (seismic_datas, vmodels, vmodel_max_min) in enumerate(test_loader):
    # Predictions
    net.eval()
    net.to(device)
    vmodels = vmodels[0].to(device)
    seismic_datas = seismic_datas[0].to(device)
    vmodel_max_min = vmodel_max_min[0].to(device)

    if NoiseFlag:
        # 添加高斯噪声
        seed = 42
        torch.manual_seed(seed)

        noise_mean = 0
        noise_std = 0.1
        noise = torch.normal(mean=noise_mean, std=noise_std, size=seismic_datas.shape).to(device)
        seismic_datas = seismic_datas + noise

    # Forward prediction
    outputs = net(seismic_datas)

    # 转换参数的输入类型为 float
    # outputs = outputs.to(torch.float32)
    # vmodels = vmodels.to(torch.float32)

    outputs = outputs.data.cpu().numpy()
    outputs = np.where(outputs > 0.0, outputs, 0.0)

    gts = vmodels.data.cpu().numpy()
    vmodel_max_min = vmodel_max_min.data.cpu().numpy()

    m = 0
    # Calculate the PSNR, SSIM
    for k in range(TestBatchSize):
        pd = outputs[k, :, :, :].reshape(ModelDim[0], ModelDim[1])
        gt = gts[k, :, :, :].reshape(ModelDim[0], ModelDim[1])
        vmax = vmodel_max_min[k, 0]
        vmin = vmodel_max_min[k, 1]
        if total in [80, 88]:
        # if total in range(100):
        # CurveFaultB [5, 26, 44, 55, 57, 65][26,44]
        # CurveFaultA [22, 54, 69, 85, 98][85, 98]
        # CurveVelB [21, 80, 87, 88, 98][80, 88]
        # CurveVelA [20, 29, 48, 62, 77, 91][48, 91]

        # 消融实验 CurveVelB [21, 80, 87, 88, 98][88]
        # CurveFaultA [22, 54, 69, 85, 98][98]
            pd_N = pd * (vmax - vmin) + vmin
            gt_N = gt * (vmax - vmin) + vmin
            gt_name = 'gt_'+ model + '_' + str(total) + '.png'
            pd_name = 'pd_' + model + '_' + str(total) + '.png'
            res_name = 'res_' + model + '_' + str(total) + '.png'
            # pain_openfwi_velocity_model1_save(gt_N, vmin, vmax, path, gt_name)
            pain_openfwi_velocity_model1_save(pd_N,vmin,vmax, path,pd_name)
            pain_openfwi_velocity_model1_save(np.abs(pd_N - gt_N),0,200, path, res_name)
            Prediction_N[m, :, :] = pd_N
            GT_N[m, :, :] = gt_N
            m = m + 1

            psnr = PSNR(gt, pd)
            ssim = SSIM(gt, pd)
            mse = MSE(pd, gt)
            mae = MAE(pd, gt)
            uqi = UQI(pd, gt)
            lpips = LPIPS(pd, gt)
            print('The %d testing psnr: %.2f, SSIM: %.4f, MSE:  %.4f, MAE:  %.4f, UQI:  %.4f, LPIPS: %.4f' % (m, psnr, ssim, mse, mae, uqi, lpips))
        total = total + 1


    SaveSelectedTestResults(Prediction_N, GT_N, model, path)  #InversionNet

