# -*- coding: utf-8 -*-
"""
Created on 2024/08/25 9:24

@author: XUQIONG  (xuqiong@swpu.edu.cn)

"""

################################################
########        IMPORT LIBARIES         ########
################################################
import time
from net.ABA_FWI import *
from net.TDMF_FWI import *
from net.InversionNet import *
from data.data import *
from data.show import *
from net.DDNet70 import *
from data.loss import *

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

################################################
########         LOAD    NETWORK        ########
################################################

cuda_available = torch.cuda.is_available()
device = torch.device('cuda' if cuda_available else 'cpu')

model_file = train_result_dir + PreModelname
# model_file  = 'D:/Xu-Qiong/ABA-FWI_2.0/model/CurveVelA_ABA-FWI.pkl'
# FlatVelA_InversionNet.pkl   CL_FlatVelA_DDNet70.pkl
# FlatFaultA_InversionNet.pkl   CL_FlatFaultA_DDNet70.pkl
# CurveVelA_InversionNet.pkl    CL_CurveVelA_DDNet70.pkl
# CurveFaultA_InversionNet.pkl    CurveFaultA_DDNet70.pkl
net = DST_FWI()  # InversionNet | VelocityGAN | DDNet70Model | ABA_FWI |TS_FWI
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

################################################
########    LOADING TESTING DATA       ########
################################################

print('***************** Loading dataset *****************')

dataset_dir = Data_path

testSet = Dataset_openfwi4_test(dataset_dir, TestSize, 1, "seismic", "test")

test_loader = DataLoader(testSet, batch_size=TestBatchSize, shuffle=False)

################################################
########            TESTING             ########
################################################

print()
print('*******************************************')
print('*******************************************')
print('                  Testing...               ')
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
Total_BMSE = np.zeros((1, TestSize), dtype=float)
Total_BMAE = np.zeros((1, TestSize), dtype=float)

Prediction = np.zeros((TestSize, ModelDim[0], ModelDim[1]), dtype=float)
GT = np.zeros((TestSize, ModelDim[0], ModelDim[1]), dtype=float)
Prediction_N = np.zeros((3, ModelDim[0], ModelDim[1]), dtype=float)
GT_N = np.zeros((3, ModelDim[0], ModelDim[1]), dtype=float)

total = 0

for i, (seismic_datas, vmodels, edges, vmodel_max_min) in enumerate(test_loader):
    # Predictions
    net.eval()
    net.to(device)
    vmodels = vmodels[0].to(device)
    seismic_datas = seismic_datas[0].to(device)
    edges = edges[0].to(device)
    vmodel_max_min = vmodel_max_min[0].to(device)
    edges = dilate_tv(edges)

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

    outputs = outputs.data.cpu().numpy()   #outputs = outputs[0].data.cpu().numpy()  for dd-net
    outputs = np.where(outputs > 0.0, outputs, 0.0)

    gts = vmodels.data.cpu().numpy()
    vmodel_max_min = vmodel_max_min.data.cpu().numpy()

    edges = edges.data.cpu().numpy()

    # Calculate the PSNR, SSIM
    for k in range(outputs.shape[0]):   # TestBatchSize
        pd = outputs[k, :, :, :].reshape(ModelDim[0], ModelDim[1])
        gt = gts[k, :, :, :].reshape(ModelDim[0], ModelDim[1])
        edge = edges[k, :, :, :].reshape(ModelDim[0], ModelDim[1])
        vmax = vmodel_max_min[k, 0]
        vmin = vmodel_max_min[k, 1]

        # FlatFaultA[46, 60, 2] CurveFaultA[0, 5466, 5868]  FlatVelA[22, 35, 44] CurveVelA[0,2,3]
        # 消融实验  FlatFaultA[76, 140, 109]    CurveFaultA[3,5,11]
        if total in [0, 2, 3]:
            output1 = torch.tensor(gt).to('cuda').unsqueeze(0).unsqueeze(0)
            vmodel1 = torch.tensor(gt).to('cuda').unsqueeze(0).unsqueeze(0)
            edge1 = torch.tensor(edge).to('cuda').unsqueeze(0).unsqueeze(0)
            loss_tv = loss_tv1(output1, vmodel1, edge1)
            pd_N = pd * (vmax - vmin) + vmin
            gt_N = gt * (vmax - vmin) + vmin
            # pain_openfwi_velocity_model(pd_N)
            # pain_openfwi_velocity_model(gt_N)
            # pain_openfwi_velocity_model(edge)
        pd_N = pd * (vmax - vmin) + vmin
        gt_N = gt * (vmax - vmin) + vmin
        Prediction[i * TestBatchSize + k, :, :] = pd_N
        GT[i * TestBatchSize + k, :, :] = gt_N

        psnr = PSNR(gt, pd)
        ssim = SSIM(gt, pd)
        mse = MSE(pd, gt)
        mae = MAE(pd, gt)
        uqi = UQI(pd, gt)
        lpips = LPIPS(pd, gt)
        bmse = local_MSE(pd, gt, edge)
        bmae = local_MAE(pd, gt, edge)

        Total_PSNR[0, total] = psnr
        Total_SSIM[0, total] = ssim
        Total_MSE[0, total] = mse
        Total_MAE[0, total] = mae
        Total_UQI[0, total] = uqi
        Total_LPIPS[0, total] = lpips
        Total_BMSE[0, total] = bmse
        Total_BMAE[0, total] = bmae

        total = total + 1

        print('The %d testing psnr: %.2f, SSIM: %.4f, MSE:  %.4f, MAE:  %.4f, UQI:  %.4f, LPIPS: %.4f, BMSE:  %.4f, BMAE:  %.4f' % (total, psnr,
                                                                                                          ssim, mse,
                                                                                                          mae, uqi,
                                                                                                          lpips,bmse,bmae))

SaveTestResults(Total_PSNR, Total_SSIM, Total_MSE, Total_MAE, Total_UQI, Total_LPIPS, Total_BMSE, Total_BMAE,
                Prediction, GT, test_result_dir)
