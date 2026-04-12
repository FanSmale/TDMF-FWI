import scipy.io
import numpy as np

import matplotlib.pyplot as plt
font21 = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': 21,
}

def plot_velocity_image(column_index, num, output1, output2, output3, output4, output5, target, vmin, vmax):
    fig = plt.figure(figsize=(12, 6))
    #column_index = 30
    pixel_values1, pixel_values2, pixel_values3, pixel_values4, pixel_values5, pixel_values6,  pixel_values7 = [], [], [], [], [], [], []
    for y in range(output1.shape[0]):
        pixel_value1 = output1[y, column_index]
        pixel_value2 = output2[y, column_index]
        pixel_value3 = output3[y, column_index]
        pixel_value4 = output4[y, column_index]
        pixel_value5 = output5[y, column_index]
        pixel_value6 = target[y, column_index]
        pixel_values1.append(pixel_value1)
        pixel_values2.append(pixel_value2)
        pixel_values3.append(pixel_value3)
        pixel_values4.append(pixel_value4)
        pixel_values5.append(pixel_value5)
        pixel_values6.append(pixel_value6)

    plt.plot(pixel_values1, color='blue', linewidth=2, label='InversionNet')
    plt.plot(pixel_values2, color='purple', linewidth=2, label='VelocityGAN')
    plt.plot(pixel_values3, color='green', linewidth=2, label='DD-Net70')
    plt.plot(pixel_values4, color='orange', linewidth=2, label='ABA-FWI')
    plt.plot(pixel_values5, color='red', linewidth=2, label='TDMF-FWI')
    plt.plot(pixel_values6, color='black', linewidth=2, label='Ground Truth')

    plt.legend(fontsize=21)
    plt.rcParams['font.size'] = 14

    # FOR MARMOUSI
    # tick_positions = np.arange(0, 2765, 395)
    # real_values = np.arange(0.5, 4, 0.5)
    # FOR OPENFWI
    tick_positions = np.arange(0, 80, 10)
    real_values_labels = [f"{i * 0.1:.1f}" for i in range(len(tick_positions))]
    plt.xticks(tick_positions, labels=real_values_labels, fontsize=18)

    # plt.xticks(range(0, 80, 10), fontsize=18)

    ticks = [np.min(vmin), 0.2*(vmax-vmin)+vmin, 0.4*(vmax-vmin)+vmin,
                0.6*(vmax-vmin)+vmin, 0.8*(vmax-vmin)+vmin, np.max(vmax)]
    tick_labels = [int(tick) for tick in ticks]
    plt.yticks(ticks, tick_labels, fontsize=18)
    # plt.yticks([np.min(vmin), 0.2*(vmax-vmin)+vmin, 0.4*(vmax-vmin)+vmin,
    #             0.6*(vmax-vmin)+vmin, 0.8*(vmax-vmin)+vmin, np.max(vmax)], fontsize=18)
    plt.xlabel('Depth (km)', font21)
    plt.ylabel('Velocity (m/s)', font21)
    plt.subplots_adjust(bottom=0.13, top=0.97, left=0.13, right=0.97)
    plt.savefig('PDD' + str(num) + '_' + str(column_index) + '.png', dpi=600)
    plt.close(fig)



dir = "E:/Code/DST_FWI_4.0/test_result/marmousi_70_70/"

InversionNet_DIR = "InversionNet_data.npz"
VelocityGAN_DIR = "VelocityGAN_data.npz"
DDNet_DIR = "DDNet70Model_data.npz"
ABA_DIR = "ABA_FWI_data.npz"
TDMF_DIR = "TDMF_data.npz"

InversionNet_MAT = np.load(InversionNet_DIR)['pd']
VelocityGAN_MAT = np.load(VelocityGAN_DIR)['pd']
DDNet_MAT = np.load(DDNet_DIR)['pd']
ABA_MAT = np.load(ABA_DIR)['pd']
TDMF_MAT = np.load(TDMF_DIR)['pd']

gt = np.load(InversionNet_DIR)['gt']
vmax = np.load(InversionNet_DIR)['vmax']
vmin = np.load(InversionNet_DIR)['vmin']

print('ok')

for i in np.arange(200, 400, 10):
    plot_velocity_image(i, InversionNet_MAT, VelocityGAN_MAT, DDNet_MAT, ABA_MAT, TDMF_MAT, gt,  vmin, vmax)


# dir = "E:/Code/DST_FWI_base_5.0/test_result/CurveVelB/"
# InversionNet_DIR = dir + "InversionNet_TrainSize24000_Epoch160_BatchSize20_LR0.0001_TestResults.mat"
# VelocityGAN_DIR = dir +"VelocityGAN_TrainSize24000_Epoch160_BatchSize20_LR0.0001_TestResults.mat"
# DDNet_DIR = dir +"DD-Net70_TrainSize24000_Epoch160_BatchSize20_LR0.0001_TestResults.mat"
# ABA_DIR = dir +"ABA_FWI_TrainSize24000_Epoch160_BatchSize20_LR0.0001_TestResults.mat"
# TDMF_DIR = dir +"DST-FWI_TrainSize24000_Epoch160_BatchSize20_LR0.0001_TestResults.mat"
# # CurveFaultB [44]10
# # CurveFaultA [85]20
# # CurveVelB [80]30
# # CurveVelA [91]5
# # i = 240 for marmousi
# i=80
# InversionNet_MAT = scipy.io.loadmat(InversionNet_DIR)['Prediction'][i,:]
# VelocityGAN_MAT = scipy.io.loadmat(VelocityGAN_DIR)['Prediction'][i,:]
# DDNet_MAT = scipy.io.loadmat(DDNet_DIR)['Prediction'][i,:]
# ABA_MAT = scipy.io.loadmat(ABA_DIR)['Prediction'][i,:]
# TDMF_MAT = scipy.io.loadmat(TDMF_DIR)['Prediction'][i,:]
# gt = scipy.io.loadmat(InversionNet_DIR)['GT'][i,:]
# vmax = np.max(gt)
# vmin = np.min(gt)
#
#
#
# for j in np.arange(25, 40, 5):
#     plot_velocity_image(j, i, InversionNet_MAT, VelocityGAN_MAT, DDNet_MAT, ABA_MAT, TDMF_MAT, gt,  vmin, vmax)
# 读数据
