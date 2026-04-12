import matplotlib as mpl
import numpy as np
import cv2

mpl.use('TkAgg')
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
font21 = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': 21,
}

font18 = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': 18,
}

def pain_openfwi_three_velocity_model(*para_velocity_models):
    fig, axs = plt.subplots(1, 4, figsize=(40, 3), dpi=150) #, sharey=True
    vmin = min(np.min(model) for model in para_velocity_models)
    vmax = max(np.max(model) for model in para_velocity_models)

    for i, ax in enumerate(axs):
        if i < len(para_velocity_models):
            im = ax.imshow(para_velocity_models[i], extent=[0, 0.7, 0.7, 0], vmin=vmin, vmax=vmax)
            # ax.set_xlabel('Position (km)', font18)
            if i == 0:
                ax.set_ylabel('Depth (km)', font18)

            ax.set_xticks(np.linspace(0, 0.7, 8))
            ax.set_yticks(np.linspace(0, 0.7, 8))

            ax.set_xticklabels(labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], size=10)
            ax.set_yticklabels(labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], size=10)

    ticks = np.linspace(vmin, vmax, num=7)
    cax = fig.add_axes([0.92, 0.1, 0.01, 0.75])

    fig.colorbar(im, cax=cax, ticks=ticks)
    cax.tick_params(labelsize=10)

    plt.subplots_adjust(left=0.06, bottom=0.06, right=0.88, top=0.88,
                        wspace=0.225, hspace=0)
    plt.show()



# CurveVelA  FlatFaultA  FlatVelA   CurveFaultA`
PATH_DIR = 'C:/Users/DELL/Desktop/OpenFWI_RESULTS/'
DataSet = 'FlatVelA/'
path_InversionNet = PATH_DIR + DataSet + 'InversionNet_TestResults.mat'
path_VelocityGAN = PATH_DIR + DataSet + 'VelocityGAN_TestResults.mat'
path_IAEDN_WTUU = PATH_DIR + DataSet + 'IAEDN_WTUU_TestResults.mat'

'''
path_InversionNet = './result/model/FlatVelA_ABA-Net_TestResults.mat'
path_VelocityGAN = './result/model/FlatVelA_ABA-Loss_TestResults.mat'
path_ABA_FWI = './result/model/FlatVelA_ABA-FWI_TestResults.mat'
'''

model_InversionNet = loadmat(path_InversionNet)
model_VelocityGAN = loadmat(path_VelocityGAN)
model_ABA_FWI = loadmat(path_IAEDN_WTUU)
gts = model_InversionNet['GT_N']
pds_InversionNet = model_InversionNet['Prediction_N']
pds_VelocityGAN = model_VelocityGAN['Prediction_N']
pds_ABA_FWI = model_ABA_FWI['Prediction_N']
for i in range(4):
    x = gts[i, ...]
    y = pds_InversionNet[i, ...]
    z = pds_VelocityGAN[i, ...]
    k = pds_ABA_FWI[i, ...]
    pain_openfwi_three_velocity_model(gts[i, ...],
                                      pds_InversionNet[i, ...],
                                      pds_VelocityGAN[i, ...],
                                      pds_ABA_FWI[i, ...])
    # pain_openfwi_three_velocity_model(gts[i, ...],
    #                                   pds_InversionNet[i, ...] - gts[i, ...],
    #                                   pds_VelocityGAN[i, ...] - gts[i, ...],
    #                                   pds_ABA_FWI[i, ...] - gts[i, ...])
