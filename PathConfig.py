# -*- coding: utf-8 -*-
"""
Created on 2025/10/17 11:01

@author: XUQIONG  (xuqiong@swpu.edu.cn)

"""
import os
from ParamConfig import *

###################################################
####                DATA   PATHS              #####
###################################################
Data_dir = 'F:/Data/OpenFWI/'
DataSet = Dataset_name + '/'
Data_path = Data_dir + DataSet

###################################################
####            RESULT   PATHS                #####
###################################################
main_dir = 'E:/Code/DST_FWI_base_5.0/'  # 当前工作目录

# Check the main directory
if len(main_dir) == 0:
    raise Exception('Please specify path to correct directory!')

# Save training result
if os.path.exists('train_result/' + DataSet):
    train_result_dir = main_dir + 'train_result/' + DataSet  # Replace your data path here
else:
    os.makedirs('train_result/' + DataSet)
    train_result_dir = main_dir + 'train_result/' + DataSet

# Save testing result
if os.path.exists('test_result/' + DataSet):
    test_result_dir = main_dir + 'test_result/' + DataSet  # Replace your data path here
else:
    os.makedirs('test_result/' + DataSet)
    test_result_dir = main_dir + 'test_result/' + DataSet

####################################################
####                   FileName                #####
####################################################

NoiseFlag = False  # | False | True  # If True add noise.
modelName = 'TDMF-FWI'
# VelocityGAN | InversionNet | DD-Net70 | ABA-FWI | DST-FWI | TDMF-FWI

tagM1 = '_TrainSize' + str(TrainSize)
tagM2 = '_Epoch' + str(Epochs)
tagM3 = '_BatchSize' + str(BatchSize)
tagM4 = '_LR' + str(LearnRate)

ModelName = modelName + tagM1 + tagM2 + tagM3 + tagM4

# Load pre-trained model
PreModelname = ''

