# -*- coding: utf-8 -*-
"""
Created on 2025/6/17 10:21

@author: XUQIONG  (xuqiong@swpu.edu.cn)

"""

####################################################
####             MAIN PARAMETERS                ####
####################################################
OutChannel = 1  # 输出速度模型的通道数
ReUse = False  # If False always re-train a network  False  True
dh = 10  # Space interval

####################################################
####             NETWORK PARAMETERS             ####
####################################################
Dataset_name = 'CurveVelB'   # CurveFaultA|FlatFaultA|FlatVelA|CurveFaultA|marmousi_70_70|marmousi_100_100|CurveVelA
InChannel = 5  # 输入的地震数据炮数  # 328  30926
SaveEpoch = 5
DisplayStep = 600  # Number of steps till outputting stats

####################################################
####            DATASET PARAMETERS              ####
####################################################

if Dataset_name  == 'SEGSimulation':
    data_dim = [400, 301]                           # Dimension of original one-shot seismic data
    model_dim = [201, 301]                          # Dimension of one velocity model
    inchannels = 29                                 # Number of input channels
    train_size = 1600                               # Number of training sets
    test_size = 100                                 # Number of testing sets

    DataDim = [400, 301]
    ModelDim = [201, 301]
    InChannel = 29
    TrainSize = 1600
    TestSize = 100

    BatchSize = 10
    TestBatchSize = 5
    LearnRate = 1e-4
    Epochs = 200

elif Dataset_name == 'FlatVelA':
    DataDim = [1000, 70]
    ModelDim = [70, 70]
    InChannel = 5
    TrainSize = 24000
    TestSize = 6000
    ValSize = 1000

    BatchSize = 20
    TestBatchSize = 5
    LearnRate = 1e-4
    Epochs = 200

elif Dataset_name == 'CurveVelA':
    DataDim = [1000, 70]
    ModelDim = [70, 70]
    InChannel = 5
    TrainSize = 24000
    TestSize = 6000
    ValSize = 1000

    BatchSize = 20
    TestBatchSize = 5
    LearnRate = 1e-4
    Epochs = 200

elif Dataset_name == 'FlatFaultA':
    DataDim = [1000, 70]
    ModelDim = [70, 70]
    InChannel = 5
    TrainSize = 48000
    TestSize = 6000
    ValSize = 1000

    BatchSize = 20
    TestBatchSize = 5
    LearnRate = 1e-4
    Epochs = 200

elif Dataset_name == 'FlatFaultB':
    DataDim = [1000, 70]
    ModelDim = [70, 70]
    InChannel = 5
    TrainSize = 48000
    TestSize = 6000
    ValSize = 0

    BatchSize = 20
    TestBatchSize = 5
    LearnRate = 1e-4
    Epochs = 200

elif Dataset_name == 'CurveFaultA':
    DataDim = [1000, 70]
    ModelDim = [70, 70]
    InChannel = 5
    TrainSize = 48000
    TestSize = 6000
    ValSize = 1000

    BatchSize = 20
    TestBatchSize = 5
    LearnRate = 1e-4
    Epochs = 115


elif Dataset_name == 'CurveFaultB':
    DataDim = [1000, 70]
    ModelDim = [70, 70]
    InChannel = 5
    TrainSize = 48000
    TestSize = 5000
    ValSize = 500

    BatchSize = 20
    TestBatchSize = 5
    LearnRate = 1e-4
    Epochs = 200

elif Dataset_name == 'CurveVelB':
    DataDim = [1000, 70]
    ModelDim = [70, 70]
    InChannel = 5
    TrainSize = 24000
    TestSize = 6000
    ValSize = 500

    BatchSize = 20
    TestBatchSize = 5
    LearnRate = 1e-4
    Epochs = 200

elif Dataset_name == 'marmousi_70_70':
    DataDim = [1000, 70]
    ModelDim = [70, 70]
    InChannel = 5
    TrainSize = 30926
    TestSize = 328
    ValSize = 328

    BatchSize = 20
    TestBatchSize = 5
    LearnRate = 1e-4
    Epochs = 200

elif Dataset_name == 'marmousi_100_100':
    DataDim = [1500, 100]
    ModelDim = [100, 100]
    InChannel = 7
    TrainSize = 17675 # 17675
    TestSize = 160
    ValSize = 100

    BatchSize = 40
    TestBatchSize = 5
    LearnRate = 1e-4
    Epochs = 200

else:
    print('The selected dataset is invalid')
    exit(0)

