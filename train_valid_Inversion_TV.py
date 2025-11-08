# -*- coding: utf-8 -*-
"""
Created on 2025/10/17 10:13

@author: XUQIONG  (xuqiong@swpu.edu.cn)

"""

################################################
########            导入库               ########
################################################
import os
import torch
import time
from torch.utils.data import DataLoader

from net.InversionNet import *
from net.TDMF_FWI import *
from net.ABA_FWI import *
from PathConfig import *
from data.data import *
from data.loss import *
from utils import *

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

################################################
########             NETWORK            ########
################################################

# Here indicating the GPU you want to use. if you don't have GPU, just leave it.
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

net = TDMF_FWI()  # ABA_FWI
net = net.to(device)

# Optimizer we want to use
optimizer = torch.optim.AdamW(net.parameters(), lr=LearnRate)

# If ReUse, it will load saved model from premodelfilepath and continue to train
if ReUse:
    print('***************** 加载预先训练的模型 *****************')
    print('')
    premodel_file = train_result_dir + PreModelname
    net.load_state_dict(torch.load(premodel_file))
    net = net.to(device)
    print('Finish downloading:', str(premodel_file))

################################################
########    LOADING TRAINING DATA       ########
################################################
print('***************** 正在加载训练数据集 *****************')

dataset_dir = Data_path

trainSet = Dataset_openfwi4(dataset_dir, TrainSize, 1, "seismic", "train")
train_loader = DataLoader(trainSet, batch_size=BatchSize, shuffle=True)

valSet = Dataset_openfwi4(dataset_dir, ValSize, 1, "seismic", "test")
val_loader = DataLoader(valSet, batch_size=BatchSize, shuffle=True)

################################################
########            TRAINING            ########
################################################

print()
print('*******************************************')
print('*******************************************')
print('                开始训练                    ')
print('*******************************************')
print('*******************************************')
print()

print('原始地震数据尺寸:%s' % str(DataDim))
print('原始速度模型尺寸:%s' % str(ModelDim))
print('培训规模:%d' % int(TrainSize))
print('培训批次大小:%d' % int(BatchSize))
print('迭代轮数:%d' % int(Epochs))
print('学习率:%.5f' % float(LearnRate))

# Initialization
loss1 = 0.0
step = int(TrainSize / BatchSize)
start = time.time()


def train():
    total_loss = 0
    net.train()

    for i, (seismic_datas, vmodels, edges) in enumerate(train_loader):

        seismic_datas = seismic_datas[0].to(device)
        vmodels = vmodels[0].to(device)
        edges = edges[0].to(device)

        # Zero the gradient buffer
        optimizer.zero_grad()

        if NoiseFlag:
            # 添加高斯噪声
            noise_mean = 0
            noise_std = 0.1
            noise = torch.normal(mean=noise_mean, std=noise_std, size=seismic_datas.shape).to(device)
            seismic_datas = seismic_datas + noise

        outputs = net(seismic_datas)

        outputs = outputs.to(torch.float32)
        vmodels = vmodels.to(torch.float32)
        edges = edges.to(torch.float32)

        loss_g1v = l1loss(outputs, vmodels)
        loss_g2v = l2loss(outputs, vmodels)
        loss_tv = loss_tv1(outputs, vmodels, edges)

        loss = loss_g1v + loss_g2v + loss_tv  # RCTB loss
        # loss = loss_g1v + loss_g2v

        if np.isnan(float(loss.item())):
            raise ValueError('loss is nan while training')

        total_loss += loss.item()

        loss = loss.to(torch.float32)
        loss.backward()

        # Optimize
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    print(len(train_loader))
    print(total_loss)
    print(avg_loss)
    return avg_loss


def validate():
    total_loss = 0
    net.eval()

    with torch.no_grad():
        for i, (seismic_datas, vmodels, edges) in enumerate(val_loader):
            seismic_datas = seismic_datas[0].to(device)
            vmodels = vmodels[0].to(device)
            edges = edges[0].to(device)

            outputs = net(seismic_datas)

            outputs = outputs.to(torch.float32)
            vmodels = vmodels.to(torch.float32)
            edges = edges.to(torch.float32)

            loss_tv = loss_tv1(outputs, vmodels, edges)
            loss_g1v = l1loss(outputs, vmodels)
            loss_g2v = l2loss(outputs, vmodels)

            loss = loss_g1v + loss_g2v + loss_tv    # RCTB loss

            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    return avg_loss


train_loss_list = 0
val_loss_list = 0

for epoch in range(Epochs):
    epoch_loss = 0.0
    since = time.time()

    train_loss = train()
    # val_loss = validate()

    if (epoch % 1) == 0:
        # print(f"Epoch: {epoch + 1}, Train loss:{train_loss:.4f},Val loss: {val_loss: .4f}")
        print(f"Epoch: {epoch + 1}, Train loss:{train_loss:.4f}")
        time_elapsed = time.time() - since
        print('Epoch consuming time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # Save net parameters every 10 epochs
    if (epoch + 1) % SaveEpoch == 0:
        torch.save(net.state_dict(), train_result_dir + ModelName + '_epoch' + str(epoch + 1) + '.pkl')
        print('Trained model saved: %d percent completed' % int((epoch + 1) * 100 / Epochs))

    train_loss_list = np.append(train_loss_list, train_loss)
    # val_loss_list = np.append(val_loss_list, val_loss)

# Record the consuming time
time_elapsed = time.time() - start
print('Training complete in {:.0f}m  {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

# Save the loss
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 17,
         }
font3 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 21,
         }

# SaveTrainValidResults(train_loss=train_loss_list, val_loss=val_loss_list, SavePath=train_result_dir, ModelName=ModelName, font2=font2, font3=font3)
SaveTrainResults(loss=train_loss_list, SavePath=train_result_dir, ModelName=ModelName, font2=font2, font3=font3)




