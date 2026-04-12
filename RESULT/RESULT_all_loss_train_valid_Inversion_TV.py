# -*- coding: utf-8 -*-
"""
Created on 2023/10/17 10:13

@author: XUQIONG  (xuqiong@swpu.edu.cn)

"""

################################################
########            导入库               ########
################################################
import os
import torch
import time
from torch.utils.data import DataLoader
from model.InversionNet import *
from model.FCNVMB import *
from model.IAEDN import *
from PathConfig import *
from data.data import *
from utils import *

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

################################################
########             NETWORK            ########
################################################

# Here indicating the GPU you want to use. if you don't have GPU, just leave it.
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

net = IAEDN() ##FCNVMB
net = net.to(device)


# Optimizer we want to use
optimizer = torch.optim.Adam(net.parameters(), lr=LearnRate)

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


# 输出当前工作目录
dataset_dir = Data_path

trainSet = Dataset_openfwi4(dataset_dir, TrainSize, 1, "seismic", "train") # TrainSize

train_loader = DataLoader(trainSet, batch_size=BatchSize, shuffle=True)

valSet = Dataset_openfwi4(dataset_dir, ValSize, 1, "seismic", "test") # TrainSize

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
    total_loss_pixel = 0
    total_loss_boundary = 0
    total_loss = 0
    # 将模型设置为训练模式
    net.train()
    # 遍历数据集

    for i, (seismic_datas, vmodels, edges) in enumerate(train_loader):
        # Set Net with train condition

        seismic_datas = seismic_datas[0].to(device)
        vmodels = vmodels[0].to(device)
        edges = edges[0].to(device)


        # Zero the gradient buffer
        optimizer.zero_grad()

        # seismic_datas = seismic_datas.reshape(BatchSize, InChannel, DataDim[0], DataDim[1])
        # vmodels = vmodels.reshape(BatchSize, OutChannel, ModelDim[0], ModelDim[1])
        # Forward prediction
        outputs = net(seismic_datas)

        # 转换参数的输入类型为 float
        outputs = outputs.to(torch.float32)
        vmodels = vmodels.to(torch.float32)
        edges = edges.to(torch.float32)

        # Calculate the MSE
        l1loss = nn.L1Loss()
        l2loss = nn.MSELoss()

        loss_tv = loss_tv_1p_edge_ref_w(outputs, vmodels, edges)
        loss_g1v = l1loss(outputs, vmodels)  # l1loss(outputs, vmodels)   l2loss(outputs, vmodels)
        loss_g2v = l2loss(outputs, vmodels)

        loss_pixel = loss_g1v + loss_g2v
        loss_boundary = 10 * loss_tv
        loss = loss_pixel + loss_boundary

        if np.isnan(float(loss.item())):
            raise ValueError('loss is nan while training')

        total_loss += loss.item()
        total_loss_pixel += loss_pixel.item()
        total_loss_boundary += loss_boundary.item()
        # Loss backward propagation
        loss = loss.to(torch.float32)
        loss.backward()

        # Optimize
        optimizer.step()

    # 计算平均损失
    avg_loss = total_loss / len(train_loader)
    avg_loss_pixel = total_loss_pixel / len(train_loader)
    total_loss_boundary = total_loss_boundary / len(train_loader)
    return avg_loss, avg_loss_pixel, total_loss_boundary

def validate():
    total_loss_pixel = 0
    total_loss_boundary = 0
    total_loss = 0
    # 将模型设置为验证模式
    net.eval()

    with torch.no_grad():
        # 遍历验证集
        for i, (seismic_datas, vmodels, edges) in enumerate(val_loader):
            # 将数据移到设备上
            seismic_datas = seismic_datas[0].to(device)
            vmodels = vmodels[0].to(device)
            edges = edges[0].to(device)


            outputs = net(seismic_datas)

            # 转换参数的输入类型为 float
            outputs = outputs.to(torch.float32)
            vmodels = vmodels.to(torch.float32)
            edges = edges.to(torch.float32)

            # Calculate the MSE
            l1loss = nn.L1Loss()
            l2loss = nn.MSELoss()

            loss_tv = loss_tv_1p_edge_ref_w(outputs, vmodels, edges)
            loss_g1v = l1loss(outputs, vmodels)  # l1loss(outputs, vmodels)   l2loss(outputs, vmodels)
            loss_g2v = l2loss(outputs, vmodels)

            loss_pixel = loss_g1v + loss_g2v
            loss_boundary = loss_tv
            loss = loss_pixel + loss_boundary

            total_loss += loss.item()
            total_loss_pixel += loss_pixel.item()
            total_loss_boundary += loss_boundary.item()
    # 计算平均损失
    avg_loss = total_loss / len(val_loader)
    avg_loss_pixel = total_loss_pixel / len(val_loader)
    total_loss_boundary = total_loss_boundary / len(val_loader)
    return avg_loss, avg_loss_pixel, total_loss_boundary

train_loss_list = 0
train_loss_pixel_list = 0
train_loss_boundary_list = 0
val_loss_list = 0
val_loss_pixel_list = 0
val_loss_boundary_list = 0

for epoch in range(Epochs):
    epoch_loss = 0.0
    since = time.time()
    train_loss, train_loss_pixel, train_loss_boundary = train()
    val_loss, val_loss_pixel, val_loss_boundary = validate()
    # 显示训练集和验证集的损失
    if (epoch % 1) == 0:
        print(f"Epoch: {epoch + 1}, Train loss:{train_loss:.4f},Val loss: {val_loss: .4f}")
        print(f"Epoch: {epoch + 1}, Train loss:{train_loss_pixel:.4f},Val loss: {val_loss_pixel: .4f}")
        print(f"Epoch: {epoch + 1}, Train loss:{train_loss_boundary:.4f},Val loss: {val_loss_boundary: .4f}")

        time_elapsed = time.time() - since
        print('Epoch consuming time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # Save net parameters every 10 epochs
    if (epoch + 1) % SaveEpoch == 0:
        torch.save(net.state_dict(), train_result_dir + ModelName + '_epoch' + str(epoch + 1) + '.pkl')
        print('Trained model saved: %d percent completed' % int((epoch + 1) * 100 / Epochs))

    train_loss_list = np.append(train_loss_list, train_loss)
    train_loss_pixel_list = np.append(train_loss_pixel_list, train_loss_pixel)
    train_loss_boundary_list = np.append(train_loss_boundary_list, train_loss_boundary)
    val_loss_list = np.append(val_loss_list, val_loss)
    val_loss_pixel_list = np.append(val_loss_pixel_list, val_loss_pixel)
    val_loss_boundary_list = np.append(val_loss_boundary_list, val_loss_boundary)

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

SaveTrainValidAllResults(train_loss=train_loss_list, train_loss_pixel=train_loss_pixel_list, train_loss_boundary=train_loss_boundary_list,
                             val_loss=val_loss_list, val_loss_pixel=val_loss_pixel_list, val_loss_boundary=val_loss_boundary_list, SavePath=train_result_dir, ModelName=ModelName, font2=font2, font3=font3)





