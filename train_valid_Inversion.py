# -*- coding: utf-8 -*-
"""
Created on 2023/10/17 10:13

@author: XUQIONG  (xuqiong@swpu.edu.cn)

"""

################################################
########            导入库               ########
################################################
import time
from net.ABA_FWI import *
from data.data import *
from data.loss import *
from net.InversionNet import *
from net.DDNet70 import *
from utils import *

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

################################################
########             NETWORK            ########
################################################

# Here indicating the GPU you want to use. if you don't have GPU, just leave it.
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

net = DDNet70Model()  # InversionNet | ABA_Net | DDNet70Model
net = net.to(device)

# Optimizer we want to use
optimizer = torch.optim.Adam(net.parameters(), lr=LearnRate)

# If ReUse, it will load saved model from premodelfilepath and continue to train
if ReUse:
    print('***************** Loading pre-training model *****************')
    print('')
    premodel_file = train_result_dir + PreModelname
    net.load_state_dict(torch.load(premodel_file))
    net = net.to(device)
    print('Finish downloading:', str(premodel_file))

################################################
########    LOADING TRAINING DATA       ########
################################################
print('***************** Loading training dataset *****************')

dataset_dir = Data_path

trainSet = Dataset_openfwi(dataset_dir, TrainSize, 1, "seismic", "train")
train_loader = DataLoader(trainSet, batch_size=BatchSize, shuffle=True)

valSet = Dataset_openfwi(dataset_dir, ValSize, 1, "seismic", "test")
val_loader = DataLoader(valSet, batch_size=BatchSize, shuffle=True)

################################################
########            TRAINING            ########
################################################

print()
print('*******************************************')
print('*******************************************')
print('                Training ...               ')
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

    for i, (seismic_datas, vmodels) in enumerate(train_loader):
        net.train()

        seismic_datas = seismic_datas[0].to(device)
        vmodels = vmodels[0].to(device)

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
        models = vmodels.to(torch.float32)

        loss, loss_g1v, loss_g2v = criterion(outputs, vmodels)   # L1 and L2 loss

        if np.isnan(float(loss.item())):
            raise ValueError('loss is nan while training')

        total_loss += loss.item()
        # Loss backward propagation
        loss = loss.to(torch.float32)
        loss.backward()

        # Optimize
        optimizer.step()

    # 计算平均损失
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate():
    total_loss = 0
    net.eval()

    with torch.no_grad():
        for i, (seismic_datas, vmodels) in enumerate(val_loader):

            seismic_datas = seismic_datas[0].to(device)
            vmodels = vmodels[0].to(device)
            outputs = net(seismic_datas)

            outputs = outputs.to(torch.float32)
            vmodels = vmodels.to(torch.float32)

            loss, loss_g1v, loss_g2v = criterion(outputs, vmodels)   # L1 and L2 loss

            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    return avg_loss


train_loss_list = 0
val_loss_list = 0

for epoch in range(Epochs):
    epoch_loss = 0.0
    since = time.time()

    train_loss = train()
    val_loss = validate()

    # Show train and val loss every 1 epoch
    if (epoch % 1) == 0:
        print(f"Epoch: {epoch + 1}, Train loss:{train_loss:.4f},Val loss: {val_loss: .4f}")
        time_elapsed = time.time() - since
        print('Epoch consuming time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # Save net parameters every 10 epochs
    if (epoch + 1) % SaveEpoch == 0:
        torch.save(net.state_dict(), train_result_dir + ModelName + '_epoch' + str(epoch + 1) + '.pkl')
        print('Trained model saved: %d percent completed' % int((epoch + 1) * 100 / Epochs))

    train_loss_list = np.append(train_loss_list, train_loss)
    val_loss_list = np.append(val_loss_list, val_loss)

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

SaveTrainValidResults(train_loss=train_loss_list, val_loss=val_loss_list, SavePath=train_result_dir, ModelName=ModelName, font2=font2, font3=font3)





