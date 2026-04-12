import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# Save the loss
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 17,
         }
font3 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 21,
         }
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 2,
         }


path_loss_InversionNet = './result/loss/ABA-InversionNet_TrainSize24000_Epoch200_BatchSize20_LR0.001TrainValidLoss.mat'
loss_InversionNet = loadmat(path_loss_InversionNet)
train_loss_boundary_InversionNet = loss_InversionNet['train_loss_boundary']
val_loss_boundary_InversionNet = loss_InversionNet['val_loss_boundary']
train_loss_pixel_InversionNet = loss_InversionNet['train_loss_pixel']
val_loss_pixel_InversionNet = loss_InversionNet['val_loss_pixel']
train_loss_InversionNet = loss_InversionNet['train_loss']
val_loss_InversionNet = loss_InversionNet['val_loss']

# path_loss_InversionNet_SAD = './result/loss/ABA-InversionNet_TrainSize48000_Epoch200_BatchSize64_LR0.0001TrainValidLoss.mat'
# loss_InversionNet_SAD = loadmat(path_loss_InversionNet_SAD)
# train_loss_InversionNet_SAD = loss_InversionNet_SAD['train_loss']
# val_loss_InversionNet_SAD = loss_InversionNet_SAD['val_loss']

# path_loss_InversionNet_RCTBL = './result/loss/InversionNet_tv_1p_edge_ref_w_TrainSize48000_Epoch200_BatchSize20_LR0.001TrainValidLoss.mat'
# loss_InversionNet_RCTBL = loadmat(path_loss_InversionNet_RCTBL)
# train_loss_InversionNet_RCTBL = loss_InversionNet_RCTBL['train_loss']
# val_loss_InversionNet_RCTBL = loss_InversionNet_RCTBL['val_loss']
#
# path_loss_InversionNet_ABA = './result/loss/InversionNet_tv_1p_edge_ref_w_TrainSize48000_Epoch200_BatchSize20_LR0.001TrainValidLoss.mat'
# loss_InversionNet_ABA = loadmat(path_loss_InversionNet_ABA)
# train_loss_InversionNet_ABA = loss_InversionNet_ABA['train_loss']
# val_loss_InversionNet_ABA = loss_InversionNet_ABA['val_loss']

fig, ax = plt.subplots()
plt.plot(train_loss_InversionNet[0][1:], color='red', linestyle='-', label=r'$L_{all}$  train ', linewidth=0.5)
plt.plot(val_loss_InversionNet[0][1:], color='red', linestyle='-.', label=r'$L_{all}$ val', linewidth=0.5)  # 绘制 y1，并添加标签
plt.plot(train_loss_pixel_InversionNet[0][1:], color='blue', linestyle='-', label=r'$L_{pixel}$  train ', linewidth=0.5)
plt.plot(val_loss_pixel_InversionNet[0][1:], color='blue', linestyle='-.', label=r'$L_{pixel}$ val', linewidth=0.5)  # 绘制 y1，并添加标签
plt.plot(train_loss_boundary_InversionNet[0][1:], color='green', linestyle='-', label=r'$L_{boundary}$  train ', linewidth=0.5)
plt.plot(val_loss_boundary_InversionNet[0][1:], color='green', linestyle='-.', label=r'$L_{boundary}$ val', linewidth=0.5)  # 绘制 y1，并添加标签
# plt.plot(train_loss_InversionNet_SAD[0][1:], color='blue', linestyle='-', label=r'$L_{InversionNet-SAD}$ train', linewidth=0.5)
# plt.plot(val_loss_InversionNet_SAD[0][1:], color='blue', linestyle='-.', label=r'$L_{InversionNet-SAD}$ val', linewidth=0.5)  # 绘制 y1，并添加标签
# plt.plot(train_loss_InversionNet_RCTBL[0][1:], label='Training')
# plt.plot(val_loss_InversionNet_RCTBL[0][1:], label='Validation')  # 绘制 y1，并添加标签
# plt.plot(train_loss_InversionNet_ABA[0][1:], label='Training')
# plt.plot(val_loss_InversionNet_ABA[0][1:], label='Validation')  # 绘制 y1，并添加标签
plt.legend(loc='upper right', fontsize=10, bbox_to_anchor=(1, 1), ncol=3)
ax.set_xlabel('Num. of epochs', font2)
ax.set_ylabel('Loss', font2)
ax.set_title('Training and validation Loss', font3)
ax.set_xlim([1, 10])
ax.set_xticks([i for i in range(0, 201, 20)])
ax.set_xticklabels((str(i) for i in range(0, 201, 20)))
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(12)
ax.grid(linestyle='dashed', linewidth=0.5)

plt.show()
plt.close()