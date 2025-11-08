import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.io
from data.show import *
from data.data import *


def loadtruemodel(data_dir, num_dims, vmodel_dim):
    """
        Load the true model
    """

    if num_dims != len(vmodel_dim.reshape(-1)):
        raise Exception('Please check the size of model_true!!')
    # prefer the depth direction first, that is the shape is `[nz, (ny, (nx))]`
    if num_dims == 2:
        model_true = (np.fromfile(data_dir, np.float32).reshape(vmodel_dim[1], vmodel_dim[0]))
        model_true = np.transpose(model_true, (1, 0))  # I prefer having depth direction first
    else:

        raise Exception('Please check the size of model_true!!')

    model_true = torch.Tensor(model_true)  # Convert to a PyTorch Tensor

    return model_true


def model_slice(para_model, para_modelSize=70, para_stepSize=10, para_egdePixelNum=280):
    """
        Slice the true model.
    """
    temp_slice_result = []
    for i in range(0, para_model.shape[0] - para_modelSize + 1, para_stepSize):
        for j in range(0, para_model.shape[1] - para_modelSize + 1, para_stepSize):
            temp_slice_result.append(para_model[i:i + para_modelSize, j:j + para_modelSize])

    temp_slice_arr = np.array(temp_slice_result)
    temp_edge_arr = np.zeros_like(temp_slice_result)

    # 初始化一个空列表来存储满足条件的二维矩阵
    matrix_3d_list = []
    for i in range(temp_edge_arr.shape[0]):
        temp_edge_arr[i, :, :] = extract_contours(temp_slice_arr[i, :, :])
        temp_num_ones = np.count_nonzero(temp_edge_arr[i, :, :] == 1)

        if temp_num_ones > para_egdePixelNum:
            matrix_2d_expanded = np.expand_dims(temp_slice_arr[i, :, :], axis=0)

            # 将满足条件的二维矩阵添加到列表中
            matrix_3d_list.append(matrix_2d_expanded)

    # 示例的三维矩阵初始化
    matrix_3d = np.empty((0, 70, 70))  # 这里设置为形状为 (0, 70, 70) 的空矩阵

    # 将列表转换为三维 NumPy 数组
    if matrix_3d_list:
        matrix_3d = np.concatenate(matrix_3d_list, axis=0)

    return matrix_3d

def save_npy(para_data, para_label, para_path):
    # 分割数组并保存
    chunk_size = 500
    num_chunks = (para_data.shape[0] + chunk_size - 1) // chunk_size  # 计算分割块数

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, para_data.shape[0])
        sub_array = para_data[start_idx:end_idx]
        filename = f"{para_label}{i + 1}.npy"
        np.save(para_path + filename, sub_array)


def save_mat(para_data, para_label, para_path):
    # 分割数组并保存
    chunk_size = 500
    num_chunks = (para_data.shape[0] + chunk_size - 1) // chunk_size  # 计算分割块数

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, para_data.shape[0])
        sub_array = para_data[start_idx:end_idx]
        filename = f"{para_label}{i + 1}.mat"
        data = {'vmodels': sub_array}
        scipy.io.savemat(para_path + filename, data)

font12 = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': 12,
}

def pain_marmousi_velocity_model(para_velocity_model, min_velocity, max_velocity, is_colorbar = 1):
    '''
    Plotting seismic data images of openfwi dataset

    :param para_velocity_model: Velocity model (70 x 70) (numpy)
    :param min_velocity:        Upper limit of velocity in the velocity model
    :param max_velocity:        Lower limit of velocity in the velocity model
    :param is_colorbar:         Whether to add a color bar (1 means add, 0 is the default, means don't add)
    :return:
    '''

    if is_colorbar == 0:
        fig, ax = plt.subplots(figsize=(10, 3.2), dpi=150)
    else:
        fig, ax = plt.subplots(figsize=(7, 4), dpi=150)

    im = ax.imshow(para_velocity_model, extent=[0, 17, 3.5, 0], vmin=min_velocity, vmax=max_velocity, cmap='jet')

    ax.set_xlabel('Position (km)', font12)
    ax.set_ylabel('Depth (km)', font12)
    ax.set_xticks(range(0, 18, 2))
    ax.set_xticks([17], minor=True)
    ax.set_yticks(np.linspace(0, 3.5, 8))
    ax.set_xticklabels(labels=[' ', 2, 4, 6, 8, 10, 12, 14, 16], size=8)
    ax.set_yticklabels(labels=[0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5], size=8)
    x_start, x_end = 7350, 8050
    y_start, y_end = 420, 280
    rect = patches.Rectangle((x_start, y_start), x_end-x_start, y_end-y_start, linewidth=10, edgecolor='r', facecolor='none')

    # 将矩形补丁添加到坐标轴上
    ax.add_patch(rect)


    if is_colorbar == 0:
        plt.subplots_adjust(bottom=0.11, top=0.95, left=0.11, right=0.95)
    else:
        plt.rcParams['font.size'] = 8      # Set colorbar font size
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="3%", pad=0.35)
        plt.colorbar(im, ax=ax, cax=cax, orientation='horizontal',
                     ticks=np.linspace(min_velocity, max_velocity, 7), format = mpl.ticker.StrMethodFormatter('{x:.0f}'))
        plt.subplots_adjust(bottom=0.10, top=0.95, left=0.13, right=0.95)

    plt.show()

if __name__ == '__main__':
    # mar_smal, _100_310  mar_big, _117_567 over
    # dataname  = 'mar_smal'
    # sizestr   = '_100_310'
    data_dir = 'F:/Data/Marmousi/'
    data_name = 'MODEL_P-WAVE_VELOCITY_1.25m.bin'
    nz = 2801
    ny = 13601
    num_dims = 2

    # 尺寸为70的设置
    # model_size = 70
    # step_size = 10   # 重叠切片步长

    # 尺寸为100的设置
    model_size = 100
    step_size = 10   # 重叠切片步长

    # Load true model
    vmodel_dim = np.array([nz, ny])
    data_path = data_dir + data_name
    model_true = loadtruemodel(data_path, num_dims, vmodel_dim).cpu().numpy()  # 2801*13601  2801*13601

    # 确定训练集和测试集范围
    matrix = np.zeros((nz, ny))
    matrix[420:, 7350:8050] = 1    # 去除水之后，选定的某一列区域
    model_train1 = model_true[420:, :7350]
    model_train2 = model_true[420:, 8050:]
    model_test = model_true[420:, 7350:8050]

    # show model
    vmin = np.min(model_true)
    vmax = np.max(model_true)
    pain_marmousi_velocity_model(model_true, vmin, vmax)
    # fig = plt.figure(figsize=(10, 6))
    #
    # ax1 = fig.add_subplot(2, 1, 1)
    # ax1.imshow(model_true, vmin=vmin, vmax=vmax, cmap='jet')
    # ax1.title.set_text('Marmousi')
    #
    # ax2 = fig.add_subplot(2, 1, 2)
    # ax2.imshow(model_true*matrix, vmin=vmin, vmax=vmax, cmap='jet')
    # ax2.title.set_text('Marmousi_test_area')
    #
    # plt.show()

    # 训练集结果
    # train1 = model_slice(model_train1, para_modelSize=70, para_stepSize=30, para_egdePixelNum=0)
    # train2 = model_slice(model_train2, para_modelSize=70, para_stepSize=30, para_egdePixelNum=0)
    # train_data = np.concatenate((train1, train2), axis=0)
    # train_data = np.expand_dims(train_data, axis=1)
    # test_data = model_slice(model_test, para_stepSize=70, para_egdePixelNum=0)   #124,130,204,205,206,207,208,209,210,232,233,234
    # test_data = np.expand_dims(test_data, axis=1)
    # print("ok")
    # 500个数据保存为一个npy
    # save_npy(train_data, 'vmodel', 'D:/Data/Marmousi/marmousi_70_70/train_data/vmodel/')
    # save_npy(test_data, 'vmodel', 'D:/Data/Marmousi/marmousi_70_70/test_data/vmodel/')

    # 500个数据保存为一个mat
    # save_mat(train_data, 'vmodel', 'D:/Data/Marmousi/marmousi_70_70/train_data/vmodel/')
    # save_mat(test_data, 'vmodel', 'D:/Data/Marmousi/marmousi_70_70/test_data/vmodel/')

