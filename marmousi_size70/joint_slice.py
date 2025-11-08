import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.io
from data.show import *
from data.data import *

font18 = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': 18,
}
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


def pain_marmousi_velocity_model_test(para_velocity_model, min_velocity, max_velocity, is_colorbar=1):
    '''
    Plotting seismic data images of openfwi dataset

    :param para_velocity_model: Velocity model (70 x 70) (numpy)
    :param min_velocity:        Upper limit of velocity in the velocity model
    :param max_velocity:        Lower limit of velocity in the velocity model
    :param is_colorbar:         Whether to add a color bar (1 means add, 0 is the default, means don't add)
    :return:
    '''


    fig, ax = plt.subplots(figsize=(3, 5), dpi=150)

    im = ax.imshow(para_velocity_model, vmin=min_velocity, vmax=max_velocity, cmap='jet')
    ax.axis('off')
    ax.set_title('TDMF-FWI', font18)

    plt.rcParams['font.size'] = 14
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.35)
    plt.colorbar(im, ax=ax, cax=cax, orientation='vertical',
                     ticks=np.linspace(min_velocity, max_velocity, 7), format=mpl.ticker.StrMethodFormatter('{x:.0f}'))
    # plt.subplots_adjust(bottom=0.10, top=0.95, left=0.13, right=0.95)

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
    model_size = 70
    step_size = 10   # 重叠切片步长

    # Load true model
    vmodel_dim = np.array([nz, ny])
    data_path = data_dir + data_name
    model_true = loadtruemodel(data_path, num_dims, vmodel_dim).cpu().numpy()  # 2801*13601

    # 确定训练集和测试集范围
    matrix = np.zeros((nz, ny))
    matrix[420:, 7350:8050] = 1    # 去除水之后，选定的某一列区域
    model_test_gt = model_true[420:, 7350:8050]

    model_test_pd = np.copy(model_test_gt)


    model_pd_path = "E:/Code/DST_FWI_3.0/test_result/marmousi_70_70/DST_FWI_TrainSize30926_Epoch200_BatchSize20_LR0.0001_TestResults.mat"
    # IAEDN_WTUU_TrainSize30926_Epoch160_BatchSize20_LR0.0001_TestResults.mat
    # DD-Net70_TrainSize30926_Epoch160_BatchSize20_LR0.0001_TestResults.mat
    # InversionNet_TrainSize30926_Epoch160_BatchSize20_LR0.0001_TestResults.mat
    # VelocityGAN_TrainSize30926_Epoch160_BatchSize20_LR0.0001_TestResults.mat
    model_pd_mat = scipy.io.loadmat(model_pd_path)

    # 读取矩阵数据, 补充缺失部分
    # 插入全0矩阵到缺失位置
    model_pd = model_pd_mat['Prediction']
    missing_indices = [123, 129, 203, 204, 205, 206, 207, 208, 209, 231, 232, 233]
    # [122, 128, 202, 203, 204, 205, 206, 207, 208, 230, 231, 232]
    # [123, 129, 203, 204, 205, 206, 207, 208, 209, 231, 232, 233]    x
    zero_matrix = np.zeros((70, 70))
    model_pd_list = list(model_pd)
    for index in missing_indices:
        model_pd_list.insert(index, zero_matrix)

    # show model
    vmin = np.min(model_true)
    vmax = np.max(model_true)

    for idx, matrix in enumerate(model_pd_list):
        row_start = (idx // 10) * 70  # 确定矩阵的起始行
        col_start = (idx % 10) * 70  # 确定矩阵的起始列
        if not np.all(matrix == 0):
            model_test_pd[row_start:row_start + 70, col_start:col_start + 70] = np.copy(matrix)

    pain_marmousi_velocity_model_test(np.abs(model_test_gt-model_test_pd), 0, 1000)
    pain_marmousi_velocity_model_test(model_test_pd, vmin, vmax)

    fig = plt.figure(figsize=(0.7,5))   #3,5   1.5, 5

    ax1 = fig.add_subplot(1, 1, 1)
    ax1.imshow(np.abs(model_test_gt-model_test_pd), vmin=0, vmax=1000, cmap='jet')
    ax1.axis('off')
    ax1.set_title('InversionNet', font18)
    # ax2 = fig.add_subplot(1, 2, 2)
    # ax2.imshow(model_test_gt, vmin=vmin, vmax=vmax, cmap='jet')
    # ax2.axis('off')
    # ax2.set_title('GT', font18)
    #ax2.tick_params(axis='both', which='both', length=0, labelsize=0)

    plt.show()



