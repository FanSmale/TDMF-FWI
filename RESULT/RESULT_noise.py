
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import cv2

mpl.use('TkAgg')

import os
import glob

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


# ---------------------------------------------------------
# 1. 定义 SNR 计算函数
# ---------------------------------------------------------
def calculate_snr_known(clean_signal, noise):
    """
    计算信噪比
    SNR = 10 * log10( P_signal / P_noise )
    """
    p_signal = np.sum(clean_signal ** 2)
    p_noise = np.sum(noise ** 2)

    if p_noise == 0:
        return float('inf')

    return 10 * np.log10(p_signal / p_noise)

# ==========================================
# 3. Calculate Spectrums
# ==========================================
def get_average_spectrum(data, dt):
    """Calculates the average amplitude spectrum of the input data."""
    # Perform FFT along the time axis (axis=0)
    # rfft is used for real input, returning positive frequencies only
    fft_vals = np.fft.rfft(data, axis=0)

    # Calculate amplitude (magnitude)
    amp_spectrum = np.abs(fft_vals)

    # Calculate mean across all traces (axis=0)
    mean_amp = np.mean(amp_spectrum, axis=1)

    # Generate frequency axis
    freqs = np.fft.rfftfreq(data.shape[0], dt)

    return freqs, mean_amp

if __name__ == '__main__':
    data_dir = r'F:\Data\OpenFWI\CurveVelB\train_data\seismic'
    data_path = r'F:\Data\OpenFWI\CurveVelB\train_data\seismic\seismic1.npy'
    # 匹配目录下所有的 .npy 文件
    file_list = glob.glob(os.path.join(data_dir, '*.npy'))

    # 噪声参数
    noise_mean = 0
    noise_std = 0.1

    # ---------------------------------------------------------
    # 3. 计算 SNR 信噪比
    # ---------------------------------------------------------
    '''
    if not file_list:
        print("未找到数据文件，请检查路径。")
    else:
        print(f"找到 {len(file_list)} 个文件，开始计算...")

    snr_list = []
 
    for i, file_path in enumerate(file_list):
        try:
            # 加载数据
            data_all = np.load(file_path, mmap_mode='r')

            # 遍历该文件中的每一炮
            for shot_idx in range(data_all.shape[0]):
                # 获取单炮数据 (直接取原值，不做归一化)
                data = data_all[shot_idx]

                # 生成噪声
                # size=data.shape 确保噪声形状与数据一致
                noise = np.random.normal(loc=noise_mean, scale=noise_std, size=data.shape)

                # 计算 SNR
                current_snr = calculate_snr_known(data, noise)
                snr_list.append(current_snr)

            # 打印进度
            if (i + 1) % 10 == 0:
                print(f"已处理文件: {i + 1}/{len(file_list)}")

        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")

      
        if snr_list:
            avg_snr = np.mean(snr_list)
            print("-" * 30)
            print(f"处理完成！")
            print(f"总样本数 (炮数): {len(snr_list)}")
            print(f"噪声标准差设置: {noise_std}")
            print(f"平均信噪比 (Average SNR): {avg_snr:.2f} dB")
            print("-" * 30)
        else:
            print("没有成功计算任何数据的 SNR。")
    '''

    # ============================
    # 2. 计算平均频谱
    # ============================
    # 2.1 执行傅里叶变换 (沿时间轴 axis=0 变换)
    # 使用 rfft 因为地震数据是实数，只计算正频率
    seismic_data = np.load(data_path, mmap_mode='r')[0,2,:,:]
    n_time = 1000  # 时间采样点
    n_trace = 70  # 道数
    dt = 0.001  # 采样间隔 (秒), 对应 500Hz 采样率 (如果是 4ms 请改为 0.004)
    fs = 1 / dt  # 采样频率

    noise = np.random.normal(loc=noise_mean, scale=noise_std, size=seismic_data.shape)

    # Add noise to original data
    noisy_data = seismic_data + noise

    # Get spectrum for Original Data
    freqs_orig, spectrum_orig = get_average_spectrum(seismic_data, dt)

    # Get spectrum for Noisy Data
    freqs_noisy, spectrum_noisy = get_average_spectrum(noisy_data, dt)

    # ==========================================
    # 4. Plotting
    # ==========================================
    plt.style.use('seaborn-v0_8-whitegrid')  # Clean style

    plt.figure(figsize=(10, 6))

    # Plot Original Spectrum
    plt.plot(freqs_orig, spectrum_orig, label='Original Data', color='blue', linewidth=2)

    # Plot Noisy Spectrum
    plt.plot(freqs_noisy, spectrum_noisy, label='Data with Noise', color='red', linewidth=2, alpha=0.7)

    # Chart Settings
    plt.title('Average Amplitude Spectrum Comparison', font21)
    plt.xlabel('Frequency (Hz)', font21)
    plt.ylabel('Amplitude', font21)

    # Limit X-axis to Nyquist frequency (or lower for better visibility)
    plt.xlim(0, 150)

    plt.legend(
        fontsize=16,  # 字体变大 (默认通常是 10 或 12)
        loc='upper right',  # 位置保持不变
        frameon=True,  # 显示边框
        fancybox=True,  # 使用圆角边框
        framealpha=1,  # 不透明背景
        shadow=True,  # 添加阴影，增加立体感
        borderpad=1.2,  # 增加内部填充 (默认是 0.4，1.0 会显得更宽敞)
        handlelength=3.5  # 增加线条长度 (默认是 2)
    )
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    # noise_mean = 0
    # noise_std = 0.1
    # noise = np.random.normal(loc=noise_mean, scale=noise_std, size=data_img.shape)
    # data_img_noise = data_img + noise
    # ny = 100
    # nt=2000
    # dx = 30
    # dt = 0.003
    # vmin, vmax = np.percentile(data_img[:, 1].cpu().numpy(), [4, 96])
    # plt.imshow(data_img[:, 15].cpu().numpy(), extent=[0, (ny - 1) * dx / 1000, (nt - 1) * dt, 0],
    #            vmin=vmin, vmax=vmax,
    #            aspect='auto', cmap="gray")
    # plt.show()

    # pain_openfwi_seismic_data(data_img)
    # pain_openfwi_seismic_data(data_img_noise)

    # edges = extract_contours(data_img)
    #
    # plt.imshow(edges, cmap='gray')  # cmap='gray' 表示灰度图
    # plt.colorbar()  # 可选：添加颜色条
    # plt.axis('off')  # 可选：去除坐标轴
    # plt.show()  # 显示图像

    # 应用高斯模糊
    # sigma = 1.0  # 设定标准差
    # blurred_matrix = scipy.ndimage.gaussian_filter(data_img, sigma=sigma)
    # pain_openfwi_velocity_model(blurred_matrix)

    # 保存模糊后的矩阵

    # vmodel_path = 'F:/Data/OpenFWI/CurveVelA/train_data/vmodel/vmodel1.npy'
    # vmodel = np.load(vmodel_path, mmap_mode='r')
    # data_path = 'F:/Data/OpenFWI/CurveVelA/train_data/seismic/seismic1.npy'
    # generate_data_path = 'G:/minfan/myTask/Data_synthesis/Data/FlatVelA/train_data/vmodel/seismic1.npy'
    # pain_openfwi_velocity_model(data[0][0][:][:])
    # generate_seismic_data = DataFromNpy(generate_data_path)[0][2][:][:]
    # PlotComparison_openfwi_seismic_data(seismic_data, generate_seismic_data)
    # pain_openfwi_seismic_data(data[300][2][:][:])
    # pain_openfwi_velocity_model(vmodel[300][0][:][:])

    # print(data[400][2][:][30])
    # matrix = data[400][2][35][:].squeeze()
    #
    #
    #
    # # 归一化处理
    # matrix_normalized = matrix / np.max(np.abs(matrix))  # 归一化到 [-1, 1]
    #
    # # 生成 x 轴
    # x = np.linspace(0, len(matrix) - 1, len(matrix))
    #
    # # 绘图
    # plt.figure(figsize=(10, 6))
    # plt.plot(x, matrix_normalized, marker='o', linestyle='-', color='b', label='Normalized Waveform')
    #
    # plt.title("Normalized Waveform Plot")
    # plt.xlabel("Time / Position")
    # plt.ylabel("Normalized Amplitude")
    # plt.axhline(0, color='black', linestyle='--', linewidth=1)  # 添加零线
    # plt.legend()
    # plt.grid()
    # plt.show()

    # # vlim = max(abs(np.max(data)), abs(np.min(data)))
    # plt.imshow(data, cmap=plt.cm.seismic)
    # # plt.imshow(data, cmap=plt.cm.seismic, vmax=1, vmin=-1, aspect="auto")
    # plt.colorbar()
    # plt.show()

    # generate_data_path = 'D:/Xu-Qiong/Data/FlatVelA/train_data/seismic_f5/seismic1.mat'
    # data = loadmat(generate_data_path)
    # # 获取数据
    # data = data['Recs']  # 最大值1.27   最小值-0.99
    # filename = generate_data_path.replace('mat', 'npy')
    #
    # np.save(filename, data)

    # seismic = np.load(data_path)[0][4][:][:]
    # seismic_gen = np.load(generate_data_path)[0][4][:][:]
    #
    # A = seismic[0][65:70:1]
    # B = seismic_gen[0][65:70:1]
    #
    #
    #
    # PlotComparison_openfwi_seismic_data(seismic, seismic_gen)

    '''
    data_f15_path = 'D:/Xu-Qiong/Data/CurveFaultA/train_data/seismic/seismic1.npy'
    data_f25_path = 'D:/Xu-Qiong/Data/CurveFaultA/train_data/seismic_f25/seismic1.npy'
    data_f5_path = 'D:/Xu-Qiong/Data/CurveFaultA/train_data/seismic_f5/seismic1.npy'
    vmodel_data_path = 'D:/Xu-Qiong/Data/CurveFaultA/train_data/vmodel/vmodel1.npy'
    seismic_f15_data = np.load(data_f15_path, mmap_mode='r')
    seismic_f25_data = np.load(data_f25_path, mmap_mode='r')
    seismic_f5_data = np.load(data_f5_path, mmap_mode='r')
    vmodel_data = np.load(vmodel_data_path, mmap_mode='r')
    # # PlotComparison_openfwi_seismic_data(seismic_data, generate_seismic_data)
    # pain_openfwi_velocity_model(vmodel_data[0, 0, :, :])
    print(np.amax(seismic_f25_data[0, 3, :, :]))
    print(np.amin(seismic_f25_data[0, 3, :, :]))
    print(np.amax(np.abs(seismic_f25_data[0, 3, :, :])))
    print(np.amin(np.abs(seismic_f25_data[0, 3, :, :])))
    data = [] #seismic_f15_data[0, 2, :, :]
    sns.kdeplot(x=data[:, 0], y=data[:, 1], cmap="Blues", shade=True, thresh=0.05)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('2D Density Plot')
    plt.show()


    # pain_openfwi_seismic_data(seismic_f15_data[0, 2, :, :])
    # # pain_openfwi_seismic_data(seismic_f5_data[0, 2, :, :])
    # pain_openfwi_seismic_data(seismic_f25_data[0, 2, :, :])
    # # pain_openfwi_seismic_data(seismic_f15_data[0, 2, :, :] - seismic_f5_data[0, 2, :, :])
    # pain_openfwi_seismic_data(seismic_f25_data[0, 2, :, :] - seismic_f15_data[0, 2, :, :])
    # # pain_openfwi_seismic_wave(seismic_f5_data[0, 2, :, 0], 'f5.png')
    # # pain_openfwi_seismic_wave(seismic_f15_data[0, 2, :, 0], 'f15.png')
    # # pain_openfwi_seismic_wave(seismic_f25_data[0, 2, :, 0], 'f25.png')

    # painComparison_openfwi_seismic_wave(seismic_f5_data[0, 2, :, 0], seismic_f15_data[0, 2, :, 0],
    #                                     seismic_f25_data[0, 2, :, 0], '0channel.png')
    # painComparison_openfwi_seismic_wave(seismic_f5_data[0, 2, :, 10], seismic_f15_data[0, 2, :, 10],
    #                                     seismic_f25_data[0, 2, :, 10], '10channel.png')
    # painComparison_openfwi_seismic_wave(seismic_f5_data[0, 2, :, 20], seismic_f15_data[0, 2, :, 20],
    #                                     seismic_f25_data[0, 2, :, 20], '20channel.png')
    # painComparison_openfwi_seismic_wave(seismic_f5_data[0, 2, :, 30], seismic_f15_data[0, 2, :, 30],
    #                                     seismic_f25_data[0, 2, :, 30], '30channel.png')
    # painComparison_openfwi_seismic_wave(seismic_f5_data[0, 2, :, 40], seismic_f15_data[0, 2, :, 40],
    #                                     seismic_f25_data[0, 2, :, 40], '40channel.png')
    # painComparison_openfwi_seismic_wave(seismic_f5_data[0, 2, :, 50], seismic_f15_data[0, 2, :, 50],
    #                                     seismic_f25_data[0, 2, :, 50], '50channel.png')
    # painComparison_openfwi_seismic_wave(seismic_f5_data[0, 2, :, 60], seismic_f15_data[0, 2, :, 60],
    #                                     seismic_f25_data[0, 2, :, 60], '60channel.png')
    '''

    # print("ok")
    # data_path = 'D:/Data/Marmousi/seismic/seismic_mar_big_117_567_75.npy'
    # seismic_data = np.load(data_path, mmap_mode='r')
    # pain_openfwi_seismic_data(seismic_data[0, 2, :, :])