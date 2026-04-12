import numpy as np
from scipy import io
from scipy.stats import percentileofscore

# 1. 读取 .mat 文件
# 请确保文件名正确，例如 'data.mat'
path = r'E:\Code\DST_FWI_base_5.0\test_result\CurveFaultB\TDMF-FWI_TrainSize48000_Epoch160_BatchSize20_LR0.0001_TestResults.mat'
mat_file = io.loadmat(path)

# 2. 提取数据
# loadmat 读取的数据是字典格式，键名对应 MATLAB 中的变量名
# 建议先打印 mat_file.keys() 确认键名是否为 'GT' 和 'Prediction'
GT = mat_file['GT']
Prediction = mat_file['Prediction']

# 数据清洗：MATLAB 读取的数据通常是二维矩阵 (N, 1)，需要展平为一维数组
GT = GT.flatten()
Prediction = Prediction.flatten()

# 3. 计算 MSE (均方误差)
# 公式：MSE = mean((真实值 - 预测值)^2)
mse_value = np.mean((GT - Prediction) ** 2)

print(f"MSE 指标结果: {mse_value:.4f}")

# 4. 计算第 7 个预测结果在总结果中的百分位
# 获取第 7 个值 (索引为 6，因为 Python 从 0 开始计数)
i = 44
val_7 = Prediction[i]

# 计算百分位排名
# percentileofscore 计算的是有多少比例的数据小于或等于给定值
# kind='rank' 是默认行为，处理相同数值时的平均排名
percentile_rank = percentileofscore(Prediction, val_7)

print(f"第 7 个预测结果的值为: {val_7:.4f}")
print(f"该值在所有预测结果中的百分位排名为: {percentile_rank:.2f}%")
