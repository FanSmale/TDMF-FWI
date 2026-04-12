import numpy as np
import matplotlib.pyplot as plt
import scipy.io
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

# ================= 配置区域 =================
# 1. 模拟您的 1*70 数据 (请在此处替换为您真实的井数据)
# 假设这是速度数据 (单位: m/s)，例如从 1500 m/s 逐渐增加到 3000 m/s

dir = "E:/Code/DST_FWI_base_5.0/test_result/CurveVelB/"
TDMF_DIR = dir +"DST-FWI_TrainSize24000_Epoch160_BatchSize20_LR0.0001_TestResults.mat"
# CurveFaultB [44]10
# CurveFaultA [85]20
# CurveVelB [80]30
# CurveVelA [91]5
# i = 240 for marmousi
i = 80
j = 30
dz = 10  # 网格间距 10 米
pd_slice = scipy.io.loadmat(TDMF_DIR)['Prediction'][i,:,j]
gt_slice = scipy.io.loadmat(TDMF_DIR)['GT'][i,:,j]


def calculate_twt(v, dz):
    twt = np.zeros(len(v))
    t = 0
    for k, vel in enumerate(v):
        if vel <= 0: vel = 1e-6
        t += 2.0 * dz / vel
        twt[k] = t
    return twt

twt_gt = calculate_twt(gt_slice, dz)
twt_pd = calculate_twt(pd_slice, dz)
depth_axis = np.arange(1, len(gt_slice) + 1) * dz

# ================= 3. 绘图 (双坐标轴方案) =================
fig, ax1 = plt.subplots(figsize=(12, 10))

# --- 左轴/底轴：时间 - 深度 (主图) ---
color_time = 'black'
ax1.set_xlabel('Two-Way Travel Time (s)', font21)
ax1.set_ylabel('Depth (m)', font21)
ax1.invert_yaxis() # 深度向下

# 绘制时间曲线
line_gt, = ax1.plot(twt_gt, depth_axis, '-', color='#1f77b4', linewidth=2.5, label='GT (Time)')
line_pd, = ax1.plot(twt_pd, depth_axis, '--', color='#d62728', linewidth=2, alpha=0.8, label='PD (Time)')

ax1.tick_params(axis='y', labelsize=12)
ax1.tick_params(axis='x', labelsize=12)
ax1.grid(True, which='both', linestyle='--', alpha=0.3, zorder=0)

# --- 右轴：速度 (副图) ---
ax2 = ax1.twinx()  # 共享 X 轴 (但在本例中，我们需要速度对应深度，所以实际上我们要把速度画在另一个维度)
# 注意：通常速度是随深度变化的，所以速度曲线应该和深度轴平行。
# 为了美观，我们通常把速度画在“右侧”作为第二个 X 轴？不，速度是 Y(深度) 的函数。
# 标准做法：左边是时间-深度，右边是速度-深度。
# 但 twinX 是共享 Y 轴。我们需要的是共享 Y (深度)，但有两个 X 轴 (时间和速度)？
# 或者：共享 Y (深度)，左边画时间，右边画速度？ -> 这样两条线会重叠，无法区分。

# ✅ 修正策略：三轴布局 或 双 X 轴
# 最好的地球物理显示方式：
# Y 轴：深度
# 左 X 轴：时间
# 右 X 轴：速度 (将速度曲线映射到右侧刻度)
# 但是时间和速度的量纲完全不同，不能共用 X 轴范围。

# 🌟 最佳实践方案：使用 secantx (第二个 X 轴)
# 左 X 轴显示时间，右 X 轴显示速度。
# 曲线1 (时间): 对应 左X & Y
# 曲线2 (速度): 对应 右X & Y (需要重新绘制速度曲线)

ax2 = ax1.twiny() # 创建一个共享 Y 轴 (深度) 的新 X 轴 (顶部或底部，默认顶部，我们移到底部右侧或保持顶部)
# 让我们把速度轴放在右侧，但这通常是 Y 轴的位置。
# 其实最简单的是：左边画时间曲线，右边画速度曲线，共用中间的深度轴。

# 重新规划布局：
# 方法 A: 两个子图并排 (最清晰)
# 方法 B: 单图，左轴时间，右轴速度 (曲线颜色区分) -> 这会导致线条交叉混乱。

# 👉 决定采用【方法 A：左右双子图】，这是最专业且不易混淆的展示方式。
# 左图：时深曲线对比
# 右图：速度曲线对比 (直接显示速度值)

fig, (ax_time, ax_vel) = plt.subplots(1, 2, figsize=(16, 10), sharey=True)

# --- 左图：时深关系 ---
ax_time.plot(twt_gt, depth_axis, '-', color='#1f77b4', linewidth=2.5, label='Ground Truth (Time)')
ax_time.plot(twt_pd, depth_axis, '--', color='#d62728', linewidth=2, label='TDMF-FWI (Time)')
ax_time.set_xlabel('Time (s)', font21)
ax_time.set_title('Time-Depth Curve', font21)
ax_time.invert_yaxis()
ax_time.grid(True, linestyle='--', alpha=0.5)
ax_time.legend(loc='lower right', prop={'family': 'Times New Roman', 'size': 18})
# 格式化时间轴
xticks = ax_time.get_xticks()
ax_time.set_xticklabels([f"{x:.3f}" for x in xticks])
ax_time.tick_params(axis='both', labelsize=16)  # 刻度数字字号调大为16

# --- 右图：速度关系 (直接显示速度值) ---
ax_vel.plot(gt_slice, depth_axis, '-', color='#1f77b4', linewidth=2.5, label='Ground Truth (Velocity)')
ax_vel.plot(pd_slice, depth_axis, '--', color='#d62728', linewidth=2, label='TDMF-FWI (Velocity)')
ax_vel.set_xlabel('Velocity (m/s)', font21)
ax_vel.set_title('Velocity Profile', font21)
ax_vel.invert_yaxis()
ax_vel.grid(True, linestyle='--', alpha=0.5)
ax_vel.legend(loc='lower right', prop={'family': 'Times New Roman', 'size': 18})
# 共享 Y 轴标签
ax_time.set_ylabel('Depth (m)', font21)
ax_vel.tick_params(axis='both', labelsize=16)  # 刻度数字字号调大为16

plt.suptitle(f'Well Comparison at Index [{i}, :, {j}]', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()