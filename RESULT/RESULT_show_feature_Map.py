import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# # 创建子图并显示图像
# fig, axes = plt.subplots(6, 5, figsize=(7, 4))  # 创建12行8列的子图
#
# for i in range(6):
#     for j in range(5):
#         img = mpimg.imread("./feature_map/CVB(DST-FWI-3)/" + str(0) + '_' +str(j) + '_' +str(i) + '.png')  #CurveFaultA  FlatFaultA
#         axes[i, j].imshow(img, cmap='gray')
#         axes[i, j].axis('off')  # 关闭坐标轴
#
# plt.subplots_adjust(top=1, bottom=0, left=0, right=1, wspace=0.05, hspace=0.01)
# fig.savefig("./feature_map/CVB(DST-FWI-3).png")
# plt.show()

# # 创建子图并显示图像
# fig, axes = plt.subplots(6, 4, figsize=(4, 6))  # 创建12行8列的子图
#
# for i in range(6):
#     for j in range(4):
#         img = mpimg.imread("./feature_map/CVB(DST-FWI)/" + str(0) + '_' +str(j) + '_' +str(i) + '.png')  #CurveFaultA  FlatFaultA
#         axes[i, j].imshow(img, cmap='gray')
#         axes[i, j].axis('off')  # 关闭坐标轴
#
# plt.subplots_adjust(top=1, bottom=0, left=0, right=1, wspace=0.05, hspace=0.01)
# fig.savefig("./feature_map/CVB(DST-FWI).png")
# plt.show()
# --- 配置区域 ---
rows = 6
cols = 5
base_dir = "./feature_map/CVB(DST-FWI-3)"
output_path = "./feature_map/CVB(DST-FWI-3)_fixed_size.png"

# 【关键】强制设置单个子图的尺寸 (单位：英寸)
# 假设你的图片是正方形或接近正方形，可以设为 1.5 x 1.5
# 如果图片是长方形 (例如宽>高)，请调整 single_h 以匹配图片比例，防止留白或拉伸
single_w = 1.5  # 每个子图的宽度
single_h = 1.5  # 每个子图的高度 (若图片非正方形，请按图片实际高宽比调整，如 single_h = single_w * 0.8)

# 【关键】强制设置子图之间的间距 (单位：英寸)
# 注意：plt.subplots_adjust 的 wspace 是相对值 (占子图宽度的比例)
# 为了达到“固定英寸间距”的效果，我们需要计算这个相对值
gap_inch = 0.1  # 期望的物理间距 0.1 英寸

# 计算 wspace 和 hspace 的相对值 (间距 / 单图尺寸)
wspace_ratio = gap_inch / single_w
hspace_ratio = gap_inch / single_h

# 【核心】反推整个画布的 figsize
# 总宽 = 列数×单宽 + (列数-1)×间距
# 总高 = 行数×单高 + (行数-1)×间距
# 注意：figsize 不需要包含 edge margin，subplots_adjust 会处理边缘
total_fig_w = cols * single_w + (cols - 1) * gap_inch
total_fig_h = rows * single_h + (rows - 1) * gap_inch

# 创建画布
fig, axes = plt.subplots(rows, cols, figsize=(total_fig_w, total_fig_h))

# 确保 axes 是二维数组 (防止行列数为1时报错)
if rows == 1 or cols == 1:
    axes = axes.reshape(rows, cols)

for i in range(rows):
    for j in range(cols):
        img_path = os.path.join(base_dir, f"0_{j}_{i}.png")

        if os.path.exists(img_path):
            img = mpimg.imread(img_path)
            # aspect='equal' 强制保持数据本身的纵横比，配合固定的 axes 尺寸，效果最佳
            axes[i, j].imshow(img, cmap='gray', aspect='equal')
        else:
            # 如果图片缺失，显示空白或提示
            axes[i, j].text(0.5, 0.5, 'No Img', ha='center', va='center')

        axes[i, j].axis('off')  # 关闭坐标轴

# 应用计算好的间距比例
# left/right/top/bottom 设置为 0 到 1 的边缘留白，这里设为极小值以利用全部空间
plt.subplots_adjust(
    left=0.01, right=0.99,
    top=0.99, bottom=0.01,
    wspace=wspace_ratio,
    hspace=hspace_ratio
)

# 保存图片 (dpi=300 保证清晰度)
fig.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"画布总尺寸: {total_fig_w:.2f} x {total_fig_h:.2f} 英寸")
print(f"单图尺寸: {single_w} x {single_h} 英寸")
print(f"物理间距: {gap_inch} 英寸")