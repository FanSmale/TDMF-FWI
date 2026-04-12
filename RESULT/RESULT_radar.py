# coding=utf-8

import numpy as np

import matplotlib
import matplotlib.pylab as plt

# 与ABA-FWI的消融
dataset_dict = {
    'CurveVelA': {
                    "ABA-FWI": [24.81222957,0.849809053,0.00407288,0.031964847,0.864742525,0.052257889,0.010308346,0.053147236],
                    "E-DST-FWI": [24.97959558,0.863630914,0.003824963,0.027828506,0.872685231,0.024581934,0.010264311,0.047258424],
                    "DST-FWI": [25.34480625,0.871529047,0.003619778,0.026478564,0.873221196,0.022678513,0.009781966,0.044867068]},
    'CurveFaultA': {
                    "ABA-FWI": [30.05954548,0.884886576,0.001240091,0.015563745,0.828864514,0.014728258,0.004674963,0.034911903],
                    "E-DST-FWI": [31.118037,0.865696398,0.000967255,0.015405569,0.82753458,0.010395816,0.003547437,0.030352466],
                    "DST-FWI": [31.42072919,0.93080918,0.000928708,0.013192255,0.836836697,0.010981284,0.003509257,0.028953138]},
    'CurveVelB': {
                    "ABA-FWI": [16.58145016,0.694872267,0.025505058,0.077409369,0.841970922,0.066362428,0.060035372,0.128495877],
                    "E-DST-FWI": [17.14647595,0.72401861,0.02244874,0.069860641,0.854341874,0.055393766,0.053452175,0.116144343],
                    "DST-FWI": [17.40170494,0.737459648,0.021295138,0.06796,0.858100433,0.052034715,0.051746835,0.113666976]},
    'CurveFaultB': {
                    "ABA-FWI": [18.78406709,0.57430699,0.01417144,0.076152331,0.907313815,0.151457334,0.019051617,0.093941951],
                    "E-DST-FWI": [19.05883756,0.588066219,0.013337726,0.07317337,0.912586503,0.139992559,0.018153792,0.091028151],
                    "DST-FWI": [19.16133223,0.588215785,0.013007347,0.072005159,0.911110423,0.12948586,0.017672065,0.089496565]},
}

for index in range(8):
    typ = index  # MSE:0 | MAE:1 | UIQ: 2 | LPIPS: 3
    typ_name = ""

    if typ == 0:
        typ_name = "PSNR"
        limset = (16, 32)
    elif typ == 1:
        typ_name = "SSIM"
        limset = (0.57, 0.94)
    elif typ == 2:
        typ_name = "MSE"
        limset = (0, 0.025)  # 依据自己的指标值确定
    elif typ == 3:
        typ_name = "MAE"
        limset = (0, 0.078)
    elif typ == 4:
        typ_name = "UIQ"
        limset = (0.8, 0.92)
    elif typ == 5:
        typ_name = "LPIPS"
        limset = (0, 0.16)
    elif typ == 6:
        typ_name = "BMSE"
        limset = (0, 0.07)  # 依据自己的指标值确定
    else:
        typ_name = "BMAE"
        limset = (0, 0.13)  # 依据自己的指标值确定
    savename = "Ablation-{}".format(typ_name)

    legends_names = ["ABA-FWI", "E-DST-FWI", "DST-FWI"]

    results = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]

    for i, dataset in enumerate(dataset_dict):
        for j, strategy in enumerate(legends_names):
            results[j].update({dataset: dataset_dict[dataset][strategy][typ]})

    data_length = len(results[0])

    # 将极坐标根据数据长度进行等分

    angles = np.linspace(np.pi / 4, 2.25 * np.pi, data_length, endpoint=False)

    labels = [str(key) for key in results[0].keys()]

    score = [[v for v in result.values()] for result in results]

    # 使雷达图数据封闭

    score_a = np.concatenate((score[0], [score[0][0]]))

    score_b = np.concatenate((score[1], [score[1][0]]))

    score_c = np.concatenate((score[2], [score[2][0]]))

    angles = np.concatenate((angles, [angles[0]]))

    labels = np.concatenate((labels, [labels[0]]))

    # 设置图形的大小

    fig = plt.figure(figsize=(4, 4), dpi=200)

    # 新建一个子图

    ax = plt.subplot(111, polar=True)

    # 绘制雷达图

    ax.plot(angles, score_a, color='g')

    ax.plot(angles, score_b, color='b')

    ax.plot(angles, score_c, color='c')

    # 设置雷达图中每一项的标签显示

    ax.set_thetagrids(angles * 180 / np.pi, labels)

    # 设置雷达图的0度起始位置

    ax.set_theta_zero_location('N')

    # 设置雷达图的坐标刻度范围

    ax.set_rlim(limset)

    # 设置雷达图的坐标值显示角度，相对于起始角度的偏移量

    ax.set_rlabel_position(300)

    # ax.set_title(Title_str)
    plt.legend(legends_names, bbox_to_anchor=(0.38, 0))

    plt.subplots_adjust(left=0, bottom=0.195, right=1, top=0.88,
                        wspace=0.225, hspace=0)
    plt.show()
    fig.savefig(str(savename) + '.png')
