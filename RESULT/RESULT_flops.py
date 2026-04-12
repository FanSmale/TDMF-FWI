from thop import profile, clever_format

from net.ABA_FWI import *
from net.FCNVMB import *
from net.DST_FWI import *
from net.DST_FWI_GUASS import *
from net.TTDM import *
# from torchstat import stat
# from torchsummary import summary
from ptflops import get_model_complexity_info
# 定义网络模型
model = DST_FWI_G_L1_tanh()  #InversionNet  IAEDN_WTUU   DDNet70Model DST_FWI_trans
# 创建一个输入张量作为模型的输入
# wtuu 模型的 FLOPs: 3.129G
# 模型的参数数量: 9.858M
# 模型的 FLOPs: 2.150G
# 模型的参数数量: 7.632M
def print_thop(model):
    input = torch.randn(1, 5, 1000, 70)  # 替换为你自己的输入尺寸
    # 使用 THOP 计算模型的 FLOPs 和参数数量
    flops, params = profile(model, inputs=(input,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")

    print(f"模型的 FLOPs: {flops}")
    print(f"模型的参数数量: {params}")

def print_ptflops(model):
    flops, params = get_model_complexity_info(model, (5, 1000, 70), as_strings=True, print_per_layer_stat=True)
    print(flops)
    print(params)

print_thop(model)
#print_ptflops(model)

# TS_FWI
# 模型的 FLOPs: 2.867G
# 模型的参数数量: 20.336M

# ABA_Loss
# 模型的 FLOPs: 3.085G
# 模型的参数数量: 9.794M

# 模型的 FLOPs: 3.236G
# 模型的参数数量: 9.980M

# ABA_FWI
# 模型的 FLOPs: 3.129G
# 模型的参数数量: 9.858M

# ABA_Loss_conv
# 模型的 FLOPs: 4.449G
# 模型的参数数量: 11.465M

# DTS-FWI
# 模型的 FLOPs: 5.792G
# 模型的参数数量: 47.132M

# TTDM-FWI
# 模型的 FLOPs: 3.455G
# 模型的参数数量: 20.435M

# DTS-FWI-CONV
# 模型的 FLOPs: 2.867G
# 模型的参数数量: 20.336M

# DTS-FWI-CONV
# 模型的 FLOPs: 6.934G
# 模型的参数数量: 33.929M