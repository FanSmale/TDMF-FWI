关于本项目的说明：
1、ABA-FWI方法从网络结构和损失函数增强边界注意。
对比方法包括：InversionNet、VelocityGAN、DD-Net70、ABA-Net。


2、对比实验的损失函数，均为L1和L2的联合损失。

3、消融实验和代码的对应关系
ABA-Net = IAEDN + L1和L2损失
ABA-Loss = InversionNet + RCTB Loss
ABA-FWI = IAEDN + RCTB Loss

4、训练参数设置
学习率0.0001，batchsize为20， VelocityGAN大概400epoch左右收敛，其他方法大概120至140epoch左右收敛。

5、训练、验证方法对应的代码文件
train_valid_Inversion.py  --------> InversionNet   ABA-Net  DD-Net70
train_valid_Inversion_TV.py -------->  ABA-Loss ABA-FWI
train_valid_velocityGAN.py -------->  VelocityGAN

6、测试代码文件
test.py

7、关于论文中展示结果的相关代码，请参考RESULT*.py