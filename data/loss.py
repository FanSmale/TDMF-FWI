import torch
from utils import *
from data.show import *
import torch.nn.functional as F


def reflection_coe(vmodels):
    """
    计算速度模型的反射系数
    """


    x_deltas = vmodels[:, :, 1:, :] - vmodels[:, :, :-1, :]
    x_sum = vmodels[:, :, 1:, :] + vmodels[:, :, :-1, :]
    ref = x_deltas / x_sum

    ref[torch.isnan(ref)] = 0 # 归一化后速度值为0，去除除数为0的情况

    result = torch.zeros_like(vmodels)


    # 将原始矩阵放入全零矩阵中
    result[:, :, 1:, :] = torch.abs(ref)
    # import matplotlib.pyplot as plt
    # plt.imshow(result.detach().cpu().numpy()[0][0][:][:], cmap='gray')
    # plt.axis('off')  # 关闭坐标轴
    # plt.show()
    return result


def reflection_weight(ref, edges):
    """
    基于反射数据的权重系数
    :param ref:
    :return:
    """
    # 定义一个3x3的最大池化层
    max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    # 计算邻域内的最大值
    ref_max = max_pool(ref)
    edge_dilate = dilate_tv(edges)
    ref_ideal = ref_max * edge_dilate
    # ref_result = torch.where(ref_ideal != 0, 1 / ref_ideal, ref_ideal)
    # ref_result = torch.where(ref_result > 100, 100, ref_result)
    ref_result = torch.where((ref_ideal != 0) & (ref_ideal < 0.05), 2, 1) * edge_dilate

    return ref_result


def total_variation_loss_xy(vmodel_ideal):
    """
    :param vmodel_ideal:   vmodels  tensor  [none, 1, 70, 70]
    :return: tensor  [none, 1, 70, 70]
    """
    # 计算图像在 x 和 y 方向的梯度
    x_deltas = vmodel_ideal[:, :, 1:, :] - vmodel_ideal[:, :, :-1, :]
    y_deltas = vmodel_ideal[:, :, :, 1:] - vmodel_ideal[:, :, :, :-1]

    x_deltas_padded_matrix = torch.zeros_like(vmodel_ideal)
    y_deltas_padded_matrix = torch.zeros_like(vmodel_ideal)

    # 将原始矩阵放入全零矩阵中
    x_deltas_padded_matrix[:, :, 1:, :] = x_deltas
    y_deltas_padded_matrix[:, :, :, 1:] = y_deltas

    return x_deltas_padded_matrix, y_deltas_padded_matrix


# def dilate_tv(loss_out_w):
#
#     # 创建膨胀的内核（kernel）
#     kernel = torch.ones((1, 1, 2, 2), dtype=torch.float).to('cuda')  # 适用于多通道的 3x3 内核
#     loss_out_w = loss_out_w.to(torch.float)
#     # 使用卷积进行膨胀操作
#     dilated_tensor = F.conv2d(loss_out_w, kernel,
#                               padding=(1,1), stride=1)
#     result = torch.zeros_like(dilated_tensor)
#     result[dilated_tensor != 0 ] = 1
#     x = result[:, :, :70, :70]
#     return x #result#result[:][:][1:][1:]

def dilate_tv(loss_out_w):

    # 创建膨胀的内核（kernel）
    kernel = torch.ones((1, 1, 3, 3), dtype=torch.float).to('cuda')  # 适用于多通道的 3x3 内核
    loss_out_w = loss_out_w.to(torch.float)
    # 使用卷积进行膨胀操作
    dilated_tensor = F.conv2d(loss_out_w, kernel,
                              padding=1, stride=1)
    result = torch.zeros_like(dilated_tensor)
    result[dilated_tensor != 0 ] = 1
    return result

def loss_tv_1p_edge_ref_w(pred, vmodels, edges):
    """
    求两图像在两个方向上偏微分的一阶导数   加反射系数权重
    :param pred:
    :param vmodel_ideal:
    :return:
    """
    pred_x, pred_y = total_variation_loss_xy(pred)
    vmodel_ideal_x, vmodel_ideal_y = total_variation_loss_xy(vmodels)
    total_variation = torch.abs(pred_x - vmodel_ideal_x) + torch.abs(pred_y - vmodel_ideal_y)
    edge_weight = dilate_tv(edges)

    ref = reflection_coe(vmodels)
    ref_weight = reflection_weight(ref, edges)
    ref_variation = total_variation * ref_weight
    # pain_openfwi_velocity_model(vmodels[0, 0, ...].detach().cpu().numpy())
    # pain_openfwi_velocity_model(edge_weight[0, 0, ...].detach().cpu().numpy())
    # pain_openfwi_velocity_model(ref[0, 0, ...].detach().cpu().numpy())
    # pain_openfwi_velocity_model(ref_weight[0, 0, ...].detach().cpu().numpy())
    # pain_openfwi_velocity_model(ref_variation[0, 0, ...].detach().cpu().numpy())
    # print(ref[0, 0, :, 1])
    # print(ref_weight[0, 0, :, 1])

    loss = torch.sum(ref_variation)

    loss = loss / (vmodels.size(0) * torch.sum(edge_weight))
    return loss

def loss_tv1(pred, vmodels, edges):
    """
    求两图像在两个方向上偏微分的一阶导数   加反射系数权重
    :param pred:
    :param vmodel_ideal:
    :return:
    """
    pred_x, pred_y = total_variation_loss_xy(pred)
    vmodel_ideal_x, vmodel_ideal_y = total_variation_loss_xy(vmodels)
    total_variation = torch.abs(pred_x - vmodel_ideal_x) + torch.abs(pred_y - vmodel_ideal_y)
    edge_weight = dilate_tv(edges)

    ref = reflection_coe(vmodels)
    # import matplotlib.pyplot as plt
    # plt.imshow(ref.detach().cpu().numpy()[0][0][:][:], cmap='gray')
    # plt.axis('off')  # 关闭坐标轴
    # plt.show()
    ref_weight = reflection_weight(ref, edges)
    # import matplotlib.pyplot as plt
    # plt.imshow(ref_weight.detach().cpu().numpy()[0][0][:][:], cmap='gray')
    # plt.axis('off')  # 关闭坐标轴
    # plt.show()
    ref_variation = total_variation * ref_weight
    # pain_openfwi_velocity_model(vmodels[0, 0, ...].detach().cpu().numpy())
    # pain_openfwi_velocity_model(edge_weight[0, 0, ...].detach().cpu().numpy())
    # pain_openfwi_velocity_model(ref[0, 0, ...].detach().cpu().numpy())
    # pain_openfwi_velocity_model(ref_weight[0, 0, ...].detach().cpu().numpy())
    # pain_openfwi_velocity_model(ref_variation[0, 0, ...].detach().cpu().numpy())
    # print(ref[0, 0, :, 1])
    # print(ref_weight[0, 0, :, 1])

    loss = torch.sum(ref_variation)

    a=vmodels.size(0)
    b=torch.sum(edge_weight)
    loss = loss / (vmodels.size(0) * torch.sum(edge_weight))
    return loss


l1loss = nn.L1Loss()
l2loss = nn.MSELoss()


def criterion(pred, gt):
    loss_g1v = l1loss(pred, gt)
    loss_g2v = l2loss(pred, gt)
    result_loss = loss_g1v + loss_g2v
    return result_loss, loss_g1v, loss_g2v


def criterion_g(pred, gt, net_d=None):
    l1loss = nn.L1Loss()
    l2loss = nn.MSELoss()
    loss_g1v = l1loss(pred, gt)
    loss_g2v = l2loss(pred, gt)
    loss = 100 * loss_g1v + 100 * loss_g2v
    if net_d is not None:
        loss_adv = -torch.mean(net_d(pred))
        loss += loss_adv
    return loss, loss_g1v, loss_g2v


class Wasserstein_GP(nn.Module):
    def __init__(self, device, lambda_gp):
        super(Wasserstein_GP, self).__init__()
        self.device = device
        self.lambda_gp = lambda_gp

    def forward(self, real, fake, model):
        gradient_penalty = self.compute_gradient_penalty(model, real, fake)
        loss_real = torch.mean(model(real))
        loss_fake = torch.mean(model(fake))
        loss = -loss_real + loss_fake + gradient_penalty * self.lambda_gp
        return loss, loss_real-loss_fake, gradient_penalty

    def compute_gradient_penalty(self, model, real_samples, fake_samples):
        alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=self.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = model(interpolates)
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(real_samples.size(0), d_interpolates.size(1)).to(self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty