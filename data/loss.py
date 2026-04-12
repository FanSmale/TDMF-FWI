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

def loss_1p(pred, vmodels):
    """
    求两图像在两个方向上偏微分的一阶导数
    """
    pred_x, pred_y = total_variation_loss_xy(pred)
    vmodel_ideal_x, vmodel_ideal_y = total_variation_loss_xy(vmodels)
    total_variation = torch.abs(pred_x - vmodel_ideal_x) + torch.abs(pred_y - vmodel_ideal_y)
    loss = torch.sum(total_variation)
    loss = loss / (vmodels.size(0) * vmodels.size(1))
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

    # a=vmodels.size(0)
    # b=torch.sum(edge_weight)
    loss = loss / (vmodels.size(0) * torch.sum(edge_weight))
    return loss

def loss_fourier(
        m_pred: torch.Tensor,
        m_true: torch.Tensor,
        alpha_h: float = -2,  # vertical / height direction (e.g., depth)
        alpha_w: float = 0,  # horizontal / width direction 水平方向不需要强力约束低频
        eps: float = 1e-8,
) -> torch.Tensor:
    """
    Fourier-domain loss for 2D velocity models of shape (B, 1, H, W).

    Encourages matching low-wavenumber (large-scale) structures by weighting
    the Fourier spectrum with |k_h|^alpha_h * |k_w|^alpha_w.

    Args:
        m_pred (torch.Tensor): Predicted model, shape (B, 1, H, W)
        m_true (torch.Tensor): Ground truth model, same shape
        alpha_h (float): Exponent for height (first spatial dim) wavenumbers
        alpha_w (float): Exponent for width (second spatial dim) wavenumbers
        eps (float): Small constant to avoid division by zero at k=0

    Returns:
        loss (torch.Tensor): Scalar (mean over batch)
    """
    # --- 1. Input validation ---
    assert m_pred.shape == m_true.shape, f"Shape mismatch: {m_pred.shape} vs {m_true.shape}"
    assert m_pred.ndim == 4 and m_pred.shape[1] == 1, "Expected input shape (B, 1, H, W)"

    device = m_pred.device
    dtype = m_pred.dtype

    # Remove channel dimension: (B, 1, H, W) → (B, H, W)
    m_pred = m_pred.squeeze(1)  # (B, H, W)
    m_true = m_true.squeeze(1)  # (B, H, W)

    B, H, W = m_pred.shape

    # --- 2. 2D real FFT along last two dimensions ---
    # rfft2 returns (B, H, W//2 + 1) complex tensor
    m_pred_f = torch.fft.rfft2(m_pred, dim=(-2, -1))  # (B, H, K), K = W//2 + 1
    m_true_f = torch.fft.rfft2(m_true, dim=(-2, -1))

    _, H_f, W_f = m_pred_f.shape  # H_f = H, W_f = W//2 + 1

    # --- 3. Create wavenumber grids ---
    # Vertical (height/depth) wavenumbers: [0, 1, 2, ..., H-1]
    k_h = torch.arange(H_f, device=device, dtype=dtype)  # (H,)

    # Horizontal (width) wavenumbers: [0, 1, 2, ..., W_f-1]
    k_w = torch.arange(W_f, device=device, dtype=dtype)  # (W_f,)

    # Apply frequency weighting
    # 安全加权：避免 k=0 处权重过大
    weight_h = torch.where(k_h == 0, torch.tensor(1.0, device=device, dtype=dtype), (k_h + eps) ** alpha_h)
    weight_w = torch.where(k_w == 0, torch.tensor(1.0, device=device, dtype=dtype), (k_w + eps) ** alpha_w)

    # Form 2D weight grid via outer product: (H, W_f)
    weights_2d = weight_h[:, None] * weight_w[None, :]  # (H, W_f)
    '''
    # 转为 numpy
    w_np = weights_2d.cpu().numpy()
    k_h_np = k_h.cpu().numpy()
    k_w_np = k_w.cpu().numpy()

    # 创建网格（注意：k_h 是 Y，k_w 是 X）
    KX, KY = np.meshgrid(k_w_np, k_h_np)  # KX: (H, W_f), KY: (H, W_f)

    # 使用对数尺度（避免动态范围过大）
    Z = np.log10(w_np + 1e-12)  # log10(weight)

    # 3D 绘图
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制曲面
    surf = ax.plot_surface(KX, KY, Z, cmap='viridis', edgecolor='none', alpha=0.9)

    # 标签
    ax.set_xlabel('Horizontal wavenumber $k_x$')
    ax.set_ylabel('Vertical wavenumber $k_z$')
    ax.set_zlabel('log10(weight)')
    ax.set_title(f'3D Weight Surface (α_h={alpha_h}, α_w={alpha_w})\nShape: {w_np.shape}')

    # 添加颜色条
    fig.colorbar(surf, shrink=0.5, aspect=15, label='log10(weight)')

    plt.tight_layout()
    plt.show()
    '''
    # Expand to batch: (1, H, W_f) → broadcast over B
    weights_2d = weights_2d.unsqueeze(0)  # (1, H, W_f)

    # --- 4. Compute weighted L2 loss in Fourier domain ---
    diff = m_pred_f - m_true_f  # (B, H, W_f)
    squared_error = torch.abs(diff) ** 2  # (B, H, W_f)
    weighted_error = weights_2d * squared_error  # broadcasting over batch

    weighted_sum = weighted_error.sum(dim=(-2, -1))  # (B,)

    total_weight = weights_2d.sum()*70*36*100  # scalar
    loss_per_sample = weighted_sum / total_weight

    loss = 0.5 * loss_per_sample.mean()  # scalar

    return loss

def loss_fourier_mask(
        m_pred: torch.Tensor,
        m_true: torch.Tensor,
        k_h_max: int = 25,   #10,           只约束前 k_h_max 个垂直频率（低频）
        alpha_h: float = -0.4, #-1,   vertical / height direction (e.g., depth)
        alpha_w: float = -0.1, # 0,  horizontal / width direction 水平方向不需要强力约束低频
        eps: float = 1e-8,
) -> torch.Tensor:
    """
    Fourier-domain loss for 2D velocity models of shape (B, 1, H, W).

    Encourages matching low-wavenumber (large-scale) structures by weighting
    the Fourier spectrum with |k_h|^alpha_h * |k_w|^alpha_w.

    Args:
        m_pred (torch.Tensor): Predicted model, shape (B, 1, H, W)
        m_true (torch.Tensor): Ground truth model, same shape
        alpha_h (float): Exponent for height (first spatial dim) wavenumbers
        alpha_w (float): Exponent for width (second spatial dim) wavenumbers
        eps (float): Small constant to avoid division by zero at k=0

    Returns:
        loss (torch.Tensor): Scalar (mean over batch)
    """
    # --- 1. Input validation ---
    assert m_pred.shape == m_true.shape, f"Shape mismatch: {m_pred.shape} vs {m_true.shape}"
    assert m_pred.ndim == 4 and m_pred.shape[1] == 1, "Expected input shape (B, 1, H, W)"

    device = m_pred.device
    dtype = m_pred.dtype

    # Remove channel dimension: (B, 1, H, W) → (B, H, W)
    m_pred = m_pred.squeeze(1)  # (B, H, W)
    m_true = m_true.squeeze(1)  # (B, H, W)

    B, H, W = m_pred.shape

    # --- 2. 2D real FFT along last two dimensions ---
    # rfft2 returns (B, H, W//2 + 1) complex tensor
    m_pred_f = torch.fft.rfft2(m_pred, dim=(-2, -1))  # (B, H, K), K = W//2 + 1
    m_true_f = torch.fft.rfft2(m_true, dim=(-2, -1))

    _, H_f, W_f = m_pred_f.shape  # H_f = H, W_f = W//2 + 1

    # --- 3. Create wavenumber grids ---
    # Vertical (height/depth) wavenumbers: [0, 1, 2, ..., H-1]
    k_h = torch.arange(H_f, device=device, dtype=dtype)  # (H,)

    # Horizontal (width) wavenumbers: [0, 1, 2, ..., W_f-1]
    k_w = torch.arange(W_f, device=device, dtype=dtype)  # (W_f,)

    # 创建 2D 掩码：仅保留 k_h <= k_h_max 的行
    mask_h = (k_h <= k_h_max).float()  # (H,)
    mask_2d = mask_h[:, None]  # (H, 1) → 广播到 (H, W_f)

    # Apply frequency weighting
    # 安全加权：避免 k=0 处权重过大
    weight_h = torch.where(k_h == 0, torch.tensor(1.0, device=device, dtype=dtype), (k_h + eps) ** alpha_h)
    weight_w = torch.where(k_w == 0, torch.tensor(1.0, device=device, dtype=dtype), (k_w + eps) ** alpha_w)

    # Form 2D weight grid via outer product: (H, W_f)
    weights_2d = weight_h[:, None] * weight_w[None, :]  # (H, W_f)

    weights_2d = weights_2d * mask_2d

    # Expand to batch: (1, H, W_f) → broadcast over B
    weights_2d = weights_2d.unsqueeze(0)  # (1, H, W_f)

    # --- 4. Compute weighted L2 loss in Fourier domain ---
    diff = m_pred_f - m_true_f  # (B, H, W_f)
    squared_error = torch.abs(diff) ** 2  # (B, H, W_f)
    weighted_error = weights_2d * squared_error  # broadcasting over batch

    weighted_sum = weighted_error.sum(dim=(-2, -1))  # (B,)

    total_weight = weights_2d.sum().clamp(min=eps)  # scalar
    loss_per_sample = weighted_sum / total_weight

    loss = 0.5 * loss_per_sample.mean()  # scalar

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