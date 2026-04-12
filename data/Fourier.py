import torch
import torch.fft


def model_fourier_loss(
        m_pred: torch.Tensor,
        m_true: torch.Tensor,
        alpha_z: float = -2.0,  # depth (vertical) frequency weighting
        alpha_x: float = -2.0,  # horizontal frequency weighting
        eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute a Fourier-based loss for 2D velocity models.
    Encourages matching of large-scale (low-wavenumber) structures first.

    Args:
        m_pred (torch.Tensor): Predicted model, shape (Nz, Nx)
        m_true (torch.Tensor): True model, same shape
        alpha_z (float): Weighting exponent for vertical wavenumber (kz)
        alpha_x (float): Weighting exponent for horizontal wavenumber (kx)
        eps (float): Small value to avoid division by zero

    Returns:
        loss (torch.Tensor): Scalar
    """
    assert m_pred.shape == m_true.shape and m_pred.ndim == 2, "Input must be 2D (Nz, Nx)"
    device = m_pred.device
    dtype = m_pred.dtype

    # 2D FFT (real input → complex spectrum)
    m_pred_f = torch.fft.rfft2(m_pred)  # Shape: (Nz, Nkx), where Nkx = Nx//2 + 1
    m_true_f = torch.fft.rfft2(m_true)

    Nz, Nkx = m_pred_f.shape

    # Create wavenumber grids
    kz = torch.arange(Nz, device=device, dtype=dtype)  # vertical wavenumbers [0, 1, ..., Nz-1]
    kx = torch.arange(Nkx, device=device, dtype=dtype)  # horizontal wavenumbers [0, 1, ..., Nkx-1]

    # Make them symmetric for negative frequencies (optional, but rfft2 already handles it)
    # For simplicity, we use the rfft2 output indexing directly

    # Compute weights: |k|^alpha = (kz^2 + kx^2)^{alpha/2}  OR separable: |kz|^αz * |kx|^αx
    # We use separable for simplicity and control
    kz_weight = (kz + eps) ** alpha_z  # (Nz,)
    kx_weight = (kx + eps) ** alpha_x  # (Nkx,)

    # Form 2D weight grid via outer product
    weights_2d = kz_weight[:, None] * kx_weight[None, :]  # (Nz, Nkx)

    # Compute weighted squared error in Fourier domain
    diff = m_pred_f - m_true_f
    loss = 0.5 * torch.sum(weights_2d * (torch.abs(diff) ** 2))

    return loss

# True and predicted velocity models: (depth, width)
m_true = torch.randn(70, 70).cuda() * 1000 + 2000  # e.g., 2000–3000 m/s
m_pred = torch.randn(70, 70).cuda() * 1000 + 2000

# Compute model-domain Fourier loss
loss = model_fourier_loss(m_pred, m_true, alpha_z=-2.0, alpha_x=-2.0)
print(loss)  # scalar

# loss.backward()  # if m_pred is from a neural network or differentiable parameter