# Copyright (c) OpenMMLab. All rights reserved.
# Enhanced LightHamHead with HFP_FSRA integration
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.device import get_device

from mmseg.registry import MODELS
from ..utils import resize
from .decode_head import BaseDecodeHead


# =========================
# HFP_FSRA Module (from first document)
# =========================
def _dct_mat(N: int, device=None, dtype=None):
    """Orthonormal DCT-II matrix of size (N, N)."""
    n = torch.arange(N, device=device, dtype=dtype).unsqueeze(0)  # [1, N]
    k = torch.arange(N, device=device, dtype=dtype).unsqueeze(1)  # [N, 1]
    mat = torch.cos((torch.pi / N) * (n + 0.5) * k)  # [N, N]
    mat *= torch.sqrt(torch.tensor(2.0 / N, device=device, dtype=dtype))
    mat[0, :] *= 1.0 / torch.sqrt(torch.tensor(2.0, device=device, dtype=dtype))
    return mat  # [N, N]


def dct2(x: torch.Tensor) -> torch.Tensor:
    """2D DCT-II with orthonormal normalization over last two dims (H, W)."""
    B, C, H, W = x.shape
    device, dtype = x.device, x.dtype
    CH = _dct_mat(H, device, dtype)  # [H, H]
    CW = _dct_mat(W, device, dtype)  # [W, W]
    y = x.reshape(B * C, H, W)
    y = torch.matmul(CH, y)  # DCT along H
    y = torch.matmul(y.transpose(1, 2), CW.T).transpose(1, 2)  # DCT along W
    return y.reshape(B, C, H, W)


def idct2(X: torch.Tensor) -> torch.Tensor:
    """2D IDCT (inverse of orthonormal DCT-II)."""
    B, C, H, W = X.shape
    device, dtype = X.device, X.dtype
    CH = _dct_mat(H, device, dtype)  # [H, H]
    CW = _dct_mat(W, device, dtype)  # [W, W]
    y = X.reshape(B * C, H, W)
    y = torch.matmul(y.transpose(1, 2), CW).transpose(1, 2)  # inverse along W
    y = torch.matmul(CH.T, y)  # inverse along H
    return y.reshape(B, C, H, W)


class SoftHighPass(nn.Module):
    """Learnable soft high-pass mask in frequency domain."""

    def __init__(self, init_alpha=(0.25, 0.25), init_beta=12.0):
        super().__init__()
        ah, aw = init_alpha
        self.alpha_h = nn.Parameter(torch.tensor(float(ah)).clamp(1e-4, 1 - 1e-4))
        self.alpha_w = nn.Parameter(torch.tensor(float(aw)).clamp(1e-4, 1 - 1e-4))
        self.log_beta = nn.Parameter(torch.log(torch.tensor(float(init_beta))))

    def forward(self, H: int, W: int, device, dtype):
        u = torch.arange(H, device=device, dtype=dtype) / H  # [H]
        v = torch.arange(W, device=device, dtype=dtype) / W  # [W]
        Uh = u.unsqueeze(1)  # [H,1]
        Vw = v.unsqueeze(0)  # [1,W]

        beta = torch.exp(self.log_beta) + 1e-6
        low_h = torch.sigmoid(beta * (self.alpha_h - Uh))  # [H,1]
        low_w = torch.sigmoid(beta * (self.alpha_w - Vw))  # [1,W]
        low_rect = low_h * low_w  # [H,W]
        mask = 1.0 - low_rect  # high-pass emphasis in [0,1]
        return mask  # [H, W]


class FSRA(nn.Module):
    """FSRA: Frequency-Selective Residual Attention."""

    def __init__(self,
                 in_channels: int,
                 groups: int = 32,
                 dw_kernel: int = 5,
                 dw_dilation: int = 1,
                 init_alpha=(0.25, 0.25),
                 init_beta=12.0):
        super().__init__()
        self.in_channels = in_channels
        self.groups = max(1, min(groups, in_channels))
        while in_channels % self.groups != 0 and self.groups > 1:
            self.groups //= 2
        self.soft_hp = SoftHighPass(init_alpha=init_alpha, init_beta=init_beta)

        # 频域通道门控
        self.spec_channel_gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=self.groups, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=self.groups, bias=False),
            nn.Sigmoid()
        )

        # 空间分支
        padding = ((dw_kernel - 1) // 2) * dw_dilation
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=dw_kernel,
                      padding=padding, dilation=dw_dilation, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.GELU()
        )

        # 频域refinement
        self.freq_refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=self.groups, bias=False),
            nn.GELU()
        )

        # 输出融合
        self.out = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(self.groups, in_channels), num_channels=in_channels)
        )

    def forward(self, x):
        identity = x

        # Frequency branch
        Xf = dct2(x)
        mask = self.soft_hp(H=Xf.size(-2), W=Xf.size(-1), device=Xf.device, dtype=Xf.dtype)
        mask = mask.view(1, 1, Xf.size(-2), Xf.size(-1)).expand_as(Xf)
        Xf = Xf * mask

        spec_pool = F.adaptive_avg_pool2d(Xf, output_size=(1, 1))
        spec_gate = self.spec_channel_gate(spec_pool)
        Xf = Xf * spec_gate

        x_freq = idct2(Xf)
        x_freq = self.freq_refine(x_freq)

        # Spatial branch
        x_spa = self.spatial_branch(x)

        # Fuse + Residual
        out = self.out(x_freq + x_spa)
        return identity + out


class HFP_FSRA(nn.Module):
    """HFP_FSRA wrapper for backward compatibility."""

    def __init__(self,
                 in_channels: int,
                 init_alpha=(0.25, 0.25),
                 init_beta=12.0,
                 groups=32,
                 dw_kernel=5,
                 dw_dilation=1):
        super().__init__()
        self.block = FSRA(in_channels=in_channels,
                          groups=groups,
                          dw_kernel=dw_kernel,
                          dw_dilation=dw_dilation,
                          init_alpha=init_alpha,
                          init_beta=init_beta)

    def forward(self, x):
        return self.block(x)


# =========================
# Original Matrix Decomposition modules
# =========================
class Matrix_Decomposition_2D_Base(nn.Module):
    """Base class of 2D Matrix Decomposition."""

    def __init__(self,
                 MD_S=1,
                 MD_R=64,
                 train_steps=6,
                 eval_steps=7,
                 inv_t=100,
                 rand_init=True):
        super().__init__()
        self.S = MD_S
        self.R = MD_R
        self.train_steps = train_steps
        self.eval_steps = eval_steps
        self.inv_t = inv_t
        self.rand_init = rand_init

    def _build_bases(self, B, S, D, R, device=None):
        raise NotImplementedError

    def local_step(self, x, bases, coef):
        raise NotImplementedError

    def local_inference(self, x, bases):
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)
        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)
        return bases, coef

    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    def forward(self, x, return_bases=False):
        B, C, H, W = x.shape
        D = C // self.S
        N = H * W
        x = x.view(B * self.S, D, N)

        if not self.rand_init and not hasattr(self, 'bases'):
            bases = self._build_bases(1, self.S, D, self.R, device=x.device)
            self.register_buffer('bases', bases)

        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R, device=x.device)
        else:
            bases = self.bases.repeat(B, 1, 1)

        bases, coef = self.local_inference(x, bases)
        coef = self.compute_coef(x, bases, coef)
        x = torch.bmm(bases, coef.transpose(1, 2))
        x = x.view(B, C, H, W)
        return x


class NMF2D(Matrix_Decomposition_2D_Base):
    """Non-negative Matrix Factorization (NMF) module."""

    def __init__(self, args=dict()):
        super().__init__(**args)
        self.inv_t = 1

    def _build_bases(self, B, S, D, R, device=None):
        if device is None:
            device = get_device()
        bases = torch.rand((B * S, D, R)).to(device)
        bases = F.normalize(bases, dim=1)
        return bases

    def local_step(self, x, bases, coef):
        numerator = torch.bmm(x.transpose(1, 2), bases)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        coef = coef * numerator / (denominator + 1e-6)

        numerator = torch.bmm(x, coef)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        bases = bases * numerator / (denominator + 1e-6)
        return bases, coef

    def compute_coef(self, x, bases, coef):
        numerator = torch.bmm(x.transpose(1, 2), bases)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        coef = coef * numerator / (denominator + 1e-6)
        return coef


class Hamburger(nn.Module):
    """Hamburger Module with matrix decomposition."""

    def __init__(self,
                 ham_channels=512,
                 ham_kwargs=dict(),
                 norm_cfg=None,
                 **kwargs):
        super().__init__()
        self.ham_in = ConvModule(
            ham_channels, ham_channels, 1, norm_cfg=None, act_cfg=None)
        self.ham = NMF2D(ham_kwargs)
        self.ham_out = ConvModule(
            ham_channels, ham_channels, 1, norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, x):
        enjoy = self.ham_in(x)
        enjoy = F.relu(enjoy, inplace=True)
        enjoy = self.ham(enjoy)
        enjoy = self.ham_out(enjoy)
        ham = F.relu(x + enjoy, inplace=True)
        return ham


# =========================
# Enhanced LightHamHead with HFP_FSRA
# =========================
@MODELS.register_module()
class EnhancedLightHamHead(BaseDecodeHead):
    """Enhanced LightHamHead with HFP_FSRA for improved segmentation accuracy.

    This enhanced version integrates HFP_FSRA (Frequency-Selective Residual Attention)
    to capture fine-grained details through frequency domain processing while maintaining
    the global modeling capability of the original Hamburger module.

    Args:
        ham_channels (int): Input channels for Hamburger. Defaults: 512.
        ham_kwargs (dict): kwargs for Hamburger. Defaults: dict().
        use_fsra (bool): Whether to use HFP_FSRA module. Defaults: True.
        fsra_kwargs (dict): kwargs for HFP_FSRA module. Defaults: dict().
        fsra_position (str): Position to apply FSRA ('before', 'after', 'both').
                            Defaults: 'after'.
    """

    def __init__(self,
                 ham_channels=512,
                 ham_kwargs=dict(),
                 use_fsra=True,
                 fsra_kwargs=dict(),
                 fsra_position='after',
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        self.ham_channels = ham_channels
        self.use_fsra = use_fsra
        self.fsra_position = fsra_position

        # Feature squeeze
        self.squeeze = ConvModule(
            sum(self.in_channels),
            self.ham_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        # FSRA before hamburger (for input enhancement)
        if self.use_fsra and self.fsra_position in ['before', 'both']:
            fsra_config = dict(
                in_channels=self.ham_channels,
                init_alpha=(0.2, 0.2),  # 较小的截止频率以保留更多细节
                init_beta=8.0,  # 较软的边界
                groups=min(32, self.ham_channels),
                dw_kernel=5,
                dw_dilation=1
            )
            fsra_config.update(fsra_kwargs)
            self.fsra_before = HFP_FSRA(**fsra_config)

        # Original hamburger module
        self.hamburger = Hamburger(ham_channels, ham_kwargs, **kwargs)

        # FSRA after hamburger (for output refinement)
        if self.use_fsra and self.fsra_position in ['after', 'both']:
            fsra_config = dict(
                in_channels=self.ham_channels,
                init_alpha=(0.3, 0.3),  # 稍大的截止频率用于后处理
                init_beta=12.0,  # 较锐利的边界
                groups=min(32, self.ham_channels),
                dw_kernel=7,  # 大核用于捕获更多空间信息
                dw_dilation=1
            )
            fsra_config.update(fsra_kwargs)
            self.fsra_after = HFP_FSRA(**fsra_config)

        # Feature alignment
        self.align = ConvModule(
            self.ham_channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        # Optional: Additional refinement for fine details
        if self.use_fsra:
            self.detail_enhance = nn.Sequential(
                ConvModule(
                    self.channels,
                    self.channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg),
                nn.Dropout2d(0.1)
            )

    def forward(self, inputs):
        """Forward function with enhanced feature processing."""
        inputs = self._transform_inputs(inputs)

        # Resize all inputs to the same size
        inputs = [
            resize(
                level,
                size=inputs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners) for level in inputs
        ]

        # Concatenate multi-scale features
        inputs = torch.cat(inputs, dim=1)

        # Feature squeeze
        x = self.squeeze(inputs)

        # Optional: Apply FSRA before hamburger for input enhancement
        if self.use_fsra and self.fsra_position in ['before', 'both']:
            x = self.fsra_before(x)

        # Apply hamburger module for global context modeling
        x = self.hamburger(x)

        # Optional: Apply FSRA after hamburger for output refinement
        if self.use_fsra and self.fsra_position in ['after', 'both']:
            x = self.fsra_after(x)

        # Feature alignment
        output = self.align(x)

        # Optional: Additional detail enhancement
        if self.use_fsra:
            output = self.detail_enhance(output)

        # Final classification
        output = self.cls_seg(output)
        return output


# Alias for backward compatibility
@MODELS.register_module()
class LightHamHeadWithFSRA(EnhancedLightHamHead):
    """Alias for EnhancedLightHamHead for backward compatibility."""
    pass


# =========================
# Configuration Examples
# =========================
def get_enhanced_lightham_configs():
    """Get different configuration examples for EnhancedLightHamHead."""

    # Configuration 1: FSRA after hamburger (recommended)
    config_after = dict(
        type='EnhancedLightHamHead',
        in_channels=[64, 128, 320, 512],  # 从backbone输入的通道数
        in_index=[0, 1, 2, 3],
        channels=128,
        num_classes=19,  # 根据数据集调整
        ham_channels=512,
        ham_kwargs=dict(
            MD_R=64,
            train_steps=6,
            eval_steps=7
        ),
        use_fsra=True,
        fsra_position='after',
        fsra_kwargs=dict(
            init_alpha=(0.25, 0.25),
            init_beta=12.0,
            groups=32,
            dw_kernel=7,
            dw_dilation=1
        ),
        dropout_ratio=0.1,
        align_corners=False
    )

    # Configuration 2: FSRA both before and after hamburger
    config_both = dict(
        type='EnhancedLightHamHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=128,
        num_classes=19,
        ham_channels=512,
        ham_kwargs=dict(
            MD_R=64,
            train_steps=6,
            eval_steps=7
        ),
        use_fsra=True,
        fsra_position='both',
        fsra_kwargs=dict(
            init_alpha=(0.2, 0.2),  # 更保守的高通设置
            init_beta=8.0,
            groups=32,
            dw_kernel=5,
            dw_dilation=1
        ),
        dropout_ratio=0.1,
        align_corners=False
    )

    # Configuration 3: Lightweight version
    config_light = dict(
        type='EnhancedLightHamHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=128,
        num_classes=19,
        ham_channels=256,  # 减少通道数
        ham_kwargs=dict(
            MD_R=32,  # 减少矩阵分解维度
            train_steps=4,
            eval_steps=5
        ),
        use_fsra=True,
        fsra_position='after',
        fsra_kwargs=dict(
            init_alpha=(0.3, 0.3),
            init_beta=10.0,
            groups=16,  # 减少组数
            dw_kernel=5,
            dw_dilation=1
        ),
        dropout_ratio=0.1,
        align_corners=False
    )

    return {
        'after_only': config_after,
        'both_sides': config_both,
        'lightweight': config_light
    }


if __name__ == "__main__":
    # Test the enhanced head
    torch.manual_seed(42)

    # Mock inputs from different backbone levels
    inputs = [
        torch.randn(2, 64, 64, 64),  # Level 0
        torch.randn(2, 128, 32, 32),  # Level 1
        torch.randn(2, 320, 16, 16),  # Level 2
        torch.randn(2, 512, 8, 8),  # Level 3
    ]

    # Create enhanced head
    head = EnhancedLightHamHead(
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=128,
        num_classes=19,
        ham_channels=512,
        use_fsra=True,
        fsra_position='after'
    )

    # Forward pass
    head.eval()
    with torch.no_grad():
        output = head(inputs)

    print("Enhanced LightHamHead with HFP_FSRA")
    print(f"Output shape: {output.shape}")
    print("Integration successful! 🎉")