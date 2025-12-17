import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from torch.nn.init import trunc_normal_
from mmcv.cnn import ConvModule
from mmengine.device import get_device

from mmseg.registry import MODELS
from ..utils import resize
from .decode_head import BaseDecodeHead

# Import natten CUDA kernels
FUSED = True
try:
    from natten.functional import na2d_qk, na2d_av, na2d
except ImportError:
    FUSED = False
    print("Warning: natten not installed, using fallback implementation")


# --------- utils ---------
def to_2tuple(x):
    return (x, x) if isinstance(x, int) else x


# --------- LayerNorm2d with custom autograd ---------
class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        # d(weight) and d(bias)
        d_weight = (grad_output * y).sum(dim=(0, 2, 3))
        d_bias = grad_output.sum(dim=(0, 2, 3))
        return gx, d_weight, d_bias, None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


# --------- Deformable Neighborhood Attention (with natten CUDA kernel) ---------
class DeformableNeighborhoodAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            kernel_size: int,
            dilation: int = 1,
            offset_range_factor=1.0,
            stride=1,
            use_pe=True,
            dwc_pe=True,
            no_off=False,
            fixed_pe=False,
            is_causal: bool = False,
            rel_pos_bias: bool = False,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        n_head_channels = dim // num_heads

        self.dwc_pe = dwc_pe
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = num_heads
        self.nc = n_head_channels * num_heads
        self.n_groups = num_heads
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.use_pe = use_pe
        self.fixed_pe = fixed_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor
        self.ksize = kernel_size
        self.kernel_size = to_2tuple(kernel_size)
        self.stride = stride
        self.dilation = dilation
        self.is_causal = is_causal

        kk = self.ksize
        pad_size = kk // 2 if kk != stride else 0

        # offset predictor (depthwise)
        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels,
                      kk, stride, pad_size, groups=self.n_group_channels),
            LayerNorm2d(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )
        if self.no_off:
            for m in self.conv_offset.parameters():
                m.requires_grad_(False)

        # qkv projections
        self.proj_q = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)

        # 2D relative position bias
        if rel_pos_bias:
            self.rpb = nn.Parameter(
                torch.zeros(
                    num_heads,
                    (2 * self.kernel_size[0] - 1),
                    (2 * self.kernel_size[1] - 1),
                )
            )
            trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)
        else:
            self.register_parameter("rpb", None)

        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        # depthwise conv as positional encoding
        self.rpe_table = nn.Conv2d(self.nc, self.nc, kernel_size=3, stride=1, padding=1, groups=self.nc)

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B*g, H, W, 2
        return ref

    def forward(self, x):
        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device

        # projections
        q = self.proj_q(x)  # [B, C, H, W]

        # offsets & sampling (deformable)
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        offset = self.conv_offset(q_off)  # [(B*g), 2, Hg, Wg]
        Hk, Wk = offset.size(2), offset.size(3)

        if self.offset_range_factor >= 0 and not self.no_off:
            offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        offset = einops.rearrange(offset, 'b p h w -> b h w p')  # [(B*g), Hg, Wg, 2]
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)  # [B*g, Hg, Wg, 2]

        if self.no_off:
            offset = offset.fill_(0.0)

        pos = offset + reference if self.offset_range_factor >= 0 else (offset + reference).clamp(-1., +1.)

        # Sample features using grid_sample or avg_pool
        if self.no_off:
            x_sampled = F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)
            assert x_sampled.size(2) == Hk and x_sampled.size(3) == Wk
        else:
            x_sampled = F.grid_sample(
                input=x.reshape(B * self.n_groups, self.n_group_channels, H, W),
                grid=pos[..., (1, 0)],  # y, x -> x, y
                mode='bilinear', align_corners=True
            )
            x_sampled = x_sampled.reshape(B, C, H, W)

        # relative positional encoding (depthwise)
        residual_lepe = self.rpe_table(q)

        # Use natten CUDA kernels for neighborhood attention
        if FUSED and self.rpb is None:
            # Fused kernel path (fastest)
            q = einops.rearrange(q, 'b (g c) h w -> b h w g c',
                                 g=self.n_groups, b=B, c=self.n_group_channels, h=H, w=W)
            k = einops.rearrange(self.proj_k(x_sampled), 'b (g c) h w -> b h w g c',
                                 g=self.n_groups, b=B, c=self.n_group_channels, h=H, w=W)
            v = einops.rearrange(self.proj_v(x_sampled), 'b (g c) h w -> b h w g c',
                                 g=self.n_groups, b=B, c=self.n_group_channels, h=H, w=W)

            out = na2d(
                q,
                k,
                v,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                is_causal=self.is_causal,
                rpb=self.rpb,
                scale=self.scale,
            )
            out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        else:
            # Unfused kernel path (with rpb support)
            q = einops.rearrange(q, 'b (g c) h w -> b g h w c',
                                 g=self.n_groups, b=B, c=self.n_group_channels, h=H, w=W)
            k = einops.rearrange(self.proj_k(x_sampled), 'b (g c) h w -> b g h w c',
                                 g=self.n_groups, b=B, c=self.n_group_channels, h=H, w=W)
            v = einops.rearrange(self.proj_v(x_sampled), 'b (g c) h w -> b g h w c',
                                 g=self.n_groups, b=B, c=self.n_group_channels, h=H, w=W)

            q = q * self.scale
            attn = na2d_qk(
                q,
                k,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                is_causal=self.is_causal,
                rpb=self.rpb,
            )
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            out = na2d_av(
                attn,
                v,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                is_causal=self.is_causal,
            )
            out = einops.rearrange(out, 'b g h w c -> b (g c) h w')

        if self.use_pe and self.dwc_pe:
            out = out + residual_lepe

        y = self.proj_drop(self.proj_out(out))
        return y


class Matrix_Decomposition_2D_Base(nn.Module):

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
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    def forward(self, x, return_bases=False):
        """Forward Function."""
        B, C, H, W = x.shape

        # (B, C, H, W) -> (B * S, D, N)
        D = C // self.S
        N = H * W
        x = x.view(B * self.S, D, N)
        if not self.rand_init and not hasattr(self, 'bases'):
            bases = self._build_bases(1, self.S, D, self.R, device=x.device)
            self.register_buffer('bases', bases)

        # (S, D, R) -> (B * S, D, R)
        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R, device=x.device)
        else:
            bases = self.bases.repeat(B, 1, 1)

        bases, coef = self.local_inference(x, bases)

        # (B * S, N, R)
        coef = self.compute_coef(x, bases, coef)

        # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        x = torch.bmm(bases, coef.transpose(1, 2))

        # (B * S, D, N) -> (B, C, H, W)
        x = x.view(B, C, H, W)

        return x


class NMF2D(Matrix_Decomposition_2D_Base):
    """Non-negative Matrix Factorization (NMF) module.

    It is inherited from ``Matrix_Decomposition_2D_Base`` module.
    """

    def __init__(self, args=dict()):
        super().__init__(**args)

        self.inv_t = 1

    def _build_bases(self, B, S, D, R, device=None):
        """Build bases in initialization."""
        if device is None:
            device = get_device()
        bases = torch.rand((B * S, D, R)).to(device)
        bases = F.normalize(bases, dim=1)

        return bases

    def local_step(self, x, bases, coef):
        """Local step in iteration to renew bases and coefficient."""
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ [(B * S, D, R)^T @ (B * S, D, R)] -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # Multiplicative Update
        coef = coef * numerator / (denominator + 1e-6)

        # (B * S, D, N) @ (B * S, N, R) -> (B * S, D, R)
        numerator = torch.bmm(x, coef)
        # (B * S, D, R) @ [(B * S, N, R)^T @ (B * S, N, R)] -> (B * S, D, R)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        # Multiplicative Update
        bases = bases * numerator / (denominator + 1e-6)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        """Compute coefficient."""
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ (B * S, D, R)^T @ (B * S, D, R) -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # multiplication update
        coef = coef * numerator / (denominator + 1e-6)

        return coef


class Hamburger(nn.Module):
    def __init__(self,
                 ham_channels=512,
                 ham_kwargs=dict(),
                 norm_cfg=None,
                 use_deformable_attention=False,
                 **kwargs):
        super().__init__()

        self.use_deformable_attention = use_deformable_attention

        self.ham_in = ConvModule(
            ham_channels, ham_channels, 1, norm_cfg=None, act_cfg=None)

        if self.use_deformable_attention:
            deformable_args = {
                'dim': ham_channels,
                'num_heads': ham_kwargs.get("num_heads", 8),
                'kernel_size': ham_kwargs.get("kernel_size", 7),
                'dilation': ham_kwargs.get("dilation", 1),
                'offset_range_factor': ham_kwargs.get("offset_range_factor", 1.0),
                'stride': 1,
                'use_pe': ham_kwargs.get("use_pe", True),
                'dwc_pe': ham_kwargs.get("dwc_pe", True),
                'no_off': ham_kwargs.get("no_off", False),
                'fixed_pe': ham_kwargs.get("fixed_pe", False),
                'is_causal': False,
                'rel_pos_bias': ham_kwargs.get("rel_pos_bias", True),
                'attn_drop': ham_kwargs.get("attn_drop", 0.0),
                'proj_drop': ham_kwargs.get("proj_drop", 0.0),
            }
            self.ham = DeformableNeighborhoodAttention(**deformable_args)
        else:
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


@MODELS.register_module()
class LightHamHeadwithdscs(BaseDecodeHead):

    def __init__(self,
                 ham_channels=512,
                 ham_kwargs=dict(),
                 use_deformable_attention=False,
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        self.ham_channels = ham_channels
        self.use_deformable_attention = use_deformable_attention

        self.squeeze = ConvModule(
            sum(self.in_channels),
            self.ham_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.hamburger = Hamburger(
            ham_channels,
            ham_kwargs,
            norm_cfg=self.norm_cfg,
            use_deformable_attention=use_deformable_attention
        )

        self.align = ConvModule(
            self.ham_channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)

        inputs = [
            resize(
                level,
                size=inputs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners) for level in inputs
        ]

        inputs = torch.cat(inputs, dim=1)
        # apply a conv block to squeeze feature map
        x = self.squeeze(inputs)
        # apply hamburger module
        x = self.hamburger(x)

        # apply a conv block to align feature map
        output = self.align(x)
        output = self.cls_seg(output)
        return output