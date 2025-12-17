# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmengine.model import BaseModule
from mmengine.model.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)

from mmseg.registry import MODELS


class FeatureToIntensity(nn.Module):
    """Convert feature maps to intensity (grayscale) representation.

    Args:
        in_channels (int): Number of input channels
        method (str): Conversion method - 'learned', 'mean', or 'max'
    """

    def __init__(self, in_channels, method='learned'):  # 默认改为 mean，减少参数
        super().__init__()
        self.method = method

        if method == 'learned':
            self.conv = nn.Conv2d(in_channels, 1, 1, bias=False)  # 去掉 bias
            nn.init.constant_(self.conv.weight, 1.0 / in_channels)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] feature map
        Returns:
            intensity_map: [B, 1, H, W] grayscale representation
        """
        if self.method == 'learned':
            intensity = self.conv(x)
        elif self.method == 'mean':
            intensity = x.mean(dim=1, keepdim=True)
        elif self.method == 'max':
            intensity = x.max(dim=1, keepdim=True)[0]
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # 归一化到 [0, 1]
        intensity = torch.sigmoid(intensity)
        return intensity


class GeoPriorGen_SC(nn.Module):
    def __init__(self, embed_dim, num_heads=8, initial_value=2, heads_range=4,
                 use_soft_contrast=True, intensity_method='max', gsa_downsample_ratio=2):
        super().__init__()
        self.intensity_method = intensity_method
        self.gsa_downsample_ratio = gsa_downsample_ratio

        # 角度位置编码
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.weight = nn.Parameter(torch.ones(2, 1, 1, 1), requires_grad=True)

        # 衰减系数
        decay = torch.log(
            1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads)
        )
        self.register_buffer("angle", angle)
        self.register_buffer("decay", decay)

        self.use_soft_contrast = use_soft_contrast
        if self.use_soft_contrast:
            self.sigma = nn.Parameter(torch.tensor(0.1), requires_grad=True)

    def generate_intensity_decay(self, H, W, intensity_grid):
        B, _, H, W = intensity_grid.shape
        grid_i = intensity_grid.reshape(B, H * W, 1)
        mask_i = grid_i[:, :, None, :] - grid_i[:, None, :, :]
        mask_i = (mask_i.abs()).sum(dim=-1)
        mask_i = mask_i.unsqueeze(1) * self.decay[None, :, None, None]
        return mask_i

    def generate_pos_decay(self, H, W):
        index_h = torch.arange(H).to(self.decay)
        index_w = torch.arange(W).to(self.decay)
        grid = torch.meshgrid(index_h, index_w, indexing='ij')
        grid = torch.stack(grid, dim=-1).reshape(H * W, 2)
        mask = grid[:, None, :] - grid[None, :, :]
        mask = (mask.abs()).sum(dim=-1)
        mask = mask * self.decay[:, None, None]
        return mask

    def generate_1d_intensity_decay(self, H, W, intensity_grid):
        mask = intensity_grid[:, :, :, :, None] - intensity_grid[:, :, :, None, :]
        mask = mask.abs()
        mask = mask * self.decay[:, None, None, None]
        return mask

    def generate_1d_decay(self, l):
        index = torch.arange(l).to(self.decay)
        mask = index[:, None] - index[None, :]
        mask = mask.abs()
        mask = mask * self.decay[:, None, None]
        return mask

    def generate_structural_contrast(self, intensity_map, threshold=0.05):
        B, _, H, W = intensity_map.shape

        # 使用分块计算减少显存
        chunk_size = min(H * W, 1024)  # 限制块大小
        i_flat = intensity_map.view(B, -1, 1)

        # 如果太大，使用稀疏近似
        if H * W > 2048:
            # 只计算对角线附近的区域
            return None  # 返回 None 表示不使用对比掩码

        diff = torch.abs(i_flat - i_flat.transpose(1, 2))

        if self.use_soft_contrast:
            contrast_mask = torch.exp(-diff ** 2 / (2 * self.sigma ** 2 + 1e-6))
        else:
            contrast_mask = (diff < threshold).float()

        return contrast_mask

    def forward(self, HW_tuple: Tuple[int], intensity_map, split_or_not=False):
        intensity_map = F.interpolate(intensity_map, size=HW_tuple, mode="bilinear", align_corners=False)

        index = torch.arange(HW_tuple[0] * HW_tuple[1]).to(self.decay)
        sin = torch.sin(index[:, None] * self.angle[None, :]).reshape(HW_tuple[0], HW_tuple[1], -1)
        cos = torch.cos(index[:, None] * self.angle[None, :]).reshape(HW_tuple[0], HW_tuple[1], -1)

        if split_or_not:
            mask_i_h = self.generate_1d_intensity_decay(HW_tuple[0], HW_tuple[1], intensity_map.transpose(-2, -1))
            mask_i_w = self.generate_1d_intensity_decay(HW_tuple[1], HW_tuple[0], intensity_map)

            mask_h = self.generate_1d_decay(HW_tuple[0])
            mask_w = self.generate_1d_decay(HW_tuple[1])

            mask_h = self.weight[0] * mask_h.unsqueeze(0).unsqueeze(2) + self.weight[1] * mask_i_h
            mask_w = self.weight[0] * mask_w.unsqueeze(0).unsqueeze(2) + self.weight[1] * mask_i_w

            contrast_mask = self.generate_structural_contrast(intensity_map)
            return (sin, cos), (mask_h, mask_w), contrast_mask
        else:
            mask = self.generate_pos_decay(HW_tuple[0], HW_tuple[1])
            mask_i = self.generate_intensity_decay(HW_tuple[0], HW_tuple[1], intensity_map)
            mask = self.weight[0] * mask + self.weight[1] * mask_i
            contrast_mask = self.generate_structural_contrast(intensity_map)

            if contrast_mask is not None:
                mask = mask.unsqueeze(0) + contrast_mask.unsqueeze(1)
            else:
                mask = mask.unsqueeze(0)

            return (sin, cos), mask, None


class DWConv2d(nn.Module):
    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        return x


def angle_transform(x, sin, cos):
    x1 = x[:, :, :, :, ::2]
    x2 = x[:, :, :, :, 1::2]
    return (x * cos) + (torch.stack([-x2, x1], dim=-1).flatten(-2) * sin)


class Full_GSA_SC(nn.Module):
    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)
        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def forward(self, x, rel_pos, split_or_not=False):
        bsz, h, w, _ = x.size()
        (sin, cos), mask, contrast_mask = rel_pos
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k = k * self.scaling
        q = q.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        k = k.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        qr = angle_transform(q, sin, cos)
        kr = angle_transform(k, sin, cos)
        vr = v.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)

        if split_or_not:
            # Height attention: [B, num_heads, w, h, dim]
            qr_h = qr.permute(0, 1, 3, 2, 4)
            kr_h = kr.permute(0, 1, 3, 2, 4)
            vr_h = vr.permute(0, 1, 3, 2, 4)
            attn_h = torch.matmul(qr_h, kr_h.transpose(-1, -2)) + mask[0]
            out_h = torch.matmul(torch.softmax(attn_h, -1), vr_h)
            #  [B, num_heads, h, w, dim]
            out_h = out_h.permute(0, 1, 3, 2, 4)

            # Width attention: [B, num_heads, h, w, dim]
            qr_w = qr
            kr_w = kr
            vr_w = vr
            attn_w = torch.matmul(qr_w, kr_w.transpose(-1, -2)) + mask[1]
            out_w = torch.matmul(torch.softmax(attn_w, -1), vr_w)

            out = (out_h + out_w) / 2
        else:
            qr = qr.flatten(2, 3)
            kr = kr.flatten(2, 3)
            vr = vr.flatten(2, 3)
            attn = torch.matmul(qr, kr.transpose(-1, -2)) + mask
            out = torch.matmul(torch.softmax(attn, -1), vr)

        out = out.transpose(1, 2).reshape(bsz, h, w, -1)
        out = out + lepe
        out = self.out_proj(out)
        return out

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)


class Mlp(BaseModule):
    """Multi Layer Perceptron (MLP) Module."""

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_cfg=dict(type='GELU'),
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = nn.Conv2d(
            hidden_features,
            hidden_features,
            3,
            1,
            1,
            bias=True,
            groups=hidden_features)
        self.act = build_activation_layer(act_cfg)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class StemConv(BaseModule):
    """Stem Block at the beginning of Semantic Branch."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels // 2,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1)),
            build_norm_layer(norm_cfg, out_channels // 2)[1],
            build_activation_layer(act_cfg),
            nn.Conv2d(
                out_channels // 2,
                out_channels,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1)),
            build_norm_layer(norm_cfg, out_channels)[1],
        )

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


class MSCAAttention(BaseModule):

    def __init__(self,
                 channels,
                 kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
                 paddings=[2, [0, 3], [0, 5], [0, 10]]):
        super().__init__()
        self.conv0 = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_sizes[0],
            padding=paddings[0],
            groups=channels)
        for i, (kernel_size,
                padding) in enumerate(zip(kernel_sizes[1:], paddings[1:])):
            kernel_size_ = [kernel_size, kernel_size[::-1]]
            padding_ = [padding, padding[::-1]]
            conv_name = [f'conv{i}_1', f'conv{i}_2']
            for i_kernel, i_pad, i_conv in zip(kernel_size_, padding_,
                                               conv_name):
                self.add_module(
                    i_conv,
                    nn.Conv2d(
                        channels,
                        channels,
                        tuple(i_kernel),
                        padding=i_pad,
                        groups=channels))
        self.conv3 = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)

        attn = attn + attn_0 + attn_1 + attn_2
        attn = self.conv3(attn)
        x = attn * u

        return x


class MSCASpatialAttention(BaseModule):

    def __init__(self,
                 in_channels,
                 attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
                 attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
                 act_cfg=dict(type='GELU'),
                 use_full_gsa_sc=False,
                 num_heads=8,
                 value_factor=1,
                 split_or_not=False,
                 use_soft_contrast=True,
                 intensity_method='max',
                 gsa_downsample_ratio=2):
        super().__init__()
        self.proj_1 = nn.Conv2d(in_channels, in_channels, 1)
        self.activation = build_activation_layer(act_cfg)
        self.spatial_gating_unit = MSCAAttention(in_channels,
                                                 attention_kernel_sizes,
                                                 attention_kernel_paddings)
        self.proj_2 = nn.Conv2d(in_channels, in_channels, 1)


        self.use_full_gsa_sc = use_full_gsa_sc
        self.gsa_downsample_ratio = gsa_downsample_ratio

        if self.use_full_gsa_sc:
            self.full_gsa = Full_GSA_SC(embed_dim=in_channels,
                                        num_heads=num_heads,
                                        value_factor=value_factor)
            self.geo_prior = GeoPriorGen_SC(embed_dim=in_channels,
                                            num_heads=num_heads,
                                            use_soft_contrast=use_soft_contrast,
                                            intensity_method=intensity_method,
                                            gsa_downsample_ratio=gsa_downsample_ratio)


            self.intensity_generator = FeatureToIntensity(in_channels, method=intensity_method)
            self.split_or_not = split_or_not


            if gsa_downsample_ratio > 1:
                self.downsample = nn.AvgPool2d(gsa_downsample_ratio, gsa_downsample_ratio)
            self.upsample = nn.Upsample(scale_factor=gsa_downsample_ratio, mode='bilinear', align_corners=False)

    def forward(self, x):
        """Forward function with memory optimization."""
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut


        if self.use_full_gsa_sc:
            B, C, H, W = x.shape

            if self.gsa_downsample_ratio > 1:
                x_down = self.downsample(x)
                _, _, H_down, W_down = x_down.shape
                intensity_map = self.intensity_generator(x_down)

                x_nhwc = x_down.permute(0, 2, 3, 1)
                rel = self.geo_prior((H_down, W_down), intensity_map, split_or_not=self.split_or_not)

                gsa_out = self.full_gsa(x_nhwc, rel, split_or_not=self.split_or_not)
                gsa_out = gsa_out.permute(0, 3, 1, 2)

                gsa_out = F.interpolate(gsa_out, size=(H, W), mode='bilinear', align_corners=False)
            else:
                intensity_map = self.intensity_generator(x)
                x_nhwc = x.permute(0, 2, 3, 1)
                rel = self.geo_prior((H, W), intensity_map, split_or_not=self.split_or_not)
                gsa_out = self.full_gsa(x_nhwc, rel, split_or_not=self.split_or_not)
                gsa_out = gsa_out.permute(0, 3, 1, 2)

            x = x + gsa_out

        return x


class MSCABlock(BaseModule):
    """Basic Multi-Scale Convolutional Attention Block."""

    def __init__(self,
                 channels,
                 attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
                 attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 use_full_gsa_sc=False,
                 num_heads=8,
                 value_factor=1,
                 split_or_not=False,
                 use_soft_contrast=True,
                 intensity_method='max',
                 gsa_downsample_ratio=2):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, channels)[1]
        self.attn = MSCASpatialAttention(
            channels,
            attention_kernel_sizes,
            attention_kernel_paddings,
            act_cfg,
            use_full_gsa_sc=use_full_gsa_sc,
            num_heads=num_heads,
            value_factor=value_factor,
            split_or_not=split_or_not,
            use_soft_contrast=use_soft_contrast,
            intensity_method=intensity_method,
            gsa_downsample_ratio=gsa_downsample_ratio)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg, channels)[1]
        mlp_hidden_channels = int(channels * mlp_ratio)
        self.mlp = Mlp(
            in_features=channels,
            hidden_features=mlp_hidden_channels,
            act_cfg=act_cfg,
            drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones(channels), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones(channels), requires_grad=True)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = x + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) *
            self.attn(self.norm1(x)))
        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) *
            self.mlp(self.norm2(x)))
        x = x.view(B, C, N).permute(0, 2, 1)
        return x


class OverlapPatchEmbed(BaseModule):
    """Image to Patch Embedding."""

    def __init__(self,
                 patch_size=7,
                 stride=4,
                 in_channels=3,
                 embed_dim=768,
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2)
        self.norm = build_norm_layer(norm_cfg, embed_dim)[1]

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


@MODELS.register_module()
class MSCANgcaa(BaseModule):
    def __init__(self,
                 in_channels=3,
                 embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[4, 4, 4, 4],
                 drop_rate=0.,
                 drop_path_rate=0.,
                 depths=[3, 4, 6, 3],
                 num_stages=4,
                 attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
                 attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 pretrained=None,
                 init_cfg=None,
                 use_full_gsa_sc=False,
                 gsa_stages=[2, 3],
                 num_heads=8,
                 value_factor=1,
                 split_or_not=True,
                 use_soft_contrast=True,
                 intensity_method='max',
                 gsa_downsample_ratio=2):
        super().__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.depths = depths
        self.num_stages = num_stages
        self.gsa_stages = gsa_stages

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = StemConv(3, embed_dims[0], norm_cfg=norm_cfg)
            else:
                patch_embed = OverlapPatchEmbed(
                    patch_size=7 if i == 0 else 3,
                    stride=4 if i == 0 else 2,
                    in_channels=in_channels if i == 0 else embed_dims[i - 1],
                    embed_dim=embed_dims[i],
                    norm_cfg=norm_cfg)

            # 只在指定 stage 启用 GSA
            use_gsa_this_stage = use_full_gsa_sc and (i in gsa_stages)

            block = nn.ModuleList([
                MSCABlock(
                    channels=embed_dims[i],
                    attention_kernel_sizes=attention_kernel_sizes,
                    attention_kernel_paddings=attention_kernel_paddings,
                    mlp_ratio=mlp_ratios[i],
                    drop=drop_rate,
                    drop_path=dpr[cur + j],
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    use_full_gsa_sc=use_gsa_this_stage,
                    num_heads=num_heads,
                    value_factor=value_factor,
                    split_or_not=split_or_not,
                    use_soft_contrast=use_soft_contrast,
                    intensity_method=intensity_method,
                    gsa_downsample_ratio=gsa_downsample_ratio) for j in range(depths[i])
            ])
            norm = nn.LayerNorm(embed_dims[i])
            cur += depths[i]

            setattr(self, f'patch_embed{i + 1}', patch_embed)
            setattr(self, f'block{i + 1}', block)
            setattr(self, f'norm{i + 1}', norm)

    def init_weights(self):
        """Initialize modules of MSCAN."""
        print('init cfg', self.init_cfg)
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super().init_weights()

    def forward(self, x):
        """Forward function with auto-generated intensity maps."""
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f'patch_embed{i + 1}')
            block = getattr(self, f'block{i + 1}')
            norm = getattr(self, f'norm{i + 1}')
            x, H, W = patch_embed(x)

            for blk in block:
                x = blk(x, H, W)

            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs
