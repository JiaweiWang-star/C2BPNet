# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmengine.model import BaseModule
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

from mmseg.registry import MODELS
from ..utils import ResLayer


class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', nn.BatchNorm2d(b))
        # 初始化 BN 权重
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)


class LKP_Dual(nn.Module):
    """双大核路径：7x7 (中程依赖) + 11x11 (远程依赖)"""

    def __init__(self, dim, sks=3, groups=8):
        super().__init__()
        # 确保groups能够整除dim
        groups = min(groups, dim)
        while dim % groups != 0 and groups > 1:
            groups -= 1

        self.cv1 = Conv2d_BN(dim, dim // 2)
        self.act = nn.ReLU()

        # 两个不同大小的卷积路径
        self.cv7 = Conv2d_BN(dim // 2, dim // 2, ks=7, pad=3, groups=dim // 2)
        self.cv11 = Conv2d_BN(dim // 2, dim // 2, ks=11, pad=5, groups=dim // 2)

        # 融合后生成动态卷积权重
        self.cv_out = nn.Conv2d(dim, sks ** 2 * dim, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=sks ** 2 * dim)


        self.sks = sks
        self.groups = groups
        self.dim = dim

    def forward(self, x):
        x = self.act(self.cv1(x))

        # 双路径
        x7 = self.cv7(x)
        x11 = self.cv11(x)

        # 在通道维度融合
        x_cat = torch.cat([x7, x11], dim=1)

        # 生成动态权重
        w = self.norm(self.cv_out(x_cat))  # [B, sks^2 * C, H, W]
        B, _, H, W = w.size()
        w = w.view(B, self.groups, self.dim // self.groups, self.sks ** 2, H, W)
        return w


class SKA_Small(nn.Module):
    """小核动态聚合：3x3 小核展开 + 动态加权"""

    def __init__(self, sks=3, groups=8):
        super().__init__()
        self.sks = sks
        self.groups = groups
        self.pad = (sks - 1) // 2

    def forward(self, x, w):
        B, C, H, W = x.shape
        g = self.groups
        Cg = C // g

        # 提取局部patch (3x3)
        patches = F.unfold(x, kernel_size=self.sks, padding=self.pad)
        patches = patches.view(B, g, Cg, self.sks ** 2, H, W)  # [B,g,Cg,K^2,H,W]

        # 动态加权
        out = (patches * w).sum(dim=3)  # [B,g,Cg,H,W]
        out = out.view(B, C, H, W)
        return out


class BDConv(nn.Module):
    """改进版 BDConv：双大核路径 + 小核动态聚合"""

    def __init__(self, dim, sks=3, groups=8):
        super().__init__()
        # 确保groups能够整除dim
        groups = min(groups, dim)
        while dim % groups != 0 and groups > 1:
            groups -= 1

        self.lkp_dual = LKP_Dual(dim, sks=sks, groups=groups)
        self.ska_small = SKA_Small(sks=sks, groups=groups)
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        return self.bn(self.ska_small(x, self.lkp_dual(x))) + x


class BDConvWrapper(nn.Module):
    """BDConv包装器，兼容标准卷积接口"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        # 通道对齐
        if in_channels != out_channels:
            self.channel_align = nn.Conv2d(
                in_channels, out_channels, 1, stride=stride, bias=False)
            self.bn_align = nn.BatchNorm2d(out_channels)
        else:
            self.channel_align = None
            self.bn_align = None

        # 步长处理
        if stride > 1:
            self.downsample = nn.AvgPool2d(kernel_size=stride, stride=stride)
        else:
            self.downsample = None

        # BDConv核心模块
        # 确保groups参数合理
        bdconv_groups = min(8, out_channels)
        while out_channels % bdconv_groups != 0 and bdconv_groups > 1:
            bdconv_groups -= 1

        self.bdconv = BDConv(out_channels, sks=3, groups=bdconv_groups)

    def forward(self, x):
        # 通道对齐
        if self.channel_align is not None:
            x = self.channel_align(x)
            x = self.bn_align(x)

        # 下采样
        if self.downsample is not None:
            x = self.downsample(x)

        # BDConv处理
        return self.bdconv(x)


class BasicBlockBD(BaseModule):
    """使用BDConv的BasicBlock"""

    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        assert dcn is None, 'DCN not supported with BDConv yet.'
        assert plugins is None, 'Plugins not supported with BDConv yet.'

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        # 使用BDConv替换标准卷积
        self.bdconv1 = BDConvWrapper(
            inplanes, planes, kernel_size=3, stride=stride, padding=dilation)
        self.bdconv2 = BDConvWrapper(
            planes, planes, kernel_size=3, stride=1, padding=1)

        self.add_module(self.norm1_name, norm1)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        def _inner_forward(x):
            identity = x

            out = self.bdconv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.bdconv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)
        return out


class BottleneckBD(BaseModule):
    """使用BDConv的Bottleneck"""

    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        assert style in ['pytorch', 'caffe']
        assert dcn is None, 'DCN not supported with BDConv yet.'
        assert plugins is None, 'Plugins not supported with BDConv yet.'

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)

        # 使用标准卷积进行通道变换
        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)

        # 使用BDConv替换3x3卷积
        self.bdconv2 = BDConvWrapper(
            planes, planes, kernel_size=3, stride=self.conv2_stride,
            padding=dilation)
        self.add_module(self.norm2_name, norm2)

        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def forward(self, x):
        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.bdconv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)
        return out


@MODELS.register_module()
class ResNetBD(BaseModule):
    """使用BDConv的ResNet backbone"""

    arch_settings = {
        18: (BasicBlockBD, (2, 2, 2, 2)),
        34: (BasicBlockBD, (3, 4, 6, 3)),
        50: (BottleneckBD, (3, 4, 6, 3)),
        101: (BottleneckBD, (3, 4, 23, 3)),
        152: (BottleneckBD, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=64,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 use_bdconv_stages=(True, True, True, True),
                 with_cp=False,
                 zero_init_residual=True,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(init_cfg)

        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')

        self.pretrained = pretrained
        self.zero_init_residual = zero_init_residual
        self.use_bdconv_stages = use_bdconv_stages

        # 处理初始化配置
        block_init_cfg = None
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'

        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(type='Constant', val=1, layer=['_BatchNorm', 'GroupNorm'])
                ]
        else:
            raise TypeError('pretrained must be a str or None')

        # 设置基本参数
        self.depth = depth
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval

        # 获取block类型和stage配置
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels

        # 构建stem层
        self._make_stem_layer(in_channels, stem_channels)

        # 构建ResNet层
        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            planes = base_channels * 2 ** i

            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                init_cfg=block_init_cfg)

            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        # 计算特征维度
        self.feat_dim = self.block.expansion * base_channels * 2 ** (len(self.stage_blocks) - 1)

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        """Make stem layer for ResNet."""
        if self.deep_stem:
            self.stem = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels)[1],
                nn.ReLU(inplace=True))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        """Freeze stages param and norm stats."""
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer freezed."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


@MODELS.register_module()
class ResNetBDV1c(ResNetBD):
    """使用BDConv的ResNetV1c变体"""

    def __init__(self, **kwargs):
        super().__init__(deep_stem=True, avg_down=False, **kwargs)


@MODELS.register_module()
class ResNetBDV1d(ResNetBD):
    """使用BDConv的ResNetV1d变体"""

    def __init__(self, **kwargs):
        super().__init__(deep_stem=True, avg_down=True, **kwargs)