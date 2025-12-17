import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import normal
from ..builder import LOSSES
from .utils import get_class_weight, weight_reduce_loss


@LOSSES.register_module()
class BlvLoss(nn.Module):
    def __init__(self, cls_num_list, sigma=4, loss_name='BlvLoss'):
        super(BlvLoss, self).__init__()
        # 不在初始化时指定设备，让它在forward时自动适应
        cls_list = torch.tensor(cls_num_list, dtype=torch.float32)
        frequency_list = torch.log(cls_list)
        total_sum = torch.tensor(sum(cls_num_list), dtype=torch.float32)
        self.register_buffer('frequency_list', torch.log(total_sum) - frequency_list)
        self.reduction = 'mean'
        self.sigma = sigma  # 添加这行保存sigma参数
        self._loss_name = loss_name

    def forward(self, pred, target, weight=None, ignore_index=255, avg_factor=None, reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        # 确保 frequency_list 在正确的设备上
        frequency_list = self.frequency_list.to(pred.device)

        # 创建正态分布采样器并采样
        sampler = normal.Normal(0, self.sigma)
        variation = sampler.sample(pred.shape).clamp(-1, 1).to(pred.device)

        # 应用变分到预测值
        variation_scaled = (variation.abs().permute(0, 2, 3, 1) / frequency_list.max() * frequency_list).permute(0, 3,
                                                                                                                 1, 2)
        pred_modified = pred + variation_scaled

        # 计算交叉熵损失
        loss = F.cross_entropy(pred_modified, target, reduction='none', ignore_index=ignore_index)

        # 确保loss是tensor
        assert isinstance(loss, torch.Tensor), f"Initial loss should be tensor, got {type(loss)}"

        # 应用权重
        if weight is not None:
            weight = weight.float()
            loss = loss * weight
            assert isinstance(loss, torch.Tensor), f"Loss after weighting should be tensor, got {type(loss)}"

        # 执行reduction
        if reduction == 'none':
            pass  # 保持原样
        elif reduction == 'mean':
            if avg_factor is not None and avg_factor > 0:
                # 确保avg_factor是数值
                if isinstance(avg_factor, torch.Tensor):
                    avg_factor = avg_factor.item()
                loss = loss.sum() / float(avg_factor)
            else:
                # 只对非ignore的像素计算平均
                valid_mask = (target != ignore_index).float()
                valid_count = valid_mask.sum()
                if valid_count > 0:
                    loss = loss.sum() / valid_count
                else:
                    loss = loss.sum()  # 避免除零
        elif reduction == 'sum':
            loss = loss.sum()

        # 最终确保返回tensor
        if not isinstance(loss, torch.Tensor):
            print(f"Warning: Converting {type(loss)} to tensor")
            loss = torch.tensor(float(loss), dtype=torch.float32, device=pred.device, requires_grad=True)

        # 确保loss有正确的形状和梯度
        if loss.dim() == 0:  # 标量tensor
            loss = loss.view(1)  # 转换为1维tensor

        assert isinstance(loss, torch.Tensor), f"Final loss must be tensor, got {type(loss)}"
        assert loss.requires_grad, f"Loss must require gradients, got requires_grad={loss.requires_grad}"

        return loss

    @property
    def loss_name(self):
        """Loss Name.
        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name