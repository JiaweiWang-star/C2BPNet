import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import normal

from ..builder import LOSSES

@LOSSES.register_module()
class GCLAngLoss(nn.Module):
    """GCL Angular Loss for mmsegmentation."""

    def __init__(self, cls_num_list, m=0.5, epsilon=0.1, weight=None, s=30,
                 easy_margin=False, train_cls=False, gamma=0.):
        super(GCLAngLoss, self).__init__()

        # 将类别数量转为 tensor，放到 GPU
        cls_list = torch.tensor(cls_num_list, dtype=torch.float32)
        m_list = torch.log(cls_list)
        m_list = m_list.max() - m_list
        self.m_list = m_list

        self.m = m
        self.epsilon = epsilon
        self.s = s
        self.weight = weight
        self.simpler = normal.Normal(0, 1 / 3)
        self.easy_margin = easy_margin
        self.train_cls = train_cls
        self.gamma = gamma

    def forward(self, cosine, target):
        """
        Args:
            cosine (Tensor): shape (B, C), logits before scaling
            target (Tensor): shape (B,), class indices
        Returns:
            Tensor: loss value (scalar)
        """
        # 构建 one-hot
        index = torch.zeros_like(cosine, dtype=torch.bool)
        index.scatter_(1, target.view(-1, 1), True)

        # 噪声扰动
        noise = self.simpler.sample(cosine.shape).clamp(-1, 1).to(cosine.device)
        noise = math.pi / 2 * self.epsilon / self.m_list.max() * self.m_list.to(cosine.device) * noise

        noise_m = noise.abs() + self.m
        m = torch.where(index, noise_m, noise.abs())

        cos_m = torch.cos(m)
        sin_m = torch.sin(m)

        th = torch.cos(math.pi - m)
        mm = torch.sin(math.pi - m)
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
        phi = cosine * cos_m - sine * sin_m

        if self.easy_margin:
            output = torch.where(cosine > 0, phi, cosine)
        else:
            output = torch.where(cosine > th, phi, cosine - mm)

        logits = self.s * output

        if self.train_cls:
            # 这里用交叉熵 + focal
            loss = F.cross_entropy(logits, target, reduction='none', weight=self.weight)
            # 如果 gamma>0，可以加 focal loss
            if self.gamma > 0:
                pt = torch.exp(-loss)
                loss = ((1 - pt) ** self.gamma * loss)
            loss = loss.mean()
        else:
            loss = F.cross_entropy(logits, target, weight=self.weight)

        # 保证返回 tensor
        if not isinstance(loss, torch.Tensor):
            loss = torch.tensor(loss, dtype=torch.float32, device=cosine.device)

        return loss
