from typing import List

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            # initialize RevIN params: (C,)
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, norm: bool):
        h, w = x.shape[-2:]
        x = rearrange(x, "b c h w -> b (h w) c")
        if norm:
            self._get_statistics(x)
            x = self._normalize(x)
        else:
            x = self._denormalize(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


class DenseMoE(nn.Module):
    def __init__(
        self,
        in_channels,
        experts: List[nn.Module],
        drop=0.1,
        revin_affine=True,
    ):
        """
        Dense MoE for Image
        :param in_channels: input image channels
        :param experts: num of expert, each expert input shape equal to output shape
        :param drop: drop rate
        :param revin_affine: use revin learnable param
        """
        super(DenseMoE, self).__init__()
        self.num_experts = len(experts)
        self.drop = drop
        self.revin_affine = revin_affine

        self.gate = nn.Conv2d(in_channels, self.num_experts, 1)
        self.experts = nn.ModuleList(experts)

        self.dropout = nn.Dropout2d(drop)
        self.rev = RevIN(in_channels, affine=revin_affine)

    def forward(self, x):
        x = self.rev(x, True)
        x = self.dropout(x)  # (B, C, H, W)

        score = F.softmax(self.gate(x), dim=1) # (B, E, H, W)
        score = score.unsqueeze(1)  # (B, 1, E, H, W)

        expert_outputs = torch.stack(
            [self.experts[i](x) for i in range(self.num_experts)], dim=2
        )  # (B, C, E, H, W)

        prediction = torch.sum(expert_outputs * score, dim=2)  # [B, C, H, W]

        prediction = self.rev(prediction, False)

        return prediction


if __name__ == "__main__":
    x = torch.randn(2, 3, 32, 32)
    experts = [nn.Conv2d(3, 3, 1) for _ in range(2)]
    model = DenseMoE(3, experts)
    print(model(x).shape)
