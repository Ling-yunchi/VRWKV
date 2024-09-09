from typing import Tuple, List

import torch
from torch import nn


class LinearClsHead(nn.Module):
    def __init__(self, num_classes: int, in_channels: List[int], **kwargs):
        super(LinearClsHead, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        total_in_channels = sum(self.in_channels)
        self.fc = nn.Linear(total_in_channels, self.num_classes)

    def forward(self, features: Tuple[torch.Tensor]) -> torch.Tensor:
        # features: [(B, C1, H1, W1), (B, C2, H2, W2), ...]
        global_features = []

        for feature in features:
            # 对每个特征图应用全局平均池化
            pooled_feature = self.avg_pool(feature).flatten(1)
            global_features.append(pooled_feature)

        # 将所有特征融合成一个向量
        pre_logits = torch.cat(global_features, dim=1)  # (N, C_total)

        # 通过全连接层进行分类
        cls_score = self.fc(pre_logits)

        return cls_score
