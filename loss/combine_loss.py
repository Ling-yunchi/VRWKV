import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from loss import IoULoss


class CombineLoss(nn.Module):
    def __init__(self, losses, weights=None):
        super(CombineLoss, self).__init__()

        # 确保 losses 是一个列表
        if not isinstance(losses, list):
            losses = [losses]

        # 初始化损失函数列表
        self.losses = nn.ModuleList(losses)

        # 初始化权重列表
        if weights is None:
            self.weights = [1.0] * len(losses)
        else:
            if len(weights) != len(losses):
                raise ValueError("Number of weights must match the number of losses.")
            self.weights = weights

    def forward(self, inputs, targets):
        total_loss = 0.0
        for i, loss_fn in enumerate(self.losses):
            loss_value = loss_fn(inputs, targets)

            # 加权损失值
            total_loss += self.weights[i] * loss_value

        return total_loss


# 示例用法
if __name__ == "__main__":
    # 假设我们有一个 4 类的分割任务
    n_classes = 4
    batch_size = 2
    height, width = 64, 64

    # 创建模拟的输入和目标
    input = torch.randn(batch_size, n_classes, height, width)
    target = torch.randint(0, n_classes, (batch_size, height, width))

    # 初始化损失函数实例
    cross_entropy_loss = CrossEntropyLoss(reduction="mean")
    iou_loss = IoULoss(num_classes=n_classes, reduction="mean")

    # 初始化组合损失函数
    combined_loss = CombineLoss(
        losses=[cross_entropy_loss, iou_loss], weights=[1.0, 0.5]
    )

    # 计算组合损失
    loss = combined_loss(input, target)
    print("Combined Loss:", loss.item())
