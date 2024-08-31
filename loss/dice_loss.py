import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_index=None, reduction="mean"):
        """
        Args:
            smooth (float): 用于平滑损失计算，防止分母为0.
            ignore_index (int or None): 需要忽略的类别索引.
            reduction (str): 选择损失聚合方式，可选 'mean'、'sum' 或 'none'.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        assert reduction in ["none", "mean", "sum"], "Unsupported reduction method."
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): 预测值，形状为 (batch_size, C, H, W)。
            targets (torch.Tensor): 真实标签，形状为 (batch_size, H, W)。
        """
        num_classes = inputs.shape[1]  # 获取类别数
        inputs = F.softmax(inputs, dim=1)  # 对预测值应用softmax

        # 初始化每个类别的Dice系数
        dice_per_class = []

        for c in range(num_classes):
            if self.ignore_index is not None and c == self.ignore_index:
                continue  # 跳过需要忽略的类别

            # 获取当前类别的预测和实际标签
            inputs_flat = inputs[:, c].contiguous().view(-1)
            targets_flat = (targets == c).float().contiguous().view(-1)

            # 计算交集和并集
            intersection = (inputs_flat * targets_flat).sum()
            dice = (2.0 * intersection + self.smooth) / (
                inputs_flat.sum() + targets_flat.sum() + self.smooth
            )

            # 记录每个类别的Dice系数
            dice_per_class.append(1 - dice)

        # 将损失转换为tensor
        dice_per_class = torch.stack(dice_per_class)

        # 根据reduction类型处理损失
        if self.reduction == "mean":
            return dice_per_class.mean()
        elif self.reduction == "sum":
            return dice_per_class.sum()
        else:
            return dice_per_class


# 示例用法
if __name__ == "__main__":
    # 假设我们有一个 4 类的分割任务
    n_classes = 21
    batch_size = 10
    height, width = 224, 224

    # 创建模拟的输入和目标
    input = torch.randn(batch_size, n_classes, height, width)
    target = torch.randint(0, n_classes, (batch_size, height, width))

    boundary_width = 1
    target[:, :boundary_width, :] = 255  # Top boundary
    target[:, -boundary_width:, :] = 255  # Bottom boundary
    target[:, :, :boundary_width] = 255  # Left boundary
    target[:, :, -boundary_width:] = 255  # Right boundary

    # 初始化损失函数
    criterion = DiceLoss(reduction="none")

    # 计算多分类 Dice Loss
    loss = criterion(input, target)
    print("Multi-class Dice Loss:", loss)
