import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, num_classes=4, epsilon=1e-6, reduction="mean"):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        assert reduction in ["none", "mean", "sum"], "Unsupported reduction method."
        self.reduction = reduction

    def forward(self, input, target):
        # 将 input 转换为 one-hot 编码
        input_one_hot = torch.zeros(
            (input.shape[0], self.num_classes, *input.shape[2:]),
            dtype=input.dtype,
            device=input.device,
        )
        input_one_hot.scatter_(1, input.argmax(dim=1, keepdim=True), 1)

        # 将 target 转换为 one-hot 编码
        target_one_hot = torch.zeros(
            (target.shape[0], self.num_classes, *target.shape[1:]),
            dtype=target.dtype,
            device=target.device,
        )
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)

        # 计算每个类别的 Dice Loss
        dice_losses = []
        for class_index in range(self.num_classes):
            dice_loss = self.dice_coefficient(
                input_one_hot[:, class_index],
                target_one_hot[:, class_index],
                self.epsilon,
            )
            dice_losses.append(1 - dice_loss)

        # 将每个类别的损失合并为一个张量
        dice_losses_tensor = torch.stack(dice_losses)

        # 根据 reduction 参数计算最终的损失
        if self.reduction == "none":
            return dice_losses_tensor.mean(dim=1)  # 返回每个样本的平均损失
        elif self.reduction == "mean":
            return dice_losses_tensor.mean()  # 返回所有样本的平均损失
        elif self.reduction == "sum":
            return dice_losses_tensor.sum()  # 返回所有样本的总和损失

    def dice_coefficient(self, input, target, epsilon=1e-6):
        """计算单个类别的 Dice 系数"""
        smooth = epsilon
        intersection = torch.sum(input * target)
        union = torch.sum(input) + torch.sum(target)
        return (2.0 * intersection + smooth) / (union + smooth)


# 示例用法
if __name__ == "__main__":
    # 假设我们有一个 4 类的分割任务
    n_classes = 2
    batch_size = 1
    height, width = 2, 2

    # 创建模拟的输入和目标
    input = torch.randn(batch_size, n_classes, height, width)
    target = torch.randint(0, n_classes, (batch_size, height, width))

    # 初始化损失函数
    criterion = DiceLoss(num_classes=n_classes)

    # 计算多分类 Dice Loss
    loss = criterion(input, target)
    print("Multi-class Dice Loss:", loss.item())
