import torch
import torch.nn as nn
import torch.nn.functional as F


class IoULoss(nn.Module):
    def __init__(self, num_classes=4, epsilon=1e-6, reduction="mean"):
        super(IoULoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        assert reduction in ["none", "mean", "sum"], "Unsupported reduction method."
        self.reduction = reduction

    def forward(self, input, target):
        # 将 input 转换为 softmax 后的概率分布
        input_softmax = F.softmax(input, dim=1)

        # 将 target 转换为 one-hot 编码
        target_one_hot = torch.zeros(
            (target.shape[0], self.num_classes, *target.shape[1:]),
            dtype=target.dtype,
            device=target.device,
        )
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)

        # 计算每个类别的 IoU
        iou_losses = []
        for class_index in range(self.num_classes):
            iou_loss = self.iou_coefficient(
                input_softmax[:, class_index],
                target_one_hot[:, class_index],
                self.epsilon,
            )
            iou_losses.append(iou_loss)

        # 将每个类别的损失合并为一个张量
        iou_losses_tensor = torch.stack(iou_losses)

        # 根据 reduction 参数计算最终的损失
        if self.reduction == "none":
            return 1 - iou_losses_tensor  # 返回每个样本的平均损失
        elif self.reduction == "mean":
            return 1 - iou_losses_tensor.mean()  # 返回所有样本的平均损失
        elif self.reduction == "sum":
            return 1 - iou_losses_tensor.sum()  # 返回所有样本的总和损失

    def iou_coefficient(self, input, target, epsilon=1e-6):
        """计算单个类别的 IoU 系数"""
        smooth = epsilon
        intersection = torch.sum(input * target)
        union = torch.sum(input) + torch.sum(target) - intersection
        return (intersection + smooth) / (union + smooth)


# 示例用法
if __name__ == "__main__":
    # 假设我们有一个 4 类的分割任务
    n_classes = 4
    batch_size = 2
    height, width = 64, 64

    # 创建模拟的输入和目标
    input = torch.randn(batch_size, n_classes, height, width)
    target = torch.randint(0, n_classes, (batch_size, height, width))

    # 初始化损失函数
    criterion = IoULoss(num_classes=n_classes, reduction="mean")

    # 计算多分类 IoU Loss
    loss = criterion(input, target)
    print("Multi-class IoU Loss:", loss)
