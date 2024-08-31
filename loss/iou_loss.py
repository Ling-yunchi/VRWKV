import torch
import torch.nn as nn
import torch.nn.functional as F


class IoULoss(nn.Module):
    def __init__(self, ignore_index=None, reduction="mean"):
        """
        IoU Loss for image segmentation tasks.

        Args:
            ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the input gradient.
            reduction (str, optional): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. Default: 'mean'.
        """
        super(IoULoss, self).__init__()
        self.ignore_index = ignore_index
        assert reduction in ["none", "mean", "sum"], "Unsupported reduction method."
        self.reduction = reduction

    def forward(self, input, target):
        """
        Forward pass of the loss function.

        Args:
            input (torch.Tensor): Predictions (logits) of shape (N, C, H, W).
            target (torch.Tensor): Ground truth labels of shape (N, H, W).

        Returns:
            torch.Tensor: Calculated IoU loss.
        """
        num_classes = input.shape[1]
        # Convert logits to probabilities using softmax
        input = F.softmax(input, dim=1)

        # Flatten the input and target tensors
        input_flat = input.permute(0, 2, 3, 1).reshape(-1, num_classes)
        target_flat = target.view(-1)

        # Create mask for ignoring index
        if self.ignore_index is not None:
            mask = target_flat != self.ignore_index
            input_flat = input_flat[mask]
            target_flat = target_flat[mask]

        # Convert target to one-hot format
        target_one_hot = F.one_hot(target_flat, num_classes=num_classes)

        # Calculate intersection and union
        intersection = (input_flat * target_one_hot).sum(dim=0)
        union = (input_flat + target_one_hot - input_flat * target_one_hot).sum(dim=0)

        # Compute IoU
        iou = intersection / (union + 1e-6)

        # Compute IoU loss
        iou_loss = 1 - iou

        # Apply reduction
        if self.reduction == "mean":
            return iou_loss.mean()
        elif self.reduction == "sum":
            return iou_loss.sum()
        else:
            return iou_loss


# 示例用法
if __name__ == "__main__":
    # 假设我们有一个 4 类的分割任务
    n_classes = 4
    batch_size = 2
    height, width = 64, 64

    # 创建模拟的输入和目标
    input = torch.randn(batch_size, n_classes, height, width)
    target = torch.randint(0, n_classes, (batch_size, height, width))

    boundary_width = 1
    target[:, :boundary_width, :] = 255  # Top boundary
    target[:, -boundary_width:, :] = 255  # Bottom boundary
    target[:, :, :boundary_width] = 255  # Left boundary
    target[:, :, -boundary_width:] = 255  # Right boundary

    # 初始化损失函数
    criterion = IoULoss(reduction="none", ignore_index=255)

    # 计算多分类 IoU Loss
    loss = criterion(input, target)
    print("Multi-class IoU Loss:", loss)
