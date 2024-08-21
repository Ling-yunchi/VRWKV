import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, weight=None, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(
            weight=self.weight, ignore_index=self.ignore_index
        )

    def forward(self, preds, labels):
        logpt = -self.ce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss


# Example usage
if __name__ == "__main__":
    # Define parameters
    batch_size = 3
    num_classes = 5
    height = 10
    width = 10

    # Create example tensors with some invalid pixels (255)
    output = torch.randn(batch_size, num_classes, height, width, requires_grad=True)
    target = torch.randint(
        low=0, high=num_classes, size=(batch_size, height, width), dtype=torch.long
    )

    # Initialize the FocalLoss with ignore_index
    criterion = FocalLoss()

    # Compute the loss
    loss = criterion(output, target)
    print(f"Loss: {loss.item()}")
