import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # 将 logits 转换为概率
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)

        # 获取每个样本的真实类别对应的概率
        probs = probs.gather(1, targets.unsqueeze(1))
        log_probs = log_probs.gather(1, targets.unsqueeze(1))

        # 计算负对数似然损失
        ce_loss = -log_probs

        # 计算权重因子
        weight = (1.0 - probs) ** self.gamma
        fl = self.alpha * weight * ce_loss

        if self.reduction == 'mean':
            return fl.mean()
        elif self.reduction == 'sum':
            return fl.sum()
        else:
            return fl