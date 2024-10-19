import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageNet
from tqdm import tqdm

from model.base_model import SegModel
from model.cls_head import LinearClsHead
from model.vrwkv import Vision_RWKV
from utils import load_checkpoint

norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(**norm_cfg),
    ]
)

test_dataset = ImageNet("data/ImageNet", split="val", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32)

model_path = None

# 初始化模型和损失函数
model = SegModel(
    backbone=Vision_RWKV(
        img_size=224,
        in_channels=3,
        patch_size=16,
        embed_dims=256,
        depth=12,
        drop_path_rate=0.3,
        out_indices=[2, 5, 8, 11],
        final_norm=True,
    ),
    decode_head=LinearClsHead(
        num_classes=1000,
        in_channels=[256, 256, 256, 256],
    ),
).cuda()

if model_path is not None:
    load_checkpoint(model_path, model)

model.eval()
total_samples = 0
correct_predictions = 0
val_process = tqdm(
    test_loader,
    desc="Validation",
    leave=False,
)
with torch.no_grad():
    for val_images, val_labels in test_loader:
        # 将数据移到GPU
        val_images, val_labels = (
            val_images.cuda(),
            val_labels.cuda(),
        )

        # 前向传播
        val_outputs = model(val_images)

        # 计算混淆矩阵
        predictions = torch.argmax(val_outputs, dim=1).cpu().numpy().flatten()
        true_labels = val_labels.cpu().numpy().flatten()
        correct_predictions += np.sum(predictions == true_labels)

        total_samples += len(val_labels)

        val_process.update(1)

    accuracy = correct_predictions / total_samples
    print(f"Iteration Top1Acc: {accuracy:.4f}")
