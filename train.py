import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from dataset.HYPSO1 import HYPSO1_PNG_Dataset
from model.upernet import UPerNet
from model.vrwkv import HWC_RWKV
from utils import create_run_dir, load_checkpoint, save_checkpoint

data_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
    ]
)

train_dataset = HYPSO1_PNG_Dataset(
    "data/HYPSO1Dataset", train=True, transform=data_transforms
)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = HYPSO1_PNG_Dataset(
    "data/HYPSO1Dataset", train=False, transform=data_transforms
)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model_path = None

# 初始化模型和损失函数
model = UPerNet(encoder=HWC_RWKV(in_channels=3), num_classes=3).cuda()
criterion = nn.CrossEntropyLoss()  # 适用于多分类问题，如图像分割

optimizer = optim.Adam(model.parameters(), lr=0.001)

if model_path is not None:
    load_checkpoint(model_path, model, optimizer)

num_iters = 10000
val_interval = 100


save_dir = "./checkpoints/vrwkv_upernet"
save_dir = create_run_dir(save_dir)
writer = SummaryWriter(log_dir=save_dir)

best_mean_IoU = 0.0

process = tqdm(range(num_iters))

iter_count = 0
while iter_count < num_iters:
    for images, labels in train_loader:
        # 将数据移到GPU
        images, labels = images.cuda(), labels.cuda()

        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(images)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == labels).sum().item() / predictions.numel()

        writer.add_scalar("Train/Loss", loss.item(), iter_count)
        writer.add_scalar("Train/Accuracy", accuracy, iter_count)

        process.set_description(f"loss: {loss.item()}, accuracy: {accuracy*100:.4f}%")
        process.update(1)
        iter_count += 1

        if iter_count % val_interval == 0:
            # 验证阶段
            model.eval()
            with torch.no_grad():
                # 初始化混淆矩阵
                class_num = len(train_dataset.CLASSES)
                confusion = np.zeros((class_num, class_num))
                val_process = tqdm(
                    test_loader, desc=f"val iter {iter_count}", leave=False
                )

                for val_images, val_labels in val_process:
                    # 将数据移到GPU
                    val_images, val_labels = val_images.cuda(), val_labels.cuda()

                    # 前向传播
                    val_outputs = model(val_images)

                    # 计算混淆矩阵
                    predictions = (
                        torch.argmax(val_outputs, dim=1).cpu().numpy().flatten()
                    )
                    true_labels = val_labels.cpu().numpy().flatten()
                    confusion += confusion_matrix(
                        true_labels, predictions, labels=np.arange(class_num)
                    )
            # 计算 IoU 和像素准确率
            intersection = np.diag(confusion)
            ground_truth_set = confusion.sum(axis=1)
            predicted_set = confusion.sum(axis=0)
            union = ground_truth_set + predicted_set - intersection
            IoU = intersection / union.astype(np.float32)
            pixel_accuracy = np.sum(intersection) / np.sum(confusion).astype(np.float32)

            # 使用类名输出IoU和准确度
            class_iou = {train_dataset.CLASS_NAMES[i]: IoU[i] for i in range(len(IoU))}
            class_accuracy = {
                train_dataset.CLASS_NAMES[i]: (
                    (intersection[i] / ground_truth_set[i])
                    if ground_truth_set[i] > 0
                    else 0
                )
                for i in range(class_num)
            }
            print(f"Iteration {iter_count}, IoU per class: {class_iou}")
            print(f"Iteration {iter_count}, Class Accuracy: {class_accuracy}")
            print(f"Iteration {iter_count}, Pixel Accuracy: {pixel_accuracy:.4f}")
            # 计算平均IoU
            mean_IoU = np.mean(IoU)
            print(f"Iteration {iter_count}, Mean IoU: {mean_IoU:.4f}")

            writer.add_scalar("Validation/MeanIoU", mean_IoU, iter_count)
            writer.add_scalar("Validation/PixelAccuracy", pixel_accuracy, iter_count)
            writer.add_scalars("Validation/IoU", class_iou, iter_count)
            writer.add_scalars("Validation/ClassAccuracy", class_accuracy, iter_count)

            save_checkpoint(
                f"{save_dir}/model_{iter_count}.pth",
                model,
                optimizer,
                loss,
                mean_IoU,
                iter_count,
            )

            if mean_IoU > best_mean_IoU:
                best_mean_IoU = mean_IoU
                save_checkpoint(
                    f"{save_dir}/best_model.pth",
                    model,
                    optimizer,
                    loss,
                    mean_IoU,
                    iter_count,
                )
                print(
                    f"Model saved at iteration {iter_count} with mean IoU: {best_mean_IoU:.4f}"
                )

            # 重置混淆矩阵
            confusion.fill(0)
            # 切换回训练模式
            model.train()

        if iter_count >= num_iters:
            break

process.close()
writer.close()
