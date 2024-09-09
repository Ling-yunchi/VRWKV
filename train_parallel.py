import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.HYPSO1 import HYPSO1_PNG_Dataset
from model.base_model import SegModel
from model.unet_rwkv import UNetRWKV, UNetDecoder
from utils import (
    create_run_dir,
    load_checkpoint,
    save_checkpoint,
    draw_confusion_matrix,
    draw_normalized_confusion_matrix,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train_data_transform(image, mask):
    # # 随机水平翻转
    # if torch.rand(1) < 0.5:
    #     image, mask = F.hflip(image), F.hflip(mask)

    # # 随机垂直翻转
    # if torch.rand(1) < 0.5:
    #     image, mask = F.vflip(image), F.vflip(mask)

    # # 随机旋转
    # angle = RandomRotation.get_params(degrees=[-90, 90])
    # image, mask = F.rotate(image, angle), F.rotate(mask, angle, fill=255)

    # # 随机颜色抖动
    # color_jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    # image = color_jitter(image)

    target_size = (224, 224)
    # if torch.rand(1) < 0.1 and min(image.size) >= target_size[0]:
    #     i, j, h, w = RandomCrop.get_params(image, output_size=target_size)
    #     image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)
    # else:
    #     # 直接调整大小
    #     image = F.resize(image, target_size)
    #     mask = F.resize(mask, target_size, interpolation=F.InterpolationMode.NEAREST)

    image = F.resize(image, target_size)
    mask = F.resize(mask, target_size, interpolation=F.InterpolationMode.NEAREST)

    # to tensor
    image = F.to_tensor(image)
    mask = torch.from_numpy(np.array(mask)).long()

    return image, mask


def test_data_transform(image, mask):
    target_size = (224, 224)

    image = F.resize(image, target_size)
    mask = F.resize(mask, target_size, interpolation=F.InterpolationMode.NEAREST)

    # to tensor
    image = F.to_tensor(image)
    mask = torch.from_numpy(np.array(mask)).long()

    return image, mask


def main(rank, world_size):
    setup(rank, world_size)
    # set_seed(114514)

    # 设置每个进程使用的GPU
    torch.cuda.set_device(rank)

    train_dataset = HYPSO1_PNG_Dataset(
        "data/HYPSO1Dataset", train=True, transforms=train_data_transform
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )
    train_loader = DataLoader(train_dataset, batch_size=16, sampler=train_sampler)

    if rank == 0:
        test_dataset = HYPSO1_PNG_Dataset(
            "data/HYPSO1Dataset", train=False, transforms=test_data_transform
        )
        # test_sampler = torch.utils.data.distributed.DistributedSampler(
        #     test_dataset, num_replicas=world_size, rank=rank
        # )
        test_loader = DataLoader(test_dataset, batch_size=16)

    model_path = None

    # 初始化模型和损失函数
    model = SegModel(
        backbone=UNetRWKV(
            in_channels=3,
            depth=4,
            embed_dims=[64, 128, 256, 512],
            out_indices=[0, 1, 2, 3],
        ),
        decode_head=UNetDecoder(
            num_classes=21,
            image_size=224,
            feature_channels=[64, 128, 256, 512],
        ),
    ).cuda()

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if model_path is not None:
        load_checkpoint(model_path, model, optimizer)

    para_model = nn.parallel.DistributedDataParallel(
        model, device_ids=[rank], find_unused_parameters=True
    )

    num_iters = 40000
    val_interval = 1000

    global_step = 0

    if rank == 0:
        save_dir = "./checkpoints/voc_unet_rwkv"
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
            outputs = para_model(images)

            # 计算损失
            loss = criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            loss_sum = torch.zeros(1, dtype=torch.float32).cuda()
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss_sum += loss.item()
            avg_loss = loss_sum / world_size

            if rank == 0:
                predictions = torch.argmax(outputs, dim=1)
                accuracy = (predictions == labels).sum().item() / predictions.numel()

                writer.add_scalar("Train/Loss", avg_loss.item(), iter_count)
                writer.add_scalar("Train/Accuracy", accuracy, iter_count)

                process.set_description(
                    f"loss: {loss.item()}, accuracy: {accuracy*100:.4f}%"
                )
                process.update(1)

            iter_count += 1

            if iter_count % val_interval == 0:
                if rank == 0:
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
                            val_images, val_labels = (
                                val_images.cuda(),
                                val_labels.cuda(),
                            )

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
                    pixel_accuracy = np.sum(intersection) / np.sum(confusion).astype(
                        np.float32
                    )

                    # 使用类名输出IoU和准确度
                    class_iou = {
                        train_dataset.CLASS_NAMES[i]: IoU[i] for i in range(len(IoU))
                    }
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
                    print(
                        f"Iteration {iter_count}, Pixel Accuracy: {pixel_accuracy:.4f}"
                    )
                    # 计算平均IoU
                    mean_IoU = np.mean(IoU)
                    print(f"Iteration {iter_count}, Mean IoU: {mean_IoU:.4f}")

                    writer.add_scalar("Validation/MeanIoU", mean_IoU, iter_count)
                    writer.add_scalar(
                        "Validation/PixelAccuracy", pixel_accuracy, iter_count
                    )
                    writer.add_scalars("Validation/IoU", class_iou, iter_count)
                    writer.add_scalars(
                        "Validation/ClassAccuracy", class_accuracy, iter_count
                    )

                    # draw confusion matrix
                    fig = draw_confusion_matrix(confusion, train_dataset.CLASS_NAMES)
                    writer.add_figure("Validation/ConfusionMatrix", fig, iter_count)

                    fig = draw_normalized_confusion_matrix(
                        confusion, train_dataset.CLASS_NAMES
                    )
                    writer.add_figure(
                        "Validation/NormalizedConfusionMatrix", fig, iter_count
                    )

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
                        with open(f"{save_dir}/best_model.txt", "w") as file:
                            file.write(
                                f"Best model at iteration {iter_count} with mean IoU: {best_mean_IoU:.4f}"
                            )

                    # 重置混淆矩阵
                    confusion.fill(0)

                    # 切换回训练模式
                    model.train()

                torch.cuda.empty_cache()

                dist.barrier()

            if iter_count >= num_iters:
                break

    if rank == 0:
        process.close()
        writer.close()

    cleanup()


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"

    world_size = n_gpus
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)
