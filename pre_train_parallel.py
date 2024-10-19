import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import ImageNet
from tqdm import tqdm

from model.base_model import SegModel
from model.cls_head import LinearClsHead
from model.unet_rwkv import UNetRWKV
from utils import (
    create_run_dir,
    load_checkpoint,
    save_checkpoint,
    save_script,
)


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(**norm_cfg),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(**norm_cfg),
    ]
)

"""
RUN NOTE:

"""


def main(rank, world_size):
    setup(rank, world_size)

    # 设置每个进程使用的GPU
    torch.cuda.set_device(rank)

    train_dataset = ImageNet("data/ImageNet", split="train", transform=train_transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)

    test_dataset = ImageNet("data/ImageNet", split="val", transform=test_transform)
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=world_size, rank=rank
    )
    test_loader = DataLoader(test_dataset, batch_size=32, sampler=test_sampler)

    model_path = None

    # 初始化模型和损失函数
    model = SegModel(
        backbone=UNetRWKV(
            in_channels=3,
            depth=4,
            embed_dims=[64, 128, 256, 512],
            out_indices=[0, 1, 2, 3],
        ),
        decode_head=LinearClsHead(
            num_classes=1000,
            in_channels=[64, 128, 256, 512],
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

    warm_up_iter = 5000
    scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warm_up_iter
    )
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, num_iters - warm_up_iter
    )

    if rank == 0:
        save_dir = "./checkpoints/pretrain_imagenet/unet_rwkv"
        save_dir = create_run_dir(save_dir)
        save_script(save_dir, __file__)
        writer = SummaryWriter(log_dir=save_dir)

        best_acc = 0.0

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

            if iter_count < warm_up_iter:
                scheduler_warmup.step()
            else:
                scheduler_cosine.step()

            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            avg_loss = loss.item() / world_size

            if rank == 0:
                predictions = torch.argmax(outputs, dim=1)
                accuracy = (predictions == labels).sum().item() / predictions.numel()

                writer.add_scalar("Train/Loss", avg_loss, iter_count)
                writer.add_scalar("Train/Accuracy", accuracy, iter_count)

                process.set_description(
                    f"loss: {avg_loss}, acc: {accuracy*100:.4f}%, lr: {scheduler_warmup.get_lr() if iter_count < warm_up_iter else scheduler_cosine.get_lr()}"
                )
                process.update(1)
            iter_count += 1

            if iter_count % val_interval == 0:
                if rank == 0:
                    save_checkpoint(
                        f"{save_dir}/model_{iter_count}.pth",
                        model,
                        optimizer,
                        avg_loss,
                        accuracy,
                        iter_count,
                    )

                # 验证阶段
                para_model.eval()
                with torch.no_grad():
                    # 初始化混淆矩阵
                    class_num = 1000
                    confusion = np.zeros((class_num, class_num))
                    if rank == 0:
                        val_process = tqdm(
                            range(len(test_loader)),
                            desc=f"val iter {iter_count}",
                            leave=False,
                        )

                    for val_images, val_labels in test_loader:
                        # 将数据移到GPU
                        val_images, val_labels = (
                            val_images.cuda(),
                            val_labels.cuda(),
                        )

                        # 前向传播
                        val_outputs = para_model(val_images)

                        # 计算混淆矩阵
                        predictions = (
                            torch.argmax(val_outputs, dim=1).cpu().numpy().flatten()
                        )
                        true_labels = val_labels.cpu().numpy().flatten()
                        confusion += confusion_matrix(
                            true_labels, predictions, labels=np.arange(class_num)
                        )

                        if rank == 0:
                            val_process.update(1)

                tensor_confusion = torch.from_numpy(confusion).cuda()
                dist.all_reduce(tensor_confusion, op=dist.ReduceOp.SUM)
                confusion = tensor_confusion.cpu().numpy()

                if rank == 0:
                    # 计算 IoU 和像素准确率
                    intersection = np.diag(confusion)
                    accuracy = np.sum(intersection) / np.sum(confusion).astype(
                        np.float32
                    )

                    print(f"Iteration {iter_count}, Acc: {accuracy:.4f}")

                    writer.add_scalar("Validation/Acc", accuracy, iter_count)

                    if accuracy > best_acc:
                        best_acc = accuracy
                        save_checkpoint(
                            f"{save_dir}/best_model.pth",
                            model,
                            optimizer,
                            loss.item(),
                            best_acc,
                            iter_count,
                        )
                        print(
                            f"Model saved at iteration {iter_count} with acc: {accuracy:.4f}"
                        )
                        with open(f"{save_dir}/best_model.txt", "w") as file:
                            file.write(
                                f"Best model at iteration {iter_count} with acc: {accuracy:.4f}"
                            )

                # 重置混淆矩阵
                confusion.fill(0)

                # 切换回训练模式
                para_model.train()

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
