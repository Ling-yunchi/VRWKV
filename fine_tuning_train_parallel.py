import os

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

from dataset.ADE20KSegmentation import ADE20KSegmentation
from model.base_model import SegModel
from model.unet_rwkv import UNetRWKV, UNetDecoder
from utils import (
    create_run_dir,
    load_checkpoint,
    save_checkpoint,
    load_backbone,
    xavier_init,
    save_script,
    random_hflip,
    set_seed,
    resize,
    random_resized_crop,
)
from utils.transforms import random_resize, random_crop, normalize, color_jitter, pad


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def get_backbone_head_params(model):
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if name.startswith("backbone"):
            backbone_params.append(param)
        else:
            head_params.append(param)
    return backbone_params, head_params


def train_data_transform(image, mask):
    image, mask = random_resized_crop(image, mask, scale=(0.5, 2.0), ratio=(0.75, 1.33), crop_size=(512, 512))

    image, mask = random_hflip(image, mask, 0.5)

    image = color_jitter(image, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)

    image = F.to_tensor(image)
    mask = torch.from_numpy(np.array(mask)).long()

    image = normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    return image, mask


def test_data_transform(image, mask):
    image, mask = resize(image, mask, (512, 512))

    # to tensor
    image = F.to_tensor(image)
    mask = torch.from_numpy(np.array(mask)).long()

    image = normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    return image, mask


"""
RUN NOTE:

"""


def main(rank, world_size):
    setup(rank, world_size)

    # 设置每个进程使用的GPU
    torch.cuda.set_device(rank)

    train_dataset = ADE20KSegmentation(
        "data/ADEChallengeData2016",
        mode="train",
        transforms=train_data_transform,
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)
    class_names = ["unknown"] + list(train_dataset.classes)

    test_dataset = ADE20KSegmentation(
        "data/ADEChallengeData2016",
        mode="val",
        transforms=test_data_transform,
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=world_size, rank=rank
    )
    test_loader = DataLoader(test_dataset, batch_size=32, sampler=test_sampler)

    # choose one path
    model_path = None
    backbone_path = None
    # set_seed(114514)

    lr_rate = 0.001
    head_lr_rate = lr_rate * 10

    # 初始化模型和损失函数
    model = SegModel(
        backbone=UNetRWKV(
            in_channels=3,
            depth=4,
            embed_dims=[64, 128, 256, 512],
            out_indices=[0, 1, 2, 3],
        ),
        decode_head=UNetDecoder(
            num_classes=151,
            image_size=224,
            feature_channels=[64, 128, 256, 512],
        ),
    ).cuda()

    class_radio = np.array(train_dataset.ratio)
    weight = 1 / np.log(class_radio + 1)
    weight = np.insert(weight, 0, 0)
    weight = torch.tensor(weight, dtype=torch.float32).cuda()
    criterion = nn.CrossEntropyLoss(weight, ignore_index=0)

    if backbone_path is not None:
        load_backbone(backbone_path, model)
        model.decode_head.apply(xavier_init)

    if model_path is not None:
        load_checkpoint(model_path, model)

    b_param, h_param = get_backbone_head_params(model)
    optimizer = optim.Adam(
        [{"params": b_param}, {"params": h_param, "lr": head_lr_rate}],
        lr=lr_rate,
        weight_decay=0.001,
    )

    para_model = nn.parallel.DistributedDataParallel(
        model, device_ids=[rank], find_unused_parameters=True
    )

    num_iters = 40000
    val_interval = 1000

    if rank == 0:
        save_dir = "./checkpoints/ade20k_unet_rwkv"
        save_dir = create_run_dir(save_dir)
        save_script(save_dir, __file__)
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
                    save_checkpoint(
                        f"{save_dir}/model_{iter_count}.pth",
                        model,
                        optimizer,
                        loss.item(),
                        accuracy,
                        iter_count,
                    )

                # 验证阶段
                para_model.eval()
                with torch.no_grad():
                    # 初始化混淆矩阵
                    class_num = 151
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
                    ground_truth_set = confusion.sum(axis=1)
                    predicted_set = confusion.sum(axis=0)
                    union = ground_truth_set + predicted_set - intersection
                    IoU = intersection / union.astype(np.float32)
                    pixel_accuracy = np.sum(intersection) / np.sum(confusion).astype(
                        np.float32
                    )

                    # 使用类名输出IoU和准确度
                    class_iou = {class_names[i]: IoU[i] for i in range(len(IoU))}
                    class_accuracy = {
                        class_names[i]: (
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

                    if mean_IoU > best_mean_IoU:
                        best_mean_IoU = mean_IoU
                        save_checkpoint(
                            f"{save_dir}/best_model.pth",
                            model,
                            optimizer,
                            loss.item(),
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
