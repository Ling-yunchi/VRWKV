import torch
import torchvision.transforms.functional as F
from torchvision.transforms import (
    RandomCrop,
    ColorJitter,
    RandomRotation,
)

import matplotlib.pyplot as plt
from dataset.HYPSO1 import HYPSO1_PNG_Dataset


def transforms(image, mask):
    # 随机水平翻转
    if torch.rand(1) < 0.5:
        image, mask = F.hflip(image), F.hflip(mask)

    # 随机垂直翻转
    if torch.rand(1) < 0.5:
        image, mask = F.vflip(image), F.vflip(mask)

    # 随机旋转
    angle = RandomRotation.get_params(degrees=[-20, 20])
    image, mask = F.rotate(image, angle), F.rotate(mask, angle, fill=255)

    # 随机颜色抖动
    color_jitter = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
    image = color_jitter(image)

    target_size = (224, 224)
    if torch.rand(1) < 0.5 and min(image.size) >= target_size[0]:
        i, j, h, w = RandomCrop.get_params(image, output_size=target_size)
        image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)
    else:
        # 直接调整大小
        image = F.resize(image, target_size)
        mask = F.resize(mask, target_size, interpolation=F.InterpolationMode.NEAREST)

    # 随机裁剪
    # i, j, h, w = RandomCrop.get_params(image, output_size=(224, 224))
    # image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)

    # Resize
    # image = F.resize(image, (224, 224))
    # mask = F.resize(mask, (224, 224), interpolation=F.InterpolationMode.NEAREST)

    return image, mask


if __name__ == "__main__":
    dataset = HYPSO1_PNG_Dataset("data/HYPSO1Dataset", transforms=transforms)
    # draw 10 samples
    plt.figure(figsize=(20, 4))
    for i in range(10):
        image, mask = dataset[i]
        image = F.to_pil_image(image)
        mask *= 64
        mask = F.to_pil_image(mask.int())
        plt.subplot(2, 10, i + 1)
        plt.imshow(image)
        plt.axis("off")
        plt.subplot(2, 10, i + 11)
        plt.imshow(mask, cmap="gray")
        plt.axis("off")
    plt.show()
