import glob
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SameTransform:
    def __init__(self, transform):
        self.transforms = transform

    def __call__(self, img, mask):
        seed = torch.randint(0, 2**32, (1,)).item()  # 随机种子
        random.seed(seed)
        torch.manual_seed(seed)
        img = self.transforms(img)
        random.seed(seed)
        torch.manual_seed(seed)
        mask = self.transforms(mask)
        return img, mask


class HRSID_Dataset(Dataset):
    CLASSES = [0, 1]
    CLASS_NAMES = ["Bg", "Boat"]

    def __init__(self, root_path, train=True, transform=None):
        self.root_path = root_path
        self.train = train
        self.transform = SameTransform(transform) if transform else None
        self.train_ratio = 0.8

        split = "train" if train else "test"

        self.image_dir = os.path.join(root_path, "images", split)
        self.annotation_dir = os.path.join(root_path, "annotations", split)
        self.image_files = sorted(glob.glob(os.path.join(self.image_dir, "*.jpg")))
        self.annotation_files = sorted(
            glob.glob(os.path.join(self.annotation_dir, "*.png"))
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        ann_path = self.annotation_files[idx]

        # 加载图像和标注
        image = Image.open(img_path).convert("RGB")
        annotation = Image.open(ann_path).convert("L")  # 'L' 表示灰度图像

        if self.transform:
            image, annotation = self.transform(image, annotation)

        image = torch.as_tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1)
        annotation = torch.as_tensor(np.array(annotation), dtype=torch.int64)

        return image, annotation
