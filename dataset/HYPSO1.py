import glob
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class HYPSO1_PNG_Dataset(Dataset):
    CLASSES = [0, 1, 2]
    CLASS_NAMES = ["Sea", "Land", "Cloud"]

    def __init__(self, root_path, train=True, transforms=None):
        self.root_path = root_path
        self.train = train
        self.transforms = transforms
        self.train_ratio = 0.8

        split = "train" if train else "validation"

        self.image_dir = os.path.join(root_path, "images", split)
        self.annotation_dir = os.path.join(root_path, "annotations", split)
        self.image_files = sorted(
            glob.glob(os.path.join(self.image_dir, "*.png"))
        )
        self.annotation_files = sorted(
            glob.glob(os.path.join(self.annotation_dir, "*.png"))
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        ann_path = self.annotation_files[idx]

        # 加载图像和标注
        image = Image.open(img_path).convert('RGB')
        annotation = Image.open(ann_path).convert('L')  # 'L' 表示灰度图像

        if self.transforms:
            image, annotation = self.transforms(image, annotation)

        image = torch.as_tensor(
            np.array(image), dtype=torch.float32).permute(2, 0, 1)
        annotation = torch.as_tensor(np.array(annotation), dtype=torch.int64)

        return image, annotation
