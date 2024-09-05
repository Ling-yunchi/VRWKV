import torch
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation

from model.base_model import SegModel
from model.unet_rwkv import UNetRWKV, UNetDecoder

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

pretrained = ""

model.load_state_dict(torch.load(pretrained))

model.eval()

test_dataset = VOCSegmentation(root="data", image_set="val")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

voc_palette = [
    [0, 0, 0],  # Background
    [128, 0, 0],  # Aeroplane
    [0, 128, 0],  # Bicycle
    [128, 128, 0],  # Bird
    [0, 0, 128],  # Boat
    [128, 0, 128],  # Bottle
    [0, 128, 128],  # Bus
    [128, 128, 128],  # Car
    [64, 0, 0],  # Cat
    [192, 0, 0],  # Chair
    [64, 128, 0],  # Cow
    [192, 128, 0],  # Dining table
    [64, 0, 128],  # Dog
    [192, 0, 128],  # Horse
    [64, 128, 128],  # Motorbike
    [192, 128, 128],  # Person
    [0, 64, 0],  # Potted plant
    [128, 64, 0],  # Sheep
    [0, 192, 0],  # Sofa
    [128, 192, 0],  # Train
    [0, 64, 128],  # TV/monitor
    [255, 255, 255],  # Boundary (255)
]
hex_palette = ["#{:02x}{:02x}{:02x}".format(r, g, b) for r, g, b in voc_palette]
cmap = ListedColormap(hex_palette, N=22)

num_samples_to_show = 10
fig, axs = plt.subplots(
   3, num_samples_to_show, figsize=(5 * num_samples_to_show, 15)
)

for i, (img, mask) in enumerate(test_loader):
    _img = img.cuda()
    pred = model(_img)
    pred = torch.argmax(pred, dim=1)
    pred = pred.cpu().numpy()

    img = img.numpy().permute(1, 2, 0)
    mask = mask.numpy()

    mask[mask == 255] = 21

    axs[0, i].imshow(img)
    axs[0, i].set_title("Image")
    axs[0, i].axis("off")

    axs[1, i].imshow(mask, cmap=cmap)
    axs[1, i].set_title("Ground Truth")
    axs[1, i].axis("off")

    axs[2, i].imshow(pred, cmap=cmap)
    axs[2, i].set_title("Prediction")
    axs[2, i].axis("off")

    if i == num_samples_to_show - 1:
        break

plt.savefig("val_model.png")


