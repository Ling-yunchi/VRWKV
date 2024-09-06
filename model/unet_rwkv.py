import torch
import torch.nn as nn
import torch.nn.functional as F

from model.vrwkv import Block


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNetRWKV(nn.Module):
    def __init__(
        self,
        in_channels=3,
        depth=4,
        embed_dims=None,
        out_indices=None,
    ):
        """
        :param in_channels:
        :param depth: net depth
        :param embed_dims: layer embedding dimensions, default to [64, 128, ..., 64*2^(depth-1)]
        :param out_indices: indices of layers whose output should be returned, default to all layers
        """
        super(UNetRWKV, self).__init__()
        self.in_channels = in_channels
        self.depth = depth

        if out_indices is None:
            out_indices = [i for i in range(depth)]
        self.out_indices = out_indices

        if embed_dims is None:
            embed_dims = [64 * (2**i) for i in range(depth)]

        # 输入层
        self.inc = DoubleConv(in_channels, embed_dims[0])

        # 下采样层
        down_layers = []
        for i in range(1, depth):
            down_layers.append(Down(embed_dims[i - 1], embed_dims[i]))
        self.downs = nn.ModuleList(down_layers)

        # rwkv层
        rwkv_layers = []
        for i in range(depth):
            rwkv_layers.append(
                Block(
                    n_embd=embed_dims[i],
                    n_layer=depth,
                    layer_id=i,
                    hidden_rate=1,
                    key_norm=False,
                )
            )
        self.rwkvs = nn.ModuleList(rwkv_layers)

    def forward(self, x):
        xs = [self.rwkvs[0](self.inc(x))]  # 存储各个下采样层的输出
        for down, rwkv in zip(self.downs, self.rwkvs[1:]):
            x = down(xs[-1])
            x = rwkv(x)
            xs.append(x)

        return [xs[i] for i in self.out_indices]


class UNetDecoder(nn.Module):
    def __init__(
        self, num_classes, image_size=224, feature_channels=None, bilinear=False
    ):
        super(UNetDecoder, self).__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        self.bilinear = bilinear

        # 上采样层
        up_layers = []
        for i in reversed(range(1, len(feature_channels))):
            in_dim = feature_channels[i]  # if bilinear else feature_channels[i] * 2
            out_dim = feature_channels[i - 1]
            up_layers.append(Up(in_dim, out_dim, bilinear))
        self.ups = nn.ModuleList(up_layers)

        # 输出层
        self.outc = OutConv(feature_channels[0], num_classes)

    def forward(self, features):
        x = features[-1]
        for up, x_down in zip(self.ups, reversed(features[:-1])):
            x = up(x, x_down)
        x = F.interpolate(x, size=self.image_size, mode="bilinear", align_corners=False)
        logits = self.outc(x)
        return logits
