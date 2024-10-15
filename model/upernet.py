import warnings
from typing import Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers import to_2tuple


class FFN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.ffn(x)


class UPerNet(nn.Module):
    def __init__(
        self,
        feature_channels=[256, 256, 256, 256],
        num_classes=3,
        img_size=224,
    ):
        super(UPerNet, self).__init__()
        self.img_size = to_2tuple(img_size)

        # 多尺度特征融合模块
        self.mffm = nn.ModuleList([FFN(c, c) for c in feature_channels])

        # 全局上下文模块
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_channels[-1], feature_channels[-1], kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels[-1], feature_channels[-1], kernel_size=1),
        )

        # 解码头
        self.decode_head = nn.Sequential(
            nn.Conv2d(sum(feature_channels), 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(self, features):
        # 多尺度特征融合
        fused_features = []
        for i, feature in enumerate(features):
            fused_feature = self.mffm[i](feature)
            if i > 0:
                fused_feature = F.interpolate(
                    fused_feature,
                    size=features[0].size()[2:],
                    mode="bilinear",
                    align_corners=False,
                )
                fused_feature += fused_features[-1]
            fused_features.append(fused_feature)

        # 全局上下文模块
        global_context = self.global_context(fused_features[-1])
        global_context = F.interpolate(
            global_context,
            size=fused_features[0].size()[2:],
            mode="bilinear",
            align_corners=False,
        )
        fused_features[0] = fused_features[0] + global_context

        # 解码头
        decode_input = torch.cat(fused_features, dim=1)
        decode_input = F.interpolate(
            decode_input, size=self.img_size, mode="bilinear", align_corners=False
        )
        decode_output = self.decode_head(decode_input)

        return decode_output


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    """3x3 convolution + BN + relu"""
    return nn.Sequential(
        nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        ),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


# https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/master/mit_semseg/models/models.py#L499
class UPerNet_1(nn.Module):
    def __init__(
        self,
        num_classes=150,
        image_size=224,
        fc_dim=4096,
        use_softmax=False,
        pool_scales=(1, 2, 3, 6),
        fpn_inplanes=(256, 512, 1024, 2048),
        fpn_dim=256,
    ):
        """
        UPerNet初始化函数。

        :param num_classes: 输出类别的数量。
        :param image_size: 输入图像的目标大小。
        :param fc_dim: 全连接层的维度。
        :param use_softmax: 是否使用softmax激活函数。
        :param pool_scales: 金字塔池化模块(PPM)使用的不同尺度。
        :param fpn_inplanes: 特征金字塔网络(FPN)每个输入层的通道数。
        :param fpn_dim: FPN输出的通道数，决定了FPN模块内部处理的特征图的深度。
        """
        super(UPerNet_1, self).__init__()
        self.image_size = to_2tuple(image_size)
        self.use_softmax = use_softmax

        # PPM Module
        self.ppm_pooling = []
        self.ppm_conv = []

        for scale in pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(
                nn.Sequential(
                    nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                )
            )
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(
            fc_dim + len(pool_scales) * 512, fpn_dim, 1
        )

        # FPN Module
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:  # skip the top layer
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(fpn_dim),
                    nn.ReLU(inplace=True),
                )
            )
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1):  # skip the top layer
            self.fpn_out.append(
                nn.Sequential(
                    conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
                )
            )
        self.fpn_out = nn.ModuleList(self.fpn_out)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_classes, kernel_size=1),
        )

    def forward(self, conv_out):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(
                pool_conv(
                    nn.functional.interpolate(
                        pool_scale(conv5),
                        (input_size[2], input_size[3]),
                        mode="bilinear",
                        align_corners=False,
                    )
                )
            )
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)  # lateral branch

            f = nn.functional.interpolate(
                f, size=conv_x.size()[2:], mode="bilinear", align_corners=False
            )  # top-down branch
            f = conv_x + f

            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse()  # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(
                nn.functional.interpolate(
                    fpn_feature_list[i],
                    output_size,
                    mode="bilinear",
                    align_corners=False,
                )
            )
        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)

        x = nn.functional.interpolate(
            x, size=self.image_size, mode="bilinear", align_corners=False
        )

        if self.use_softmax:  # is True during inference
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)
        return x


# https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/models/decode_heads/uper_head.py#L13
class UPerHead(nn.Module):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(
        self,
        in_channels,
        dim_channels,
        num_classes,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        align_corners=False,
        image_size=224,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.channels = dim_channels
        self.num_classes = num_classes
        self.align_corners = align_corners
        self.image_size = to_2tuple(image_size)

        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            align_corners=self.align_corners,
        )
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
        )
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                inplace=False,
            )
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                inplace=False,
            )
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
        )

        self.conv_seg = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """

        # build laterals
        laterals = [
            lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode="bilinear",
                align_corners=self.align_corners,
            )

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)
        return feats

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        output = resize(
            output,
            size=self.image_size,
            mode="nearest",
        )
        return output


class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(
        self,
        pool_scales,
        in_channels,
        channels,
        align_corners,
    ):
        super().__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvModule(
                        self.in_channels,
                        self.channels,
                        1,
                    ),
                )
            )

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = resize(
                ppm_out,
                size=x.size()[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


class ConvModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        inplace: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.activate = nn.ReLU(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activate(x)
        return x


def resize(
    input,
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=None,
    warning=True,
):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if (
                    (output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1)
                    and (output_h - 1) % (input_h - 1)
                    and (output_w - 1) % (input_w - 1)
                ):
                    warnings.warn(
                        f"When align_corners={align_corners}, "
                        "the output would more aligned if "
                        f"input size {(input_h, input_w)} is `x+1` and "
                        f"out size {(output_h, output_w)} is `nx+1`"
                    )
    return F.interpolate(input, size, scale_factor, mode, align_corners)
