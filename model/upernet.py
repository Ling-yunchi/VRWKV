import torch
import torch.nn as nn
import torch.nn.functional as F

from model.vrwkv import to_2tuple


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
