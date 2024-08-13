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
    def __init__(self, encoder, feature_channels=[256, 256, 256, 256], num_classes=3, img_size=224):
        super(UPerNet, self).__init__()
        self.encoder = encoder
        self.img_size = to_2tuple(img_size)

        # 多尺度特征融合模块
        self.mffm = nn.ModuleList([FFN(c, c) for c in feature_channels])

        # 全局上下文模块
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_channels[-1],
                      feature_channels[-1], kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels[-1],
                      feature_channels[-1], kernel_size=1),
        )

        # 解码头
        self.decode_head = nn.Sequential(
            nn.Conv2d(sum(feature_channels), 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(self, x):
        # 编码器特征
        features = self.encoder(x)

        # 多尺度特征融合
        fused_features = []
        for i, feature in enumerate(features):
            fused_feature = self.mffm[i](feature)
            if i > 0:
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
            decode_input,
            size=self.img_size,
            mode="bilinear",
            align_corners=False
        )
        decode_output = self.decode_head(decode_input)

        return decode_output
