from torch import nn


class SegModel(nn.Module):
    def __init__(self, backbone, decode_head):
        super(SegModel, self).__init__()
        self.backbone = backbone
        self.decode_head = decode_head

    def forward(self, x):
        out = self.backbone(x)
        out = self.decode_head(out)
        return out