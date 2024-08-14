import torch
from model.upernet import UPerNet
from model.vrwkv import HWC_RWKV

x = torch.randn(16, 3, 224, 224).cuda()
model = UPerNet(encoder=HWC_RWKV(in_channels=3), num_classes=3).cuda()

output = model(x)
print(output.shape)
