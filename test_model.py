import torch

from model.adapter import VRWKV_Adapter
from model.upernet import UPerNet

model = UPerNet(encoder=VRWKV_Adapter(
    224, 64, 4, 4, 0.0, [[2, 5, 8, 11]], True, 0.25, 1.0, True, True, False, in_channels=3
), num_classes=3).cuda()
x = torch.randn(1, 3, 224, 224).cuda()
criterion = torch.nn.CrossEntropyLoss()
output = model(x)
loss = criterion(output, torch.randint(0, 3, (1, 224, 224)).cuda())
loss.backward()
print(output.shape)
print(loss)
