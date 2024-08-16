import torch
from einops import rearrange

from model.vrwkv import VRWKV_ChannelMix

model = VRWKV_ChannelMix(3, 1, 1, 2).cuda()
x = torch.randn(1, 14 * 14, 3).cuda()
criterion = torch.nn.CrossEntropyLoss()
output = model(x, (14, 14))
output = rearrange(output, "b t c -> b c t")
loss = criterion(output, torch.randint(0, 3, (1, 196)).cuda())
loss.backward()
