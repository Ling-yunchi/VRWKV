import torch

from model.adapter import VRWKV_Adapter
from model.upernet import UPerNet
from model.vrwkv import HWC_RWKV

# model = UPerNet(
#     encoder=VRWKV_Adapter(
#         224,
#         64,
#         4,
#         4,
#         0.0,
#         [[3, 7, 11, 15]],
#         True,
#         0.25,
#         1.0,
#         True,
#         True,
#         False,
#         in_channels=3,
#         depth=16,
#         embed_dims=256,
#         out_indices=[3, 7, 11, 15],
#     ),
#     num_classes=21,
#     feature_channels=[256, 256, 256, 256],
#     img_size=224,
# ).cuda()
model = UPerNet(
    # encoder=VRWKV_Adapter(
    #     224,
    #     64,
    #     4,
    #     4,
    #     0.0,
    #     [[3, 7, 11, 15]],
    #     True,
    #     0.25,
    #     1.0,
    #     True,
    #     True,
    #     False,
    #     patch_size=16,
    #     in_channels=3,
    #     depth=16,
    #     embed_dims=384,
    #     out_indices=[3, 7, 11, 15]
    # ),
    encoder=HWC_RWKV(
        in_channels=3,
        depth=16,
        embed_dims=384,
        out_indices=[3, 7, 11, 15]),
    num_classes=21,
    feature_channels=[384, 384, 384, 384],
).cuda()
x = torch.randn(1, 3, 224, 224).cuda()
criterion = torch.nn.CrossEntropyLoss()
output = model(x)
loss = criterion(output, torch.randint(0, 3, (1, 224, 224)).cuda())
loss.backward()
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(total_params)
print(output.shape)
print(loss)
