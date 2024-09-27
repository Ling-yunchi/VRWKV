import torch

# from model.adapter import VRWKV_Adapter
from model.base_model import SegModel
from model.cls_head import LinearClsHead
from model.rwkvsr import RWKVNet
from model.unet_rwkv import UNetRWKV, UNetDecoder
from model.upernet import UPerNet, UPerNet_1
from model.vrwkv import Vision_RWKV
from utils import load_backbone

model = SegModel(
    # backbone=VRWKV_Adapter(
    #     224,
    #     64,
    #     4,
    #     4,
    #     0.0,
    #     [[0, 3], [4, 7], [8, 11], [12, 15]],
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
    # backbone=Vision_RWKV(
    #     in_channels=3, depth=16, embed_dims=384, out_indices=[3, 7, 11, 15]
    # ),
    # backbone=UNetRWKV(
    #     in_channels=3, depth=4, embed_dims=[64, 128, 256, 512], out_indices=[0, 1, 2, 3]
    # ),
    # decode_head=UPerNet(
    #     num_classes=21,
    #     feature_channels=[256, 256, 256, 256],
    #     img_size=224,
    # ),
    # decode_head=UPerNet_1(
    #     num_classes=21,
    #     image_size=224,
    #     fc_dim=384,
    #     fpn_inplanes=[384, 384, 384, 384],
    #     fpn_dim=256,
    # ),
    # decode_head=UNetDecoder(
    #     num_classes=21,
    #     image_size=64,
    #     feature_channels=[64, 128, 256, 512],
    # ),
    # decode_head=LinearClsHead(num_classes=3,in_channels=[64, 128, 256, 512])
    backbone=RWKVNet(
        img_size=224,
        in_channels=3,
        n_feats=[64, 128, 256, 512],
        out_slices=[0, 1, 2, 3],
        patch_size=8,
    ),
    decode_head=UPerNet_1(
        num_classes=151,
        image_size=224,
        fc_dim=512,
        fpn_inplanes=[64, 128, 256, 512],
        fpn_dim=256,
    ),
    # backbone=Vision_RWKV(
    #     img_size=224,
    #     in_channels=3,
    #     patch_size=16,
    #     embed_dims=768,
    #     depth=12,
    #     drop_path_rate=0.3,
    #     out_indices=[2, 5, 8, 11],
    #     final_norm=True,
    # ),
    # decode_head=UPerNet_1(
    #     num_classes=151,
    #     image_size=224,
    #     fc_dim=768,
    #     fpn_inplanes=[768, 768, 768, 768],
    #     fpn_dim=256,
    # ),
).cuda()

backbone_path = "checkpoints/model_50000.pth"

if backbone_path:
    load_backbone(backbone_path, model)

x = torch.randn(1, 3, 224, 224).cuda()
target = torch.randint(0, 151, (1, 224, 224)).cuda()
# target = torch.randint(0,3,(1,)).cuda()

criterion = torch.nn.CrossEntropyLoss()
output = model(x)

loss = criterion(output, target)
loss.backward()

print(model)
total_params = sum(p.numel() for p in model.parameters())
print(total_params)
print(output.shape)
print(loss)

unused_params = [
    name
    for name, param in model.named_parameters()
    if param.grad is None or param.grad.abs().sum() == 0
]
print(unused_params)
