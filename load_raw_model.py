import torch

from model.base_model import SegModel
from model.cls_head import LinearClsHead
from model.vrwkv import Vision_RWKV
from model.vvrwkv import VVision_RWKV
from utils import save_checkpoint

model = SegModel(
    backbone=VVision_RWKV(
        img_size=224,
        in_channels=3,
        patch_size=16,
        embed_dims=192,
        depth=12,
        drop_path_rate=0.3,
        out_indices=[2, 5, 8, 11],
        final_norm=True,
    ),
    # decode_head=UPerNet_1(
    #     num_classes=151,
    #     image_size=512,
    #     fc_dim=192,
    #     fpn_inplanes=[192, 192, 192, 192],
    #     fpn_dim=256,
    # ),
    decode_head=LinearClsHead(
        num_classes=1000,
        in_channels=[192, 192, 192, 192],
    ),
).cuda()

model_dict = model.state_dict()

raw_model_dict = torch.load("checkpoints/vvrwkv_t_in1k_224.pth", weights_only=False)[
    "state_dict"
]

model_key = set(model_dict.keys())
raw_model_key = set(raw_model_dict.keys())

same_key = model_key & raw_model_key
only_model_key = model_key - raw_model_key
only_raw_model_key = raw_model_key - model_key

same_key = sorted(list(same_key))
only_model_key = sorted(list(only_model_key))
only_raw_model_key = sorted(list(only_raw_model_key))

print(f"same_key: {same_key}")
print(f"only_model_key: {only_model_key}")
print(f"only_raw_model_key: {only_raw_model_key}")

for key in same_key:
    model_dict[key] = raw_model_dict[key]

for key in only_model_key:
    torch.nn.init.normal_(model_dict[key])

model.load_state_dict(model_dict)

save_checkpoint("checkpoints/vvrwkv_t_in1k_cls_convert.pth", model, None, 0, 0, 0)
