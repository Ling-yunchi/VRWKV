import math
import os

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from PIL import Image
from einops import rearrange
from torch import nn
from torchvision import transforms

from model.base_model import SegModel
from model.cls_head import LinearClsHead
from model.layers import DropPath
from model.scan import (
    s_hw,
    s_wh,
    sr_hw,
    sr_wh,
    s_rhrw,
    sr_rhrw,
    s_rwrh,
    sr_rwrh,
)
from model.vrwkv import PatchEmbed, resize_pos_embed
from model.wkv import RUN_CUDA


def q_shift(input, shift_pixel=1, gamma=1 / 4, patch_resolution=None):
    assert gamma <= 1 / 4
    B, N, C = input.shape
    input = input.transpose(1, 2).reshape(
        B, C, patch_resolution[0], patch_resolution[1]
    )
    B, C, H, W = input.shape
    output = torch.zeros_like(input)
    output[:, 0 : int(C * gamma), :, shift_pixel:W] = input[
        :, 0 : int(C * gamma), :, 0 : W - shift_pixel
    ]
    output[:, int(C * gamma) : int(C * gamma * 2), :, 0 : W - shift_pixel] = input[
        :, int(C * gamma) : int(C * gamma * 2), :, shift_pixel:W
    ]
    output[:, int(C * gamma * 2) : int(C * gamma * 3), shift_pixel:H, :] = input[
        :, int(C * gamma * 2) : int(C * gamma * 3), 0 : H - shift_pixel, :
    ]
    output[:, int(C * gamma * 3) : int(C * gamma * 4), 0 : H - shift_pixel, :] = input[
        :, int(C * gamma * 3) : int(C * gamma * 4), shift_pixel:H, :
    ]
    output[:, int(C * gamma * 4) :, ...] = input[:, int(C * gamma * 4) :, ...]
    return output.flatten(2).transpose(1, 2)


class VVRWKV_SpatialMix(nn.Module):
    def __init__(
        self,
        n_embd,
        n_layer,
        layer_id,
        shift_mode="q_shift",
        channel_gamma=1 / 4,
        shift_pixel=1,
        init_mode="fancy",
        key_norm=False,
        with_cp=False,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.device = None
        self.attn_sz = n_embd
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode

        # MoE System
        self.num_experts = 4

        self._init_weights(init_mode)

        if shift_pixel > 0:
            self.shift_func = eval(shift_mode)
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_v = None
            self.spatial_mix_r = None

        self.key = nn.Linear(n_embd, self.attn_sz, bias=False)
        self.value = nn.Linear(n_embd, self.attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, self.attn_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(self.attn_sz)
        else:
            self.key_norm = None
        self.output = nn.Linear(self.attn_sz, n_embd, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

        self.with_cp = with_cp

    def _init_weights(self, init_mode):
        multi_dim = self.n_embd * self.num_experts
        if init_mode == "fancy":
            with torch.no_grad():  # fancy init
                ratio_0_to_1 = self.layer_id / (self.n_layer - 1)  # 0 to 1
                ratio_1_to_almost0 = 1.0 - (self.layer_id / self.n_layer)  # 1 to ~0

                # fancy time_decay
                decay_speed = torch.ones(multi_dim)
                for h in range(multi_dim):
                    decay_speed[h] = -5 + 8 * (h / (multi_dim - 1)) ** (
                        0.7 + 1.3 * ratio_0_to_1
                    )
                self.spatial_decay = nn.Parameter(decay_speed)

                # fancy time_first
                zigzag = torch.tensor([(i + 1) % 3 - 1 for i in range(multi_dim)]) * 0.5
                self.spatial_first = nn.Parameter(
                    torch.ones(multi_dim) * math.log(0.3) + zigzag
                )

                # fancy time_mix
                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd
                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_v = nn.Parameter(
                    torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1
                )
                self.spatial_mix_r = nn.Parameter(
                    torch.pow(x, 0.5 * ratio_1_to_almost0)
                )
        elif init_mode == "local":
            self.spatial_decay = nn.Parameter(torch.ones(multi_dim))
            self.spatial_first = nn.Parameter(torch.ones(multi_dim))
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]))
        elif init_mode == "global":
            self.spatial_decay = nn.Parameter(torch.zeros(multi_dim))
            self.spatial_first = nn.Parameter(torch.zeros(multi_dim))
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        else:
            raise NotImplementedError

    def jit_func(self, x, patch_resolution):
        # Mix x with the previous timestep to produce xk, xv, xr
        B, T, C = x.size()
        if self.shift_pixel > 0:
            xx = self.shift_func(
                x, self.shift_pixel, self.channel_gamma, patch_resolution
            )
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xv = x * self.spatial_mix_v + xx * (1 - self.spatial_mix_v)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = x
            xv = x
            xr = x

        # Use xk, xv, xr to produce k, v, r
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)

        return sr, k, v

    def save_tensor(self, tensor, file_path):
        if os.path.exists(file_path):
            existing_data = torch.load(file_path)
            new_data = existing_data + [tensor]
        else:
            new_data = [tensor]
        torch.save(new_data, file_path)

    def forward(self, x, patch_resolution=None):
        debug = True
        save_dir = "../cls_debug_outputs"
        os.makedirs(save_dir, exist_ok=True)

        def _inner_forward(x):
            B, T, C = x.size()
            self.device = x.device

            h, w = patch_resolution
            sr, k, v = self.jit_func(x, patch_resolution)

            k = rearrange(k, "b (h w) c -> b c h w", h=h, w=w)
            v = rearrange(v, "b (h w) c -> b c h w", h=h, w=w)

            scan_func = [s_hw, s_wh, s_rhrw, s_rwrh]
            re_scan_func = [sr_hw, sr_wh, sr_rhrw, sr_rwrh]

            ks = torch.cat(
                [scan_func[i](k) for i in range(self.num_experts)], dim=2
            )  # b (h w) (c e)
            vs = torch.cat(
                [scan_func[i](v) for i in range(self.num_experts)], dim=2
            )  # b (h w) (c e)

            expert_output = RUN_CUDA(
                B,
                T,
                C * self.num_experts,
                self.spatial_decay / T,
                self.spatial_first / T,
                ks,
                vs,
            )
            expert_outputs = [
                expert_output[:, :, i * self.attn_sz : (i + 1) * self.attn_sz]
                for i in range(self.num_experts)
            ]  # (b (h w) c) * e
            expert_outputs = [
                rearrange(
                    re_scan_func[i](expert_outputs[i], h, w), "b c h w -> b (h w) c"
                )
                for i in range(self.num_experts)
            ]
            if debug:
                self.save_tensor(
                    expert_outputs,
                    os.path.join(save_dir, "expert_outputs_before_sr.pt"),
                )
            expert_output = torch.stack(expert_outputs, dim=0).mean(dim=0)  # b (h w) c
            if debug:
                self.save_tensor(
                    expert_output, os.path.join(save_dir, "expert_output.pt")
                )
            if self.key_norm is not None:
                expert_output = self.key_norm(expert_output)
            x = expert_output * sr
            if debug:
                self.save_tensor(
                    x,
                    os.path.join(save_dir, "expert_output_after_sr.pt"),
                )

            x = self.output(x)
            if debug:
                self.save_tensor(x, os.path.join(save_dir, "output.pt"))
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class VRWKV_ChannelMix(nn.Module):
    def __init__(
        self,
        n_embd,
        n_layer,
        layer_id,
        shift_mode="q_shift",
        channel_gamma=1 / 4,
        shift_pixel=1,
        hidden_rate=4,
        init_mode="fancy",
        key_norm=False,
        with_cp=False,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.with_cp = with_cp
        self._init_weights(init_mode)
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        if shift_pixel > 0:
            self.shift_func = eval(shift_mode)
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_r = None

        hidden_sz = hidden_rate * n_embd
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

        self.value.scale_init = 0
        self.receptance.scale_init = 0

    def _init_weights(self, init_mode):
        if init_mode == "fancy":
            with torch.no_grad():  # fancy init of time_mix
                ratio_1_to_almost0 = 1.0 - (self.layer_id / self.n_layer)  # 1 to ~0
                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd
                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
        elif init_mode == "local":
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]))
        elif init_mode == "global":
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        else:
            raise NotImplementedError

    def forward(self, x, patch_resolution=None):
        def _inner_forward(x):
            if self.shift_pixel > 0:
                xx = self.shift_func(
                    x, self.shift_pixel, self.channel_gamma, patch_resolution
                )
                xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
                xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
            else:
                xk = x
                xr = x

            k = self.key(xk)
            k = torch.square(torch.relu(k))
            if self.key_norm is not None:
                k = self.key_norm(k)
            kv = self.value(k)
            x = torch.sigmoid(self.receptance(xr)) * kv
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        n_embd,
        n_layer,
        layer_id,
        shift_mode="q_shift",
        channel_gamma=1 / 4,
        shift_pixel=1,
        drop_path=0.0,
        hidden_rate=4,
        init_mode="fancy",
        init_values=None,
        post_norm=False,
        key_norm=False,
        with_cp=False,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)

        self.att = VVRWKV_SpatialMix(
            n_embd,
            n_layer,
            layer_id,
            shift_mode,
            channel_gamma,
            shift_pixel,
            init_mode,
            key_norm=key_norm,
        )

        self.ffn = VRWKV_ChannelMix(
            n_embd,
            n_layer,
            layer_id,
            shift_mode,
            channel_gamma,
            shift_pixel,
            hidden_rate,
            init_mode,
            key_norm=key_norm,
        )
        self.layer_scale = init_values is not None
        self.post_norm = post_norm
        if self.layer_scale:
            self.gamma1 = nn.Parameter(
                init_values * torch.ones((n_embd)), requires_grad=True
            )
            self.gamma2 = nn.Parameter(
                init_values * torch.ones((n_embd)), requires_grad=True
            )
        self.with_cp = with_cp

    def forward(self, x, patch_resolution=None):
        def _inner_forward(x):
            if self.layer_id == 0:
                x = self.ln0(x)
            if self.post_norm:
                if self.layer_scale:
                    x = x + self.drop_path(
                        self.gamma1 * self.ln1(self.att(x, patch_resolution))
                    )
                    x = x + self.drop_path(
                        self.gamma2 * self.ln2(self.ffn(x, patch_resolution))
                    )
                else:
                    x = x + self.drop_path(self.ln1(self.att(x, patch_resolution)))
                    x = x + self.drop_path(self.ln2(self.ffn(x, patch_resolution)))
            else:
                if self.layer_scale:
                    x = x + self.drop_path(
                        self.gamma1 * self.att(self.ln1(x), patch_resolution)
                    )
                    x = x + self.drop_path(
                        self.gamma2 * self.ffn(self.ln2(x), patch_resolution)
                    )
                else:
                    x = x + self.drop_path(self.att(self.ln1(x), patch_resolution))
                    x = x + self.drop_path(self.ffn(self.ln2(x), patch_resolution))
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class VVision_RWKV(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        depth=12,
        drop_path_rate=0.0,
        in_channels=3,
        img_size=224,
        patch_size=16,
        hidden_rate=4,
        interpolation_mode="bicubic",
        drop_after_pos_rate=0.0,
        out_indices=[2, 5, 8, 11],
        final_norm=True,
    ):
        """
        Args:
            embed_dims: Number of embedding dimensions
            depth: Number of layers
            drop_path_rate: Drop path rate
            in_channels: Number of input channels
            img_size: Size of the input image
            patch_size: Size of the patch
            drop_after_pos_rate: Dropout rate after positional encoding
            out_indices: Indices of the output layers

        Output:
            tuple: Tuple containing the output of the layers specified in out_indices
        """
        super(VVision_RWKV, self).__init__()
        self.embed_dims = embed_dims
        self.num_extra_tokens = 0
        self.num_layers = depth
        self.drop_path_rate = drop_path_rate
        self.interpolate_mode = interpolation_mode

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

        self.patch_resolution = self.patch_embed.init_out_size
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]

        # Set position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dims))

        self.drop_after_pos = nn.Dropout(p=drop_after_pos_rate)

        self.out_indices = out_indices

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(
                Block(
                    n_embd=embed_dims,
                    n_layer=depth,
                    layer_id=i,
                    hidden_rate=hidden_rate,
                    key_norm=False,
                )
            )

        self.ln1 = nn.LayerNorm(embed_dims) if final_norm else None

    def forward(self, x):
        # B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)
        h, w = patch_resolution

        x = x + resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=0,
        )

        x = self.drop_after_pos(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x, patch_resolution)

            if i == len(self.layers) - 1 and self.ln1 is not None:
                # x = rearrange(x, "b c h w -> b (h w) c")
                x = self.ln1(x)
                # x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

            if i in self.out_indices:
                x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
                outs.append(x)
                x = rearrange(x, "b c h w -> b (h w) c")

        return tuple(outs)


if __name__ == "__main__":
    # spatial_mix = VVRWKV_SpatialMix(3, 2, 1, key_norm=True).cuda()
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
        decode_head=LinearClsHead(
            num_classes=1000,
            in_channels=[192, 192, 192, 192],
        ),
    ).cuda()
    checkpoint = torch.load("../checkpoints/vvrwkv_t_in1k_cls_convert.pth")[
        "model_state_dict"
    ]
    model.load_state_dict(checkpoint)
    # x = torch.randn(2, 3, 224, 224).cuda()
    # output = spatial_mix(x, (2, 2))
    preprocess_image = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img = Image.open("../raw.jpg")
    x = preprocess_image(img).unsqueeze(0).cuda()
    output = model(x)
    print(output[0].shape)
    print(output)
