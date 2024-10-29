import math

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from einops import rearrange
from torch import nn

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
        drop=0.1,
        revin_affine=True,
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
        self.drop = drop
        self.revin_affine = revin_affine
        self.num_experts = 4

        self.gate = nn.Conv2d(n_embd, self.num_experts, 1)

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
        self.receptance = nn.Linear(n_embd, self.attn_sz * self.num_experts, bias=False)
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

    def forward(self, x, patch_resolution=None):
        def _inner_forward(x):
            B, T, C = x.size()
            self.device = x.device

            h, w = patch_resolution
            score = F.softmax(
                self.gate(rearrange(x, "b (h w) c -> b c h w", h=h, w=w)), dim=1
            )  # b e h w
            score = score.unsqueeze(1)  # b 1 e h w

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
            if self.key_norm is not None:
                expert_outputs = [self.key_norm(eo) for eo in expert_outputs]
            expert_outputs = torch.cat(expert_outputs, dim=2)  # b (h w) (c e)
            expert_outputs = sr * expert_outputs  # b (h w) (c e)
            expert_outputs = rearrange(
                expert_outputs,
                "b (h w) (c e) -> b c e h w",
                h=h,
                w=w,
                c=self.attn_sz,
                e=self.num_experts,
            )  # b c e h w

            prediction = torch.sum(expert_outputs * score, dim=2)  # b c h w
            x = rearrange(prediction, "b c h w -> b (h w) c")

            x = self.output(x)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


if __name__ == "__main__":
    spatial_mix = VVRWKV_SpatialMix(3, 2, 1, key_norm=True).cuda()
    x = torch.randn(2, 2 * 2, 3).cuda()
    output = spatial_mix(x, (2, 2))
    print(output)

