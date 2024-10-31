import math
import unittest

import torch
from torch import Tensor, nn
import torch.utils.checkpoint as cp

from model.vrwkv import Block
from model.wkv import RUN_CUDA
from model.vvrwkv import q_shift


class RevShuffle(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.len = 0
        self.idx = None  # save shuffle idx

    def forward(self, x: Tensor, shuffle: bool, gen_state=False):
        """
        :param x: input tensor
        :param shuffle: shuffle or reverse shuffle
        :param gen_state: whether gen shuffle state, if False, use last shuffle state. mast call once with True.
        :return: shuffled x
        """
        if shuffle:
            if gen_state:
                self._gen_shuffle_state(x)
            return self._shuffle(x)
        else:
            return self._rev_shuffle(x)

    def _gen_shuffle_state(self, x: Tensor):
        self.len = x.shape[self.dim]
        self.idx = torch.randperm(self.len)

    def _shuffle(self, x: Tensor):
        return x.index_select(self.dim, self.idx.to(x.device))

    def _rev_shuffle(self, x):
        rev_idx = torch.zeros_like(self.idx)
        rev_idx[self.idx] = torch.arange(self.len)
        return x.index_select(self.dim, rev_idx.to(x.device))


class MonteCarlo(nn.Module):
    def __init__(self, n_embd, sample_num=16):
        super().__init__()
        self.sample_num = sample_num
        self.attn = Block(n_embd=n_embd, n_layer=1, layer_id=0)
        self.rev_shuffle = RevShuffle(dim=2)

    def forward(self, x: Tensor):
        if self.training:
            return self._forward(x)
        else:
            return self._val_forward(x)

    def _forward(self, x):
        x = self.rev_shuffle(x, True, True)
        x = self.attn(x)
        x = self.rev_shuffle(x, False)
        return x

    def _val_forward(self, x):
        xs = []
        for _ in range(self.sample_num):
            _x = self.rev_shuffle(x, True, True)
            _x = self.attn(_x)
            _x = self.rev_shuffle(_x, False)
            xs.append(_x)
        x_avg = torch.stack(xs).mean(dim=0)
        return x_avg


class Random_SpatialMix(nn.Module):
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
        sample_num=16,
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

        self.sample_num = sample_num
        # random shuffle
        self.rev_shuffle = RevShuffle(dim=2)

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
        if init_mode == "fancy":
            with torch.no_grad():  # fancy init
                ratio_0_to_1 = self.layer_id / (self.n_layer - 1)  # 0 to 1
                ratio_1_to_almost0 = 1.0 - (self.layer_id / self.n_layer)  # 1 to ~0

                # fancy time_decay
                decay_speed = torch.ones(self.n_embd)
                for h in range(self.n_embd):
                    decay_speed[h] = -5 + 8 * (h / (self.n_embd - 1)) ** (
                        0.7 + 1.3 * ratio_0_to_1
                    )
                self.spatial_decay = nn.Parameter(decay_speed)

                # fancy time_first
                zigzag = (
                    torch.tensor([(i + 1) % 3 - 1 for i in range(self.n_embd)]) * 0.5
                )
                self.spatial_first = nn.Parameter(
                    torch.ones(self.n_embd) * math.log(0.3) + zigzag
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
            self.spatial_decay = nn.Parameter(torch.ones(self.n_embd))
            self.spatial_first = nn.Parameter(torch.ones(self.n_embd))
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]))
        elif init_mode == "global":
            self.spatial_decay = nn.Parameter(torch.zeros(self.n_embd))
            self.spatial_first = nn.Parameter(torch.zeros(self.n_embd))
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
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(self._inner_forward, x, patch_resolution)
        else:
            x = self._inner_forward(x, patch_resolution)
        return x

    def _inner_forward(self, x, patch_resolution):
        B, T, C = x.size()
        self.device = x.device

        sr, k, v = self.jit_func(x, patch_resolution)

        if self.training:
            wkv = self._train_wkv(k, v, B, T, C)
        else:
            wkv = self._val_wkv(k, v, B, T, C)
        if self.key_norm is not None:
            wkv = self.key_norm(wkv)
        x = sr * wkv

        x = self.output(x)
        return x

    def _train_wkv(self, k, v, B, T, C):
        k = self.rev_shuffle(k, True, True)  # gen shuffle state
        v = self.rev_shuffle(v, True)  # use same shuffle state
        wkv = RUN_CUDA(
            B,
            T,
            C,
            self.spatial_decay / T,
            self.spatial_first / T,
            k,
            v,
        )
        wkv = self.rev_shuffle(wkv, False)  # reverse shuffle
        return wkv

    def _val_wkv(self, k, v, B, T, C):
        wkvs = []
        for _ in range(self.sample_num):
            k_ = self.rev_shuffle(k, True, True)  # gen new shuffle state
            v_ = self.rev_shuffle(v, True)  # use same shuffle state
            wkv_ = RUN_CUDA(
                B,
                T,
                C,
                self.spatial_decay / T,
                self.spatial_first / T,
                k_,
                v_,
            )
            wkv_ = self.rev_shuffle(wkv_, False)  # reverse shuffle
            wkvs.append(wkv_)
        wkv_avg = torch.stack(wkvs).mean(dim=0)
        return wkv_avg


class TestRandomRWKV(unittest.TestCase):
    def test_RevShuffle(self):
        rev_shuffle = RevShuffle(dim=1)
        x = torch.randn(1, 4, 1)
        print(x)
        shuffle_x = rev_shuffle(x, True)
        print(shuffle_x)
        rev_x = rev_shuffle(shuffle_x, False)
        self.assertTrue(torch.equal(x, rev_x))

    def test_train_RandomSpatialMix(self):
        random_shuffle = Random_SpatialMix(
            n_embd=3, n_layer=2, layer_id=0, sample_num=16
        ).cuda()
        x = torch.randn(2, 4, 3).cuda()
        target = torch.randn(2, 4, 3).cuda()
        criterion = nn.MSELoss()
        output = random_shuffle(x, (2, 2))
        loss = criterion(output, target)
        loss.backward()
        print(output)

    def test_val_RandomSpatialMix(self):
        random_shuffle = Random_SpatialMix(
            n_embd=3, n_layer=2, layer_id=0, sample_num=16
        ).cuda()
        x = torch.randn(2, 4, 3).cuda()
        output = random_shuffle(x, (2, 2))
        print(output)

