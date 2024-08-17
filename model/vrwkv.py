import collections

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from model.token_shift import OmniShift
from model.wkv import RUN_CUDA


class VRWKV_SpatialMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, recurrence=4, key_norm=False):

        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.device = None
        self.attn_sz = n_embd

        self.dwconv = nn.Conv2d(
            n_embd,
            n_embd,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=n_embd,
            bias=False,
        )

        assert recurrence % 4 == 0, "recurrence must be divisible by 4"
        self.recurrence = recurrence

        self.omni_shift = OmniShift(dim=n_embd)
        # self.q_shift = QShift()

        self.key = nn.Linear(n_embd, self.attn_sz, bias=False)
        self.value = nn.Linear(n_embd, self.attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, self.attn_sz, bias=False)

        self.forward_conv1d = nn.Conv1d(
            in_channels=self.attn_sz, out_channels=self.attn_sz, kernel_size=1
        )
        self.backward_conv1d = nn.Conv1d(
            in_channels=self.attn_sz, out_channels=self.attn_sz, kernel_size=1
        )

        if key_norm:
            self.key_norm = nn.LayerNorm(self.attn_sz)
        else:
            self.key_norm = None

        self.output = nn.Linear(self.attn_sz, n_embd, bias=False)

        with torch.no_grad():
            self.spatial_decay = nn.Parameter(
                torch.randn((self.recurrence, self.attn_sz))
            )
            self.spatial_first = nn.Parameter(
                torch.randn((self.recurrence, self.attn_sz))
            )

    def jit_func(self, x, resolution):
        h, w = resolution
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = self.omni_shift(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        # x = self.q_shift(x, resolution)

        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)
        sr = torch.sigmoid(r)

        return sr, k, v

    def forward(self, x, resolution):
        B, T, C = x.size()
        self.device = x.device

        h, w = resolution
        sr, k, v = self.jit_func(x, resolution)

        _forward = True

        for j in range(self.recurrence):
            if j % 2 == 0:
                k = rearrange(k, "b t c -> b c t")
                v = rearrange(v, "b t c -> b c t")
                if _forward:
                    k = self.forward_conv1d(k)
                    v = self.forward_conv1d(v)
                    _forward = False
                else:
                    k = self.backward_conv1d(k)
                    v = self.backward_conv1d(v)
                    _forward = True
                k = rearrange(k, "b c t -> b t c")
                v = rearrange(v, "b c t -> b t c")

            if j % 2 == 0:
                v = RUN_CUDA(
                    B,
                    T,
                    self.attn_sz,
                    self.spatial_decay[j] / T,
                    self.spatial_first[j] / T,
                    k,
                    v,
                )
            else:
                k = rearrange(k, "b (h w) c -> b (w h) c", h=h, w=w)
                v = rearrange(v, "b (h w) c -> b (w h) c", h=h, w=w)
                v = RUN_CUDA(
                    B,
                    T,
                    self.attn_sz,
                    self.spatial_decay[j] / T,
                    self.spatial_first[j] / T,
                    k,
                    v,
                )
                k = rearrange(k, "b (w h) c -> b (h w) c", h=h, w=w)
                v = rearrange(v, "b (w h) c -> b (h w) c", h=h, w=w)

        x = v
        if self.key_norm is not None:
            x = self.key_norm(x)
        x = sr * x
        x = self.output(x)
        return x


class VRWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, hidden_rate=1, key_norm=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd

        self.omni_shift = OmniShift(dim=n_embd)
        # self.q_shift = QShift()

        self.hidden_sz = int(hidden_rate * n_embd)

        self.key = nn.Linear(n_embd, self.hidden_sz, bias=False)

        if key_norm:
            self.key_norm = nn.LayerNorm(self.hidden_sz)
        else:
            self.key_norm = None

        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(self.hidden_sz, n_embd, bias=False)
        # self.value = nn.Linear(n_embd, self.hidden_sz, bias=False)

        self.output = nn.Linear(self.hidden_sz, n_embd, bias=False)

        with torch.no_grad():
            self.spatial_decay = nn.Parameter(torch.randn(self.n_embd))
            self.spatial_first = nn.Parameter(torch.randn(self.n_embd))

    def forward(self, x, resolution):
        B, T, C = x.size()

        h, w = resolution
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = self.omni_shift(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        # x = self.q_shift(x, resolution)

        k = self.key(x)
        k = torch.square(torch.relu(k))
        if self.key_norm is not None:
            k = self.key_norm(k)

        # v = self.value(x)
        # v = RUN_CUDA(
        #     B, T, self.hidden_sz, self.spatial_decay / T, self.spatial_first / T, k, v
        # )
        # x = torch.sigmoid(self.receptance(x)) * self.output(v)

        kv = self.value(k)
        x = torch.sigmoid(self.receptance(x)) * kv
        return x


class FFN(nn.Module):
    def __init__(self, n_embd, hidden_rate=4):
        super().__init__()
        hidden_sz = int(hidden_rate * n_embd)
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden_sz),
            nn.GELU(),
            nn.Linear(hidden_sz, n_embd),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, hidden_rate=1, key_norm=False):
        super().__init__()
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.att = VRWKV_SpatialMix(n_embd, n_layer, layer_id, key_norm=key_norm)

        self.ffn = VRWKV_ChannelMix(
            n_embd, n_layer, layer_id, hidden_rate=hidden_rate, key_norm=key_norm
        )
        # self.ffn = FFN(n_embd=n_embd)

        self.gamma1 = nn.Parameter(torch.ones(n_embd), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.ones(n_embd), requires_grad=True)

    def forward(self, x):
        _, _, h, w = x.shape

        resolution = (h, w)

        # x = self.dwconv1(x) + x
        x = rearrange(x, "b c h w -> b (h w) c")
        x = x + self.gamma1 * self.att(self.ln1(x), resolution)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

        # x = self.dwconv2(x) + x
        x = rearrange(x, "b c h w -> b (h w) c")
        x = x + self.gamma2 * self.ffn(self.ln2(x), resolution)
        # x = x + self.gamma2 * self.ffn(self.ln2(x))
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

        return x


# class PatchEmbed(nn.Module):
#     def __init__(
#         self,
#         in_channels=3,
#         input_size=224,
#         embed_dims=256,
#         kernel_size=16,
#         stride=16,
#         bias=True,
#     ):
#         super(PatchEmbed, self).__init__()
#         self.input_size = input_size
#         self.in_channels = in_channels
#         self.patch_resolution = (
#             math.floor((input_size - kernel_size) / stride) + 1,
#             math.floor((input_size - kernel_size) / stride) + 1,
#         )

#         self.proj = nn.Conv2d(
#             in_channels=in_channels,
#             out_channels=embed_dims,
#             kernel_size=kernel_size,
#             stride=stride,
#             bias=bias,
#         )

#     def forward(self, x):
#         x = self.proj(x)
#         x = rearrange(x, "b c h w -> b (h w) c")
#         return x, self.patch_resolution


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return tuple([x, x])


class PatchEmbed(nn.Module):
    def __init__(
        self,
        in_channels=3,
        embed_dims=768,
        kernel_size=16,
        stride=16,
        padding=0,
        dilation=1,
        bias=True,
        input_size=224,
    ):
        super().__init__()

        self.embed_dims = embed_dims
        if stride is None:
            stride = kernel_size

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)
        padding = to_2tuple(padding)

        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        input_size = (input_size, input_size)
        self.init_input_size = input_size
        h_out = (
            input_size[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1
        ) // stride[0] + 1
        w_out = (
            input_size[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1
        ) // stride[1] + 1
        self.init_out_size = (h_out, w_out)

    def forward(self, x):
        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        return x, out_size


def resize_pos_embed(
    pos_embed, src_shape, dst_shape, mode="bicubic", num_extra_tokens=1
):
    """Resize pos_embed weights.

    Args:
        pos_embed (torch.Tensor): Position embedding weights with shape
            [1, L, C].
        src_shape (tuple): The resolution of downsampled origin training
            image, in format (H, W).
        dst_shape (tuple): The resolution of downsampled new training
            image, in format (H, W).
        mode (str): Algorithm used for upsampling. Choose one from 'nearest',
            'linear', 'bilinear', 'bicubic' and 'trilinear'.
            Defaults to 'bicubic'.
        num_extra_tokens (int): The number of extra tokens, such as cls_token.
            Defaults to 1.

    Returns:
        torch.Tensor: The resized pos_embed of shape [1, L_new, C]
    """
    if src_shape[0] == dst_shape[0] and src_shape[1] == dst_shape[1]:
        return pos_embed
    assert pos_embed.ndim == 3, "shape of pos_embed must be [1, L, C]"
    _, L, C = pos_embed.shape
    src_h, src_w = src_shape
    assert L == src_h * src_w + num_extra_tokens, (
        f"The length of `pos_embed` ({L}) doesn't match the expected "
        f"shape ({src_h}*{src_w}+{num_extra_tokens}). Please check the"
        "`img_size` argument."
    )
    extra_tokens = pos_embed[:, :num_extra_tokens]

    src_weight = pos_embed[:, num_extra_tokens:]
    src_weight = src_weight.reshape(1, src_h, src_w, C).permute(0, 3, 1, 2)

    dst_weight = F.interpolate(
        src_weight, size=dst_shape, align_corners=False, mode=mode
    )
    dst_weight = torch.flatten(dst_weight, 2).transpose(1, 2)

    return torch.cat((extra_tokens, dst_weight), dim=1)


class HWC_RWKV(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        depth=12,
        drop_path_rate=0.0,
        in_channels=3,
        img_size=224,
        patch_size=16,
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
        super(HWC_RWKV, self).__init__()
        self.embed_dims = embed_dims
        self.num_extra_tokens = 0
        self.num_layers = depth
        self.drop_path_rate = drop_path_rate

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
                    hidden_rate=1,
                    key_norm=False,
                )
            )

        self.ln_final = nn.LayerNorm(embed_dims) if final_norm else None

    def forward(self, x):
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)
        h, w = patch_resolution

        x = x + resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode="bicubic",
            num_extra_tokens=0,
        )

        x = self.drop_after_pos(x)

        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i == len(self.layers) - 1 and self.ln_final is not None:
                x = rearrange(x, "b c h w -> b (h w) c")
                x = self.ln_final(x)
                x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)
