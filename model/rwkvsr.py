import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from model.base_model import SegModel
from model.upernet import UPerNet_1
from model.vrwkv import PatchEmbed, resize_pos_embed
from model.wkv import RUN_CUDA


class OmniShift(nn.Module):
    def __init__(self, wn, dim):
        super(OmniShift, self).__init__()
        # Define the layers for training
        self.conv1x1 = wn(
            nn.Conv2d(
                in_channels=dim, out_channels=dim, kernel_size=1, groups=dim, bias=False
            )
        )
        self.conv3x3 = wn(
            nn.Conv2d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=3,
                padding=1,
                groups=dim,
                bias=False,
            )
        )
        self.conv5x5 = wn(
            nn.Conv2d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=5,
                padding=2,
                groups=dim,
                bias=False,
            )
        )
        self.alpha = nn.Parameter(torch.randn(4), requires_grad=True)

        # Define the layers for testing
        self.conv5x5_reparam = wn(
            nn.Conv2d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=5,
                padding=2,
                groups=dim,
                bias=False,
            )
        )
        self.repram_flag = True

    def forward_train(self, x):
        out1x1 = self.conv1x1(x)
        out3x3 = self.conv3x3(x)
        out5x5 = self.conv5x5(x)
        # import pdb
        # pdb.set_trace()

        out = (
            self.alpha[0] * x
            + self.alpha[1] * out1x1
            + self.alpha[2] * out3x3
            + self.alpha[3] * out5x5
        )
        return out

    def reparam_5x5(self):
        # Combine the parameters of conv1x1, conv3x3, and conv5x5 to form a single 5x5 depth-wise convolution

        padded_weight_1x1 = F.pad(self.conv1x1.weight, (2, 2, 2, 2))
        padded_weight_3x3 = F.pad(self.conv3x3.weight, (1, 1, 1, 1))

        identity_weight = F.pad(torch.ones_like(self.conv1x1.weight), (2, 2, 2, 2))

        combined_weight = (
            self.alpha[0] * identity_weight
            + self.alpha[1] * padded_weight_1x1
            + self.alpha[2] * padded_weight_3x3
            + self.alpha[3] * self.conv5x5.weight
        )

        device = self.conv5x5_reparam.weight.device

        combined_weight = combined_weight.to(device)

        self.conv5x5_reparam.weight = nn.Parameter(combined_weight).cuda()

    def forward(self, x):

        if self.training:
            self.repram_flag = True
            out = self.forward_train(x)
        elif self.training == False and self.repram_flag == True:
            self.reparam_5x5()
            self.repram_flag = False
            out = self.conv5x5_reparam(x)
        elif self.training == False and self.repram_flag == False:
            out = self.conv5x5_reparam(x)

        return out


class FeedForward(nn.Module):
    def __init__(self, wn, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            wn(nn.Conv2d(dim, dim * mult, 1)),
            nn.GELU(),
            nn.Dropout(dropout),
            wn(nn.Conv2d(dim * mult, dim, 1)),
        )

    def forward(self, x):
        return self.net(x)


class BasicConv2d(nn.Module):
    def __init__(
        self, wn, in_channel, out_channel, kernel_size, stride, padding=(0, 0)
    ):
        super(BasicConv2d, self).__init__()
        self.conv = wn(
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class S2dBlock(nn.Module):
    def __init__(self, wn, n_feats):
        super(S2dBlock, self).__init__()

        self.conv = nn.Sequential(
            BasicConv2d(
                wn, n_feats, n_feats, kernel_size=(1, 3), stride=1, padding=(0, 1)
            ),
            BasicConv2d(
                wn, n_feats, n_feats, kernel_size=(3, 1), stride=1, padding=(1, 0)
            ),
        )

    def forward(self, x):
        return self.conv(x)


def _to_3d_tensor(x, depth_stride=None):
    """Converts a 4d tensor to 3d by stackin
    the batch and depth dimensions."""
    x = x.transpose(0, 2)  # swap batch and depth dimensions: NxCxHxW => HxCxNxW
    if depth_stride:
        x = x[::depth_stride]  # downsample feature maps along depth dimension
    depth = x.size()[0]
    # x = rearrange(x,"h c n w -> n h c w")
    x = x.permute(2, 0, 1, 3)  # HxCxNxW => NxHxCxW
    x = torch.split(x, 1, dim=0)  # split along batch dimension: NxHxCxW => N*[1xHxCxW]
    x = torch.cat(x, 1)  # concatenate along depth dimension: N*[1xHxCxW] => 1x(N*H)xCxW
    x = x.squeeze(0)  # 1x(N*H)xCxW => (N*H)xCxW
    return x, depth


def _to_4d_tensor(x, depth_stride=None):
    """Converts a 5d tensor to 4d by stackin
    the batch and depth dimensions."""
    x = x.transpose(0, 2)  # swap batch and depth dimensions: NxCxDxHxW => DxCxNxHxW
    if depth_stride:
        x = x[::depth_stride]  # downsample feature maps along depth dimension
    depth = x.size()[0]
    x = x.permute(2, 0, 1, 3, 4)  # DxCxNxHxW => NxDxCxHxW
    x = torch.split(
        x, 1, dim=0
    )  # split along batch dimension: NxDxCxHxW => N*[1xDxCxHxW]
    x = torch.cat(
        x, 1
    )  # concatenate along depth dimension: N*[1xDxCxHxW] => 1x(N*D)xCxHxW
    x = x.squeeze(0)  # 1x(N*D)xCxHxW => (N*D)xCxHxW
    return x, depth


def _to_5d_tensor(x, depth):
    """Converts a 4d tensor back to 5d by splitting
    the batch dimension to restore the depth dimension."""
    x = torch.split(x, depth)  # (N*D)xCxHxW => N*[DxCxHxW]
    x = torch.stack(x, dim=0)  # re-instate the batch dimension: NxDxCxHxW
    x = x.transpose(
        1, 2
    )  # swap back depth and channel dimensions: NxDxCxHxW => NxCxDxHxW
    return x


class VRWKV_SpatialMix(nn.Module):
    def __init__(self, wn, n_embd, n_layer, layer_id, key_norm=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.device = None
        attn_sz = n_embd

        # self.dwconv = wn(
        #     nn.Conv2d(
        #         n_embd,
        #         n_embd,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1,
        #         groups=n_embd,
        #         bias=False,
        #     )
        # )

        self.recurrence = 2

        self.omni_shift = OmniShift(wn, dim=n_embd)

        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(n_embd)
        else:
            self.key_norm = None
        self.output = nn.Linear(attn_sz, n_embd, bias=False)

        with torch.no_grad():
            self.spatial_decay = nn.Parameter(
                torch.randn((self.recurrence, self.n_embd))
            )
            self.spatial_first = nn.Parameter(
                torch.randn((self.recurrence, self.n_embd))
            )

    def jit_func(self, x, resolution):
        # Mix x with the previous timestep to produce xk, xv, xr

        h, w = resolution

        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = self.omni_shift(x)
        x = rearrange(x, "b c h w -> b (h w) c")

        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)
        sr = torch.sigmoid(r)

        return sr, k, v

    def forward(self, x, resolution):
        B, T, C = x.size()
        self.device = x.device

        sr, k, v = self.jit_func(x, resolution)

        for j in range(self.recurrence):
            if j % 2 == 0:
                v = RUN_CUDA(
                    B, T, C, self.spatial_decay[j] / T, self.spatial_first[j] / T, k, v
                )
            else:
                h, w = resolution
                k = rearrange(k, "b (h w) c -> b (w h) c", h=h, w=w)
                v = rearrange(v, "b (h w) c -> b (w h) c", h=h, w=w)
                v = RUN_CUDA(
                    B, T, C, self.spatial_decay[j] / T, self.spatial_first[j] / T, k, v
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
    def __init__(
        self,
        wn,
        n_embd,
        n_layer,
        layer_id,
        hidden_rate=4,
        key_norm=False,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd

        hidden_sz = int(hidden_rate * n_embd)
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)

        self.omni_shift = OmniShift(wn, dim=n_embd)

        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

    def forward(self, x, resolution):

        h, w = resolution

        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = self.omni_shift(x)
        x = rearrange(x, "b c h w -> b (h w) c")

        k = self.key(x)
        k = torch.square(torch.relu(k))
        if self.key_norm is not None:
            k = self.key_norm(k)
        kv = self.value(k)
        x = torch.sigmoid(self.receptance(x)) * kv

        return x


class rwkvblock(nn.Module):
    def __init__(
        self,
        wn,
        n_embd,
        n_layer,
        layer_id,
        hidden_rate=4,
        key_norm=False,
    ):
        super().__init__()
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)

        self.att = VRWKV_SpatialMix(wn, n_embd, n_layer, layer_id, key_norm=key_norm)

        self.ffn = VRWKV_ChannelMix(
            wn, n_embd, n_layer, layer_id, hidden_rate, key_norm=key_norm
        )

        self.ffn_mlp = FeedForward(wn, n_embd)

        self.gamma1 = nn.Parameter(torch.ones(n_embd), requires_grad=True).cuda()
        self.gamma2 = nn.Parameter(torch.ones(n_embd), requires_grad=True).cuda()

    def forward(self, x):
        b, c, h, w = x.shape
        residual = x

        resolution = (h, w)
        x1 = rearrange(x, "b c h w -> b (h w) c")
        x1 = self.att(self.ln1(x1), resolution)
        x1 = rearrange(x1, "b (h w) c -> b h w c", h=h, w=w)

        x2 = rearrange(x, "b c h w -> b (h w) c")
        x2 = self.ffn(self.ln2(x2), resolution)
        x2 = rearrange(x2, "b (h w) c -> b h w c", h=h, w=w)

        x3 = x.permute(0, 2, 3, 1) + self.gamma1 * x1 + self.gamma2 * x2
        out = x3.permute(0, 3, 1, 2) + self.ffn_mlp(self.ln3(x3).permute(0, 3, 1, 2))
        return out


class Block(nn.Module):
    def __init__(self, wn, n_feats, n_conv, num_blocks=[1, 2, 2, 4]):
        super(Block, self).__init__()

        self.relu = nn.ReLU(inplace=False)

        block1 = []
        for i in range(n_conv):
            block1.append(S2dBlock(wn, n_feats))
        self.block1 = nn.Sequential(*block1)

        block2 = []
        for i in range(n_conv):
            block2.append(S2dBlock(wn, n_feats))
        self.block2 = nn.Sequential(*block2)

        block3 = []
        for i in range(n_conv):
            block3.append(S2dBlock(wn, n_feats))
        self.block3 = nn.Sequential(*block3)

        self.reduceF = BasicConv2d(wn, n_feats * 3, n_feats, kernel_size=1, stride=1)
        self.conv = S2dBlock(wn, n_feats)
        self.gamma = nn.Parameter(torch.ones(3))
        self.conv1 = nn.Sequential(
            *[
                rwkvblock(wn, n_embd=n_feats, n_layer=num_blocks[0], layer_id=i)
                for i in range(num_blocks[0])
            ]
        )

        self.conv2 = nn.Sequential(
            *[
                rwkvblock(wn, n_embd=n_feats, n_layer=num_blocks[1], layer_id=i)
                for i in range(num_blocks[1])
            ]
        )

        self.conv3 = nn.Sequential(
            *[
                rwkvblock(wn, n_embd=n_feats, n_layer=num_blocks[2], layer_id=i)
                for i in range(num_blocks[2])
            ]
        )

    def forward(self, x):
        res = x
        x1 = self.block1(x) + x
        x2 = self.block2(x1) + x1
        x3 = self.block3(x2) + x2

        # x1, depth = _to_4d_tensor(x1, depth_stride=1)
        # print(res.shape, x1.shape)
        x1 = self.conv1(x1)
        # x1 = _to_5d_tensor(x1, depth)

        # x2, depth = _to_4d_tensor(x2, depth_stride=1)
        x2 = self.conv2(x2)
        # x2 = _to_5d_tensor(x2, depth)

        # x3, depth = _to_4d_tensor(x3, depth_stride=1)
        x3 = self.conv3(x3)
        # x3 = _to_5d_tensor(x3, depth)

        x = torch.cat([self.gamma[0] * x1, self.gamma[1] * x2, self.gamma[2] * x3], 1)
        x = self.reduceF(x)
        x = self.relu(x)
        x = x + res

        x = self.conv(x)
        return x


class RWKVNet(nn.Module):
    def __init__(
        self,
        img_size=256,
        in_channels=31,
        patch_size=4,
        n_feats=[64, 128, 256, 512],
        out_slices=[0, 1, 2, 3],
        n_conv=1,
    ):
        super(RWKVNet, self).__init__()
        self.out_slices = out_slices

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=n_feats[0],
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

        self.patch_resolution = self.patch_embed.init_out_size
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]

        # Set position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, n_feats[0]))

        wn = lambda x: torch.nn.utils.parametrizations.weight_norm(x)

        convs = []
        for i in range(len(n_feats) - 1):
            convs.append(nn.Conv2d(n_feats[i], n_feats[i + 1], kernel_size=1))
        self.convs = nn.ModuleList(convs)

        ssrms = []
        for feat in n_feats:
            ssrms.append(Block(wn, feat, n_conv))
        self.ssrms = nn.ModuleList(ssrms)

    def forward(self, x):
        x, patch_resolution = self.patch_embed(x)
        h, w = patch_resolution

        # pos embedding
        T = x + resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode="bilinear",
            num_extra_tokens=0,
        )
        x = rearrange(T, "b (h w) c -> b c h w", h=h, w=w)

        res = []
        for i, ssrm in enumerate(self.ssrms):
            if i > 0:
                x = self.convs[i - 1](x)
            x = ssrm(x)
            if i in self.out_slices:
                res.append(x)

        return res


if __name__ == "__main__":
    model = SegModel(
        backbone=RWKVNet(
            img_size=256,
            in_channels=31,
            n_feats=[64, 128, 256, 512],
            out_slices=[0, 1, 2, 3],
        ),
        decode_head=UPerNet_1(
            num_classes=21,
            image_size=256,
            fc_dim=512,
            fpn_inplanes=[64, 128, 256, 512],
            fpn_dim=256,
        ),
    ).cuda()

    x = torch.randn(2, 31, 256, 256).cuda()
    output = model(x)
    print(output.shape)
