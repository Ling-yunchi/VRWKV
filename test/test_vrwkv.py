import unittest

from model.vrwkv import *


class TestVrwkv(unittest.TestCase):
    def test_VRWKV_Multi_HW_SpatialMix(self):
        model = VRWKV_Multi_HW_SpatialMix(
            n_embd=32, n_layer=1, layer_id=0, key_norm=True
        ).cuda()

        x = torch.randn(1, 64 * 64, 32).cuda()
        out = model(x, (64, 64))

        self.assertEqual(out.shape, (1, 64 * 64, 32))

    def test_VRWKV_Multi_HWC_SpatialMix(self):
        model = VRWKV_Multi_HWC_SpatialMix(
            n_embd=32, n_layer=1, layer_id=0, key_norm=True
        ).cuda()

        x = torch.randn(1, 64 * 64, 32).cuda()
        out = model(x, (64, 64))

        self.assertEqual(out.shape, (1, 64 * 64, 32))

    def test_VRWKV_HWC_SpatialMix(self):
        model = VRWKV_HWC_SpatialMix(
            n_embd=32, n_layer=1, layer_id=0, recurrence=4, key_norm=False
        ).cuda()

        x = torch.randn(1, 64 * 64, 32).cuda()  # B T C
        out = model(x, (64, 64))

        self.assertEqual(out.shape, (1, 64 * 64, 32))

    def test_VRWKV_HW_SpatialMix(self):
        model = VRWKV_HW_SpatialMix(
            n_embd=32, n_layer=1, layer_id=0, recurrence=2, key_norm=False
        ).cuda()

        x = torch.randn(1, 64 * 64, 32).cuda()
        out = model(x, (64, 64))

        self.assertEqual(out.shape, (1, 64 * 64, 32))

    def test_VRWKV_SpatialMix(self):
        model = VRWKV_SpatialMix(
            n_embd=32,
            n_layer=1,
            layer_id=0,
            channel_gamma=1 / 4,
            shift_pixel=1,
            key_norm=False,
        ).cuda()

        x = torch.randn(1, 64 * 64, 32).cuda()
        out = model(x, (64, 64))

        self.assertEqual(out.shape, (1, 64 * 64, 32))

    def test_VRWKV_WKV_ChannelMix(self):
        model = VRWKV_WKV_ChannelMix(
            n_embd=32, n_layer=1, layer_id=0, hidden_rate=1, key_norm=False
        ).cuda()

        x = torch.randn(1, 64 * 64, 32).cuda()
        out = model(x, (64, 64))

        self.assertEqual(out.shape, (1, 64 * 64, 32))

    def test_VRWKV_ChannelMix(self):
        model = VRWKV_ChannelMix(
            n_embd=32, n_layer=1, layer_id=0, hidden_rate=1, key_norm=False
        ).cuda()

        x = torch.randn(1, 64 * 64, 32).cuda()
        out = model(x, (64, 64))

        self.assertEqual(out.shape, (1, 64 * 64, 32))

    def test_FFN(self):
        model = FFN(32, hidden_rate=4).cuda()

        x = torch.randn(1, 64 * 64, 32).cuda()
        out = model(x)

        self.assertEqual(out.shape, (1, 64 * 64, 32))

    def test_Block(self):
        model = Block(32, 1, 0, hidden_rate=1, key_norm=False).cuda()

        x = torch.randn(1, 32, 64, 64).cuda()
        out = model(x)

        self.assertEqual(out.shape, (1, 32, 64, 64))

    def test_PatchEmbed(self):
        model = PatchEmbed(
            in_channels=3,
            embed_dims=16,
            kernel_size=2,
            stride=2,
            padding=0,
            dilation=1,
            bias=True,
            input_size=64,
        ).cuda()

        x = torch.randn(1, 3, 64, 64).cuda()
        out, out_size = model(x)

        self.assertEqual(out.shape, (1, 32 * 32, 16))
        self.assertEqual(out_size, (32, 32))

    def test_Vision_RWKV(self):
        model = Vision_RWKV(
            embed_dims=256,
            depth=12,
            drop_path_rate=0.0,
            in_channels=3,
            img_size=224,
            patch_size=16,
            interpolation_mode="bicubic",
            drop_after_pos_rate=0.0,
            out_indices=[2, 5, 8, 11],
            final_norm=True,
        ).cuda()

        x = torch.randn(1, 3, 224, 224).cuda()
        out = model(x)

        self.assertEqual(len(out), 4)
        self.assertEqual(out[0].shape, (1, 256, 14, 14))
        self.assertEqual(out[1].shape, (1, 256, 14, 14))
        self.assertEqual(out[2].shape, (1, 256, 14, 14))
        self.assertEqual(out[3].shape, (1, 256, 14, 14))


if __name__ == "__main__":
    unittest.main()
