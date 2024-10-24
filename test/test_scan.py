import unittest

import torch

from model.scan import (
    s_hw,
    sr_hw,
    s_wh,
    sr_wh,
    s_rhw,
    sr_rhw,
    s_hrw,
    sr_hrw,
    s_rwh,
    sr_rwh,
    s_wrh,
    sr_wrh,
    s_rhrw,
    sr_rhrw,
    s_rwrh,
    sr_rwrh,
)


class TestScan(unittest.TestCase):
    def setUp(self):
        # 3x3 矩阵，用作测试
        self.tensor = torch.tensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]])
        self.h, self.w = 3, 3

    def test_s_hw(self):
        expected = torch.tensor([[[1], [2], [3], [4], [5], [6], [7], [8], [9]]])
        result = s_hw(self.tensor)
        self.assertTrue(torch.equal(result, expected))

    def test_sr_hw(self):
        flat_tensor = s_hw(self.tensor)
        result = sr_hw(flat_tensor, self.h, self.w)
        self.assertTrue(torch.equal(result, self.tensor))

    def test_s_wh(self):
        expected = torch.tensor([[[1], [4], [7], [2], [5], [8], [3], [6], [9]]])
        result = s_wh(self.tensor)
        self.assertTrue(torch.equal(result, expected))

    def test_sr_wh(self):
        flat_tensor = s_wh(self.tensor)
        result = sr_wh(flat_tensor, self.h, self.w)
        self.assertTrue(torch.equal(result, self.tensor))

    def test_s_rhw(self):
        expected = torch.tensor([[[7], [8], [9], [4], [5], [6], [1], [2], [3]]])
        result = s_rhw(self.tensor)
        self.assertTrue(torch.equal(result, expected))

    def test_sr_rhw(self):
        flat_tensor = s_rhw(self.tensor)
        result = sr_rhw(flat_tensor, self.h, self.w)
        self.assertTrue(torch.equal(result, self.tensor))

    def test_s_hrw(self):
        expected = torch.tensor([[[3], [2], [1], [6], [5], [4], [9], [8], [7]]])
        result = s_hrw(self.tensor)
        self.assertTrue(torch.equal(result, expected))

    def test_sr_hrw(self):
        flat_tensor = s_hrw(self.tensor)
        result = sr_hrw(flat_tensor, self.h, self.w)
        self.assertTrue(torch.equal(result, self.tensor))

    def test_s_rwh(self):
        expected = torch.tensor([[[3], [6], [9], [2], [5], [8], [1], [4], [7]]])
        result = s_rwh(self.tensor)
        self.assertTrue(torch.equal(result, expected))

    def test_sr_rwh(self):
        flat_tensor = s_rwh(self.tensor)
        result = sr_rwh(flat_tensor, self.h, self.w)
        self.assertTrue(torch.equal(result, self.tensor))

    def test_s_wrh(self):
        expected = torch.tensor([[[7], [4], [1], [8], [5], [2], [9], [6], [3]]])
        result = s_wrh(self.tensor)
        self.assertTrue(torch.equal(result, expected))

    def test_sr_wrh(self):
        flat_tensor = s_wrh(self.tensor)
        result = sr_wrh(flat_tensor, self.h, self.w)
        self.assertTrue(torch.equal(result, self.tensor))

    def test_s_rhrw(self):
        expected = torch.tensor([[[9], [8], [7], [6], [5], [4], [3], [2], [1]]])
        result = s_rhrw(self.tensor)
        self.assertTrue(torch.equal(result, expected))

    def test_sr_rhrw(self):
        flat_tensor = s_rhrw(self.tensor)
        result = sr_rhrw(flat_tensor, self.h, self.w)
        self.assertTrue(torch.equal(result, self.tensor))

    def test_s_rwrh(self):
        expected = torch.tensor([[[9], [6], [3], [8], [5], [2], [7], [4], [1]]])
        result = s_rwrh(self.tensor)
        self.assertTrue(torch.equal(result, expected))

    def test_sr_rwrh(self):
        flat_tensor = s_rwrh(self.tensor)
        result = sr_rwrh(flat_tensor, self.h, self.w)
        self.assertTrue(torch.equal(result, self.tensor))


if __name__ == "__main__":
    unittest.main()
