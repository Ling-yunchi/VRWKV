import torch


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

class QShift(torch.nn.Module):
    def __init__(self, channel_gamma=1/4, shift_pixel=1, patch_resolution=None):
        super().__init__()
        self.channel_gamma = channel_gamma
        self.shift_pixel = shift_pixel
        self.patch_resolution = patch_resolution
