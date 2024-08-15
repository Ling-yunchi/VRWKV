import torch
from torch import nn


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


class QShift(nn.Module):
    def __init__(self, shift_pixel=1, gamma=1 / 4):
        super(QShift, self).__init__()
        assert gamma <= 1 / 4, "Gamma should be less than or equal to 1/4"
        self.shift_pixel = shift_pixel
        self.gamma = gamma

    def forward(self, input, patch_resolution=None):
        if patch_resolution is None:
            raise ValueError("Patch resolution must be provided")

        B, N, C = input.shape
        input = input.transpose(1, 2).reshape(B, C, *patch_resolution)

        B, C, H, W = input.shape
        output = torch.zeros_like(input)

        # Shift operations
        output[:, : int(C * self.gamma), :, self.shift_pixel : W] = input[
            :, : int(C * self.gamma), :, : W - self.shift_pixel
        ]
        output[
            :, int(C * self.gamma) : int(C * self.gamma * 2), :, : W - self.shift_pixel
        ] = input[
            :, int(C * self.gamma) : int(C * self.gamma * 2), :, self.shift_pixel : W
        ]
        output[
            :,
            int(C * self.gamma * 2) : int(C * self.gamma * 3),
            self.shift_pixel : H,
            :,
        ] = input[
            :,
            int(C * self.gamma * 2) : int(C * self.gamma * 3),
            : H - self.shift_pixel,
            :,
        ]
        output[
            :,
            int(C * self.gamma * 3) : int(C * self.gamma * 4),
            : H - self.shift_pixel,
            :,
        ] = input[
            :,
            int(C * self.gamma * 3) : int(C * self.gamma * 4),
            self.shift_pixel : H,
            :,
        ]
        output[:, int(C * self.gamma * 4) :, ...] = input[
            :, int(C * self.gamma * 4) :, ...
        ]

        return output.flatten(2).transpose(1, 2)
