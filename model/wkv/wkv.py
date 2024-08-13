import os

import torch
from torch.utils.cpp_extension import load

T_MAX = 8192  # increase this if your ctx_len is long [NOTE: TAKES LOTS OF VRAM!]
# it's possible to go beyond CUDA limitations if you slice the ctx and pass the hidden state in each slice

file_path = os.path.dirname(os.path.realpath(__file__))
is_windows = os.name == "nt"

# fmt: off
wkv_cuda = load(
    name="wkv",
    sources=[f"{file_path}/cuda/wkv_op.cpp", f"{file_path}/cuda/wkv_cuda.cu"],
    verbose=True,
    build_directory=f"{file_path}/cuda/build",
    extra_cuda_cflags=[
        "-res-usage",
        f"--maxrregcount{'=' if is_windows else ' '}60",
        "--use_fast_math",
        "-O3",
        "-Xptxas", "-O3",
        f"-DTmax={T_MAX}",
    ],
)
# fmt: on


class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0

        half_mode = w.dtype == torch.half
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()

        ctx.save_for_backward(w, u, k, v)
        y = torch.empty((B, T, C), device="cuda", memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        if half_mode:
            y = y.half()
        return y

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device="cuda").contiguous()
        gu = torch.zeros((B, C), device="cuda").contiguous()
        gk = torch.zeros((B, T, C), device="cuda").contiguous()
        gv = torch.zeros((B, T, C), device="cuda").contiguous()
        half_mode = gy.dtype == torch.half
        wkv_cuda.backward(B, T, C, w, u, k, v, gy.contiguous(), gw, gu, gk, gv)
        if half_mode:
            gw = torch.sum(gw.half(), dim=0)
            gu = torch.sum(gu.half(), dim=0)
            return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
        else:
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            return (None, None, None, gw, gu, gk, gv)


def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())


if __name__ == "__main__":
    B, T, C = 2, 64, 512
    w = torch.randn((B, C), device="cuda")
    u = torch.randn((B, C), device="cuda")
    k = torch.randn((B, T, C), device="cuda")
    v = torch.randn((B, T, C), device="cuda")
    y = RUN_CUDA(B, T, C, w, u, k, v)
    print(y)
