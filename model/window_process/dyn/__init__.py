import os

from torch.utils.cpp_extension import load

this_dir = os.path.dirname(os.path.abspath(__file__))
extensions_dir = os.path.join(this_dir, "..")

sources = [
    os.path.join(extensions_dir, f)
    for f in ["swin_window_process.cpp", "swin_window_process_kernel.cu"]
]

swin_window_process = load(name="swin_window_process", sources=sources)

__all__ = ["swin_window_process"]
