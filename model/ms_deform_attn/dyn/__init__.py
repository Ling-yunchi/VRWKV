import glob
import os

import torch
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import load


def load_extension():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, '../src')

    main_file = glob.glob(os.path.join(extensions_dir, '*.cpp'))
    source_cpu = glob.glob(os.path.join(extensions_dir, 'cpu', '*.cpp'))
    source_cuda = glob.glob(os.path.join(extensions_dir, 'cuda', '*.cu'))

    sources = main_file + source_cpu
    extra_cflags = []
    extra_cuda_cflags = []

    if torch.cuda.is_available() and torch.utils.cpp_extension.CUDA_HOME is not None:
        sources += source_cuda
        extra_cflags.append('-DWITH_CUDA')
        extra_cuda_cflags = [
            '-DCUDA_HAS_FP16=1',
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
    else:
        raise NotImplementedError('Cuda is not available')

    sources = [os.path.join(extensions_dir, s) for s in sources]
    extra_include_paths = [extensions_dir]

    return load(
        name='MultiScaleDeformableAttention',
        sources=sources,
        extra_include_paths=extra_include_paths,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
    )


# 加载扩展
MultiScaleDeformableAttention = load_extension()

__all__ = ['MultiScaleDeformableAttention']
