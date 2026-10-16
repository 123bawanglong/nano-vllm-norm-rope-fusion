"""Bounded BF16 Q16/K8 head-dim128 fused inference operator (CUDA SM120)."""
from functools import lru_cache
from pathlib import Path
import os
import torch

@lru_cache(None)
def load_extension():
    """Load Dispatcher registrations before Graph capture; return the ops namespace."""
    from torch.utils.cpp_extension import load
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST','12.0')
    os.environ.setdefault('MAX_JOBS','2')
    build=Path.home()/'.cache/nano-vllm-fusion-learning/cuda_qk_norm_rope_dispatcher'
    build.mkdir(parents=True,exist_ok=True)
    load(name='cuda_qk_norm_rope_dispatcher',sources=[str(Path(__file__).with_name('kernel.cu'))],
         build_directory=str(build),extra_cuda_cflags=['-O3','-lineinfo','--fmad=false'],
         extra_cflags=['-O3'],is_python_module=False,verbose=False)
    return torch.ops.qk_norm_rope

prepare=load_extension
build=load_extension

def fused(q,k,q_weight,k_weight,positions,cos_sin_cache,eps,*,warps=4,reduction_mode='standalone'):
    """Return dense Q/K. Supports positive input strides and Graph/current stream.

    Inputs: CUDA BF16 [B,16,128]/[B,8,128], BF16 weights [128],
    int64 positions [B], contiguous float32 cache [P,1,128], finite positive eps.
    Position bounds are device asserted (no synchronous CPU read in Graph).
    Inference only; no autograd or FakeTensor implementation is registered.
    """
    if reduction_mode not in ('standalone','native_graph'):
        raise ValueError('reduction_mode must be standalone or native_graph')
    load_extension()
    return torch.ops.qk_norm_rope.fused(q,k,q_weight,k_weight,positions,cos_sin_cache,float(eps),warps,reduction_mode=='native_graph')
