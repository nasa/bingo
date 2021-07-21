import numpy as num_lib

USING_PARALLEL_CPU = False
USING_GPU = False

def set_use_gpu(use_gpu):
    global num_lib
    global USING_GPU
    USING_GPU = use_gpu
    if use_gpu:
        import cupy as num_lib
    else:
        import numpy as num_lib

jl = None


def set_use_parallel_cpu(flag):
    global jl
    global USING_PARALLEL_CPU
    USING_PARALLEL_CPU = flag
    if flag:
        import joblib as jl