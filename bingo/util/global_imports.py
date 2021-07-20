import numpy as num_lib

def set_use_gpu(use_gpu):
    global num_lib
    if use_gpu:
        import cupy as num_lib
    else:
        import numpy as num_lib

jl = None
USING_PARALLEL_CPU = False

def set_use_parallel_cpu(flag):
    global jl
    global USING_PARALLEL_CPU
    USING_PARALLEL_CPU = flag
    if flag:
        import joblib as jl