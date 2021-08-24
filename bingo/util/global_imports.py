import numpy as num_lib
gpu_eval = None

USING_PARALLEL_CPU = False
USING_GPU = False

GPU_THREADS_PER_BLOCK = 256

def set_use_gpu(use_gpu):
    global num_lib
    global gpu_eval
    global USING_GPU

    USING_GPU = use_gpu
    if use_gpu:
        import cupy as num_lib
        import bingo.util.gpu.gpu_evaluation_kernel as gpu_eval
    else:
        import numpy as num_lib

jl = None


def set_use_parallel_cpu(flag):
    global jl
    global USING_PARALLEL_CPU

    USING_PARALLEL_CPU = flag
    if flag:
        import joblib as jl