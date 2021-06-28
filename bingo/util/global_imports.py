import numpy as num_lib

def set_use_gpu(use_gpu):
    global num_lib
    if use_gpu:
        import cupy as num_lib
    else:
        import numpy as num_lib