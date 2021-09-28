import numpy as np
import cupy as cp
import bingo.util.global_imports as gi
import math


from bingo.util.gpu.gpu_evaluation_kernel import _f_eval_gpu_kernel, \
                                                 _f_eval_gpu_kernel_parallel


if __name__ == "__main__":
    NUM_PARTICLES = 4

    DATA_SIZE = 5
    DATA = cp.arange(10, dtype=float).reshape(DATA_SIZE, 2)

    STACKS_FOR_SERIAL = [cp.array([[0, 0, 0],
                                   [1, 0, 0],
                                   [2, 0, 1]]),  # x + a
                         cp.array([[0, 0, 0],
                                   [0, 1, 1],
                                   [1, 0, 0],
                                   [1, 1, 1],
                                   [4, 0, 2],
                                   [2, 4, 1],
                                   [2, 5, 3]]),  # a*x + y + b
                         cp.array([[0, 0, 0],
                                   [4, 0, 0]]),  # x*x
                         cp.array([[1, 0, 0],
                                   [4, 0, 0]]),  # a*a
                         cp.array([[0, 0, 0],
                                   [0, 0, 0]]),  # x
                         cp.array([[1, 0, 0],
                                   [1, 0, 0]]),  # a
                         ]

    # column-major for future implementation TODO
    CONSTANTS_FOR_SERIAL = [cp.linspace(1, NUM_PARTICLES, num=NUM_PARTICLES).reshape(1, NUM_PARTICLES),
                            cp.linspace(1, NUM_PARTICLES*2, num=NUM_PARTICLES*2).reshape(2, NUM_PARTICLES),  #  TODO Try with different end range
                            cp.empty((0, NUM_PARTICLES)),
                            cp.linspace(1, NUM_PARTICLES, num=NUM_PARTICLES).reshape(1, NUM_PARTICLES),
                            cp.empty((0, NUM_PARTICLES)),
                            cp.linspace(1, NUM_PARTICLES, num=NUM_PARTICLES).reshape(1, NUM_PARTICLES),
                            ]

    # STACKS_FOR_SERIAL = STACKS_FOR_SERIAL[-2:]
    # CONSTANTS_FOR_SERIAL = CONSTANTS_FOR_SERIAL[-2:]
    NUM_STACKS = len(STACKS_FOR_SERIAL)
    MAX_STACK_SIZE = max([len(c) for c in STACKS_FOR_SERIAL])
    print("MAX STACK SIZE", MAX_STACK_SIZE)


    # current split kernel
    BUFFER1 = cp.full((MAX_STACK_SIZE, DATA_SIZE, NUM_PARTICLES), np.inf)
    RESULTS1 = cp.full((NUM_STACKS, NUM_PARTICLES, DATA_SIZE), np.inf)
    blockspergrid = math.ceil(DATA.shape[0] * NUM_PARTICLES / gi.GPU_THREADS_PER_BLOCK)
    for i, (stack, consts) in enumerate(zip(STACKS_FOR_SERIAL, CONSTANTS_FOR_SERIAL)):
        _f_eval_gpu_kernel[blockspergrid, gi.GPU_THREADS_PER_BLOCK](
                stack, DATA, consts, NUM_PARTICLES, DATA_SIZE,
                len(stack), BUFFER1)
        RESULTS1[i, :, :] = cp.copy(BUFFER1[len(stack)-1, :, :].T)
    print("SERIEAL RESULTS")
    print(RESULTS1)


    # joined kernel
    BUFFER2 = cp.full((MAX_STACK_SIZE, NUM_STACKS, NUM_PARTICLES, DATA_SIZE), np.inf)
    STACKS_FOR_PARALLEL = cp.vstack(STACKS_FOR_SERIAL)
    CONSTANTS_FOR_PARALLEL = cp.zeros((NUM_STACKS, NUM_PARTICLES, 2))
    for i, c in enumerate(CONSTANTS_FOR_SERIAL):
        CONSTANTS_FOR_PARALLEL[i, :, :c.shape[0]] = c.T
    STACK_SIZES = cp.asarray(np.cumsum([0] + [len(s) for s in STACKS_FOR_SERIAL]))
    print("STACK SIZES", STACK_SIZES)


    RESULTS2 = cp.full((NUM_STACKS, NUM_PARTICLES, DATA_SIZE), np.inf)
    blockspergrid = math.ceil(DATA_SIZE * NUM_PARTICLES * NUM_STACKS / gi.GPU_THREADS_PER_BLOCK)
    _f_eval_gpu_kernel_parallel[blockspergrid, gi.GPU_THREADS_PER_BLOCK](
            STACKS_FOR_PARALLEL, DATA, CONSTANTS_FOR_PARALLEL, NUM_PARTICLES,
            DATA_SIZE, NUM_STACKS, STACK_SIZES, BUFFER2, RESULTS2)
    print("PARALLEL RESULTS")
    print(RESULTS2)
    #
    #
    # test RESULTS1 == RESULTS2

