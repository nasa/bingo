from cupyx import jit
import cupy as cp
import bingo.symbolic_regression.agraph.operator_definitions as defs
import math
import bingo.util.global_imports as gi
import numpy as np
import nvtx


# @nvtx.annotate(message="gpu_eval_wrapper", color="orange")
def f_eval_gpu_with_kernel(stack, x, constants):

    with nvtx.annotate(message="setup", color="red"):
        num_particles = 1
        if hasattr(constants, 'shape'):
            num_particles = constants.shape[1]
        elif isinstance(constants, tuple):
            if len(constants) == 0:
                constants = cp.asarray([[]])
            else:
                print("before", constants)
                constants = cp.stack(constants, axis=0)
                num_particles = constants.shape[1]
                print("after", constants)


    with nvtx.annotate(message="setup_2", color="red"):
        forward_eval = cp.full((len(stack), x.shape[0], num_particles), np.inf)
        blockspergrid = math.ceil(x.shape[0] * num_particles / gi.GPU_THREADS_PER_BLOCK)


    with nvtx.annotate(message="kernel_launch", color="red"):
        _f_eval_gpu_kernel[blockspergrid, gi.GPU_THREADS_PER_BLOCK](stack, x, constants, num_particles, x.shape[0],
                                                                    stack.shape[0], forward_eval)
    return forward_eval


@jit.rawkernel()
def _f_eval_gpu_kernel(stack, x, constants, num_particles, data_size, stack_size, f_eval_arr):
    index = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    # fwd_buff = [1.0]*stack_size

    if index < data_size * num_particles:
        data_index = index // num_particles
        constant_index = index % num_particles



        for i in range(stack_size):
            node = stack[i, 0]
            param1 = stack[i, 1]
            param2 = stack[i, 2]

            if node == defs.INTEGER:
                f_eval_arr[i, data_index, constant_index] = float(param1)
            elif node == defs.VARIABLE:
                f_eval_arr[i, data_index, constant_index] = x[data_index, param1]
                # fwd_buff[i] = x[data_index, param1]
            elif node == defs.CONSTANT:
                #if num_particles < 2: # case doesn't work for some reason
                #    f_eval_arr[i, data_index, constant_index] = constants[int(param1)]
                f_eval_arr[i, data_index, constant_index] = constants[int(param1), constant_index]
            elif node == defs.ADDITION:
                f_eval_arr[i, data_index, constant_index] = f_eval_arr[int(param1), data_index, constant_index] + \
                                                            f_eval_arr[int(param2), data_index, constant_index]
            elif node == defs.SUBTRACTION:
                f_eval_arr[i, data_index, constant_index] = f_eval_arr[int(param1), data_index, constant_index] - \
                                                            f_eval_arr[int(param2), data_index, constant_index]
            elif node == defs.MULTIPLICATION:
                f_eval_arr[i, data_index, constant_index] = f_eval_arr[int(param1), data_index, constant_index] * \
                                                            f_eval_arr[int(param2), data_index, constant_index]
            elif node == defs.DIVISION:
                f_eval_arr[i, data_index, constant_index] = f_eval_arr[int(param1), data_index, constant_index] / \
                                                            f_eval_arr[int(param2), data_index, constant_index]
            elif node == defs.SIN:
                f_eval_arr[i, data_index, constant_index] = cp.sin(f_eval_arr[int(param1), data_index, constant_index])
            elif node == defs.COS:
                f_eval_arr[i, data_index, constant_index] = cp.cos(f_eval_arr[int(param1), data_index, constant_index])
            elif node == defs.EXPONENTIAL:
                f_eval_arr[i, data_index, constant_index] = cp.exp(f_eval_arr[int(param1), data_index, constant_index])
            elif node == defs.LOGARITHM:
                f_eval_arr[i, data_index, constant_index] = cp.log(cp.abs(f_eval_arr[int(param1), data_index, constant_index]))
            elif node == defs.POWER:
                f_eval_arr[i, data_index, constant_index] = cp.power(f_eval_arr[int(param1), data_index, constant_index],
                                                                     f_eval_arr[int(param2), data_index, constant_index])
            elif node == defs.ABS:
                f_eval_arr[i, data_index, constant_index] = cp.abs(f_eval_arr[int(param1), data_index, constant_index])
            elif node == defs.SQRT:
                f_eval_arr[i, data_index, constant_index] = cp.sqrt(f_eval_arr[int(param1), data_index, constant_index])

