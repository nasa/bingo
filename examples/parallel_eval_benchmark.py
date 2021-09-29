import time
import numpy as np
import cupy as cp
import math
import numba

from bingo.symbolic_regression import AGraphGenerator, ComponentGenerator

from bingo.util.gpu.gpu_evaluation_kernel import _f_eval_gpu_kernel, \
    _f_eval_gpu_kernel_parallel, \
    _f_eval_gpu_kernel_parallel_numba, _f_eval_gpu_kernel_parallel_batch_numba

import nvtx


@nvtx.annotate(color="red")
def set_up_problem(data_dim, num_equations, num_particles, stack_size):
    stacks_for_serial, constants_for_serial = _get_random_stacks_and_constants(
            num_equations, stack_size, data_dim, num_particles)
    max_stack_size = max([len(c) for c in stacks_for_serial])
    constants_for_parallel, stacks_for_parallel, stack_sizes = \
        _make_parallel_data_structures(stacks_for_serial, constants_for_serial,
                                       num_equations, num_particles)
    return constants_for_parallel, constants_for_serial, max_stack_size, \
           stacks_for_parallel, stacks_for_serial, stack_sizes


def _get_random_stacks_and_constants(num_equations, max_stack_size, data_dim,
                                     num_particles):
    USE_SIMPLIFICATION = True
    OPERATORS = ['+', '-', '*', 'sin', 'cos']

    # generation
    component_generator = ComponentGenerator(input_x_dimension=data_dim)
    for comp in OPERATORS:
        component_generator.add_operator(comp)
    generator = AGraphGenerator(max_stack_size, component_generator,
                                use_python=True,
                                use_simplification=USE_SIMPLIFICATION)
    stacks = []
    constants = []

    for _ in range(num_equations):
        equ = generator()
        num_consts = equ.get_number_local_optimization_params()
        constants.append(cp.asarray(np.random.random((num_consts, num_particles))))
        stacks.append(cp.asarray(equ._simplified_command_array))

    return stacks, constants


def _make_parallel_data_structures(stacks_for_serial, constants_for_serial,
                                   num_equations, num_particles):
    stacks_for_parallel = cp.vstack(stacks_for_serial)
    constants_for_parallel = cp.zeros((num_equations, num_particles, 2))
    for i, c in enumerate(constants_for_serial):
        constants_for_parallel[i, :, :c.shape[0]] = c.T
    stack_sizes = cp.asarray(
            np.cumsum([0] + [len(s) for s in stacks_for_serial]))
    return constants_for_parallel, stacks_for_parallel, stack_sizes


@nvtx.annotate(color="blue")
def serial_kernel_calls(stacks, constants, data, data_size, num_equations,
                        num_particles):
    with nvtx.annotate(message="result allocation", color="blue"):
        results = cp.full((num_equations, num_particles, data_size), np.inf)
    blockspergrid = \
        math.ceil(data.shape[0] * num_particles / THREADS_PER_BLOCK)
    for i, (stack, consts) in enumerate(zip(stacks, constants)):
        with nvtx.annotate(message="buffer allocation", color="blue"):
            buffer = cp.full((len(stack), data_size, num_particles), np.inf)
        with nvtx.annotate(message="individual kernel", color="blue"):
            _f_eval_gpu_kernel[blockspergrid, THREADS_PER_BLOCK](
                    stack, data, consts, num_particles, data_size,
                    len(stack), buffer)
            results[i, :, :] = cp.copy(buffer[len(stack) - 1, :, :].T)
    cp.cuda.get_current_stream().synchronize()
    return results


@nvtx.annotate(color="green")
def parallel_kernel_call(constants, data, data_size,
                         max_stack_size, num_equations, num_particles,
                         stacks, stack_sizes):
    with nvtx.annotate(message="buffer allocation", color="green"):
        buffer = cp.full((max_stack_size, num_equations, num_particles, data_size),
                         np.inf)
    with nvtx.annotate(message="result allocation", color="green"):
        results = cp.full((num_equations, num_particles, data_size), np.inf)
    blockspergrid = math.ceil(
            data_size * num_particles * num_equations / THREADS_PER_BLOCK)
    with nvtx.annotate(message="parallel kernel", color="green"):
        _f_eval_gpu_kernel_parallel[blockspergrid, THREADS_PER_BLOCK](
                stacks, data, constants, num_particles,
                data_size, num_equations, stack_sizes, buffer, results)
    cp.cuda.get_current_stream().synchronize()
    return results

@nvtx.annotate(color="green")
def numba_parallel_kernel_call(constants, data, data_size,
                         max_stack_size, num_equations, num_particles,
                         stacks, stack_sizes):
    NUMBA_THREADS_PER_BLOCK = 128
    with nvtx.annotate(message="result allocation", color="green"):
        results = cp.full((num_equations, num_particles, data_size), np.inf)
    blockspergrid = math.ceil(
            data_size * num_particles * num_equations / NUMBA_THREADS_PER_BLOCK)
    with nvtx.annotate(message="parallel kernel", color="green"):
        _f_eval_gpu_kernel_parallel_numba[blockspergrid, NUMBA_THREADS_PER_BLOCK](
                stacks, data, constants, num_particles,
                data_size, num_equations, stack_sizes, results)
    numba.cuda.synchronize()
    return results


@nvtx.annotate(color="green")
def numba_parallel_batch_kernel_call(constants, data, data_size,
                               max_stack_size, num_equations, num_particles,
                               stacks, stack_sizes, batch_size, num_batches):
    NUMBA_THREADS_PER_BLOCK = 128
    with nvtx.annotate(message="result allocation", color="green"):
        results = cp.full((num_equations, num_particles, data_size), np.inf)
    blockspergrid = math.ceil(
        data_size * num_particles * num_batches / NUMBA_THREADS_PER_BLOCK)
    with nvtx.annotate(message="parallel kernel", color="green"):
        _f_eval_gpu_kernel_parallel_batch_numba[blockspergrid, NUMBA_THREADS_PER_BLOCK](
            stacks, data, constants, num_particles,
            data_size, num_equations, stack_sizes, results, batch_size, num_batches)
    numba.cuda.synchronize()
    return results


if __name__ == '__main__':
    print("current memory limit:",
          cp.get_default_memory_pool().get_limit())


    THREADS_PER_BLOCK = 256

    NUM_EQUATIONS = 128
    STACK_SIZE = 64
    NUM_PARTICLES = 800
    DATA_SIZE = 150
    DATA_DIM = 3

    BATCH_SIZE = 128
    NUM_BATCHES = NUM_EQUATIONS // BATCH_SIZE  # might not work if batch size doesn't equally split NUM_EQUATIONS

    # setup
    np.random.seed(10)
    DATA = cp.asarray(np.random.uniform(1, 10, size=(DATA_SIZE, DATA_DIM)))
    CONSTANTS_FOR_PARALLEL, CONSTANTS_FOR_SERIAL, MAX_STACK_SIZE, \
        STACKS_FOR_PARALLEL, STACKS_FOR_SERIAL, STACK_SIZES = set_up_problem(
        DATA_DIM, NUM_EQUATIONS, NUM_PARTICLES, STACK_SIZE)


    # kernel calls
    t0 = time.time()
    serial_kernel_calls(STACKS_FOR_SERIAL, CONSTANTS_FOR_SERIAL,
                            DATA, DATA_SIZE, NUM_EQUATIONS,
                            NUM_PARTICLES)
    t1 = time.time()
    RESULTS1 = serial_kernel_calls(STACKS_FOR_SERIAL, CONSTANTS_FOR_SERIAL,
                                   DATA, DATA_SIZE, NUM_EQUATIONS,
                                   NUM_PARTICLES)
    t2 = time.time()
    parallel_kernel_call(CONSTANTS_FOR_PARALLEL, DATA, DATA_SIZE,
                             MAX_STACK_SIZE, NUM_EQUATIONS,
                             NUM_PARTICLES, STACKS_FOR_PARALLEL,
                             STACK_SIZES)
    t3 = time.time()
    rng = nvtx.start_range(message="parallel_kernel", color="green")
    RESULTS2 = parallel_kernel_call(CONSTANTS_FOR_PARALLEL, DATA, DATA_SIZE,
                                    MAX_STACK_SIZE, NUM_EQUATIONS,
                                    NUM_PARTICLES, STACKS_FOR_PARALLEL,
                                    STACK_SIZES)
    nvtx.end_range(rng)
    t4 = time.time()
    numba_parallel_kernel_call(CONSTANTS_FOR_PARALLEL, DATA, DATA_SIZE,
                             MAX_STACK_SIZE, NUM_EQUATIONS,
                             NUM_PARTICLES, STACKS_FOR_PARALLEL,
                             STACK_SIZES)
    t5 = time.time()
    rng = nvtx.start_range(message="Numba Parallel", color="green")
    RESULTS3 = numba_parallel_kernel_call(CONSTANTS_FOR_PARALLEL, DATA, DATA_SIZE,
                                    MAX_STACK_SIZE, NUM_EQUATIONS,
                                    NUM_PARTICLES, STACKS_FOR_PARALLEL,
                                    STACK_SIZES)
    nvtx.end_range(rng)
    t6 = time.time()

    BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
    for batch_size in BATCH_SIZES:
        n_batches = NUM_EQUATIONS // batch_size
        t7 = time.time()
        numba_parallel_batch_kernel_call(CONSTANTS_FOR_PARALLEL, DATA, DATA_SIZE,
                                         MAX_STACK_SIZE, NUM_EQUATIONS,
                                         NUM_PARTICLES, STACKS_FOR_PARALLEL,
                                         STACK_SIZES, batch_size, n_batches)
        t8 = time.time()
        rng = nvtx.start_range(message=f"Numba Batch Parallel {batch_size}", color="green")
        RESULTS5 = numba_parallel_batch_kernel_call(CONSTANTS_FOR_PARALLEL, DATA, DATA_SIZE,
                                                    MAX_STACK_SIZE, NUM_EQUATIONS,
                                                    NUM_PARTICLES, STACKS_FOR_PARALLEL,
                                                    STACK_SIZES, batch_size, n_batches)
        nvtx.end_range(rng)
        t9 = time.time()
        np.testing.assert_array_almost_equal(RESULTS1.get(), RESULTS5.get())
        print(f"Numba Batch Size {batch_size} Parallel (with compile) time: {t8-t7} seconds")
        print(f"Numba Batch Size {batch_size} Parallel time: {t9 - t8} seconds")
        print(f"Speedup: {(t2-t1)/(t9 - t8)} [Slowdown: {(t9 - t8) / (t2 - t1)}]")


    # display
    np.testing.assert_array_almost_equal(RESULTS1.get(), RESULTS2.get())
    np.testing.assert_array_almost_equal(RESULTS1.get(), RESULTS3.get())
    print("Results Match")

    print(f"Serial (with compile) time: {t1-t0} seconds")
    print(f"Serial time: {t2-t1} seconds")
    print(f"Parallel (with compile) time: {t3-t2} seconds")
    print(f"Parallel time: {t4-t3} seconds")
    print(f"Speedup: {(t2-t1)/(t4-t3)} [Slowdown: {(t4-t3)/(t2-t1)}]")
    print(f"Numba Parallel (with compile) time: {t5-t4} seconds")
    print(f"Numba Parallel time: {t6-t5} seconds")
    print(f"Speedup: {(t2-t1)/(t6-t5)} [Slowdown: {(t6-t5)/(t2-t1)}]")
