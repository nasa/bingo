# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import inspect
import sys

from mpi4py import MPI

COMM = MPI.COMM_WORLD
COMM_RANK = COMM.Get_rank()
COMM_SIZE = COMM.Get_size()


def mpi_assert_equal(actual, expected):
    equal = actual == expected
    if not equal:
        message = f"\tproc {COMM_RANK}:  {actual} != {expected}\n"
    else:
        message = ""
    all_equals = COMM.allgather(equal)
    all_messages = COMM.allreduce(message, op=MPI.SUM)
    return all(all_equals), all_messages


def mpi_assert_true(value):
    if not value:
        message = f"\tproc {COMM_RANK}: False, expected True\n"
    else:
        message = ""
    all_values = COMM.allgather(value)
    all_messages = COMM.allreduce(message, op=MPI.SUM)
    return all(all_values), all_messages


def mpi_assert_exactly_n_false(value, n):
    all_values = COMM.allgather(value)
    if sum(all_values) == len(all_values) - n:
        return True, ""

    message = f"\tproc {COMM_RANK}: {value}\n"
    all_messages = COMM.allreduce(message, op=MPI.SUM)
    all_messages = "\tExpected exactly " + str(n) + " False\n" + all_messages
    return False, all_messages


def mpi_assert_mean_near(value, expected_mean, rel=1e-6, abs=None):
    actual_mean = COMM.allreduce(value, op=MPI.SUM)
    actual_mean /= COMM_SIZE
    allowable_error = rel * expected_mean
    if abs is not None:
        allowable_error = max(allowable_error, abs)

    if -allowable_error <= actual_mean - expected_mean <= allowable_error:
        return True, ""

    message = f"\tproc {COMM_RANK}:  {value}\n"
    all_messages = COMM.allreduce(message, op=MPI.SUM)
    all_messages += f"\tMean {actual_mean} != {expected_mean} +- " \
                    f"{allowable_error}"
    return False, all_messages


def run_t(test_name, test_func):
    if COMM_RANK == 0:
        print(test_name, end=" ")
    COMM.barrier()
    success, message = test_func()
    COMM.barrier()
    if success:
        if COMM_RANK == 0:
            print(".")
    else:
        if COMM_RANK == 0:
            print("F")
            print(message, end=" ")
    return success


def run_t_in_module(module_name):
    results = []
    tests = [(name, func)
             for name, func in inspect.getmembers(sys.modules[module_name],
                                                  inspect.isfunction)
             if "test" in name]
    if COMM_RANK == 0:
        print("========== collected", len(tests), "items ==========")

    for name, func in tests:
        results.append(run_t(name, func))

    num_success = sum(results)
    num_failures = len(results) - num_success
    if COMM_RANK == 0:
        print("==========", end="  ")
        if num_failures > 0:
            print(num_failures, "failed,", end=" ")
        print(num_success, "passed ==========")

    if num_failures > 0:
        sys.exit(-1)
