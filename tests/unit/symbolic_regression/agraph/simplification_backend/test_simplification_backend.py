# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import numpy as np
import pytest

from bingo.symbolic_regression.agraph.operator_definitions import *
from bingo.symbolic_regression.agraph.simplification_backend \
    import simplification_backend as py_simp_backend

try:
    from bingocpp.build import symbolic_regression as cpp_simp_backend
except ImportError:
    cpp_simp_backend = None

CPP_PARAM = pytest.param("Cpp",
                         marks=pytest.mark.skipif(not cpp_simp_backend,
                                                  reason='BingoCpp import '
                                                         'failure'))


@pytest.fixture(params=["Python", CPP_PARAM])
def engine(request):
    return request.param


@pytest.fixture
def simp_backend(engine):
    if engine == "Python":
        return py_simp_backend
    return cpp_simp_backend


@pytest.fixture
def sample_command_array():
    return np.array([[VARIABLE, 0, 0],
                     [VARIABLE, 1, 1],
                     [ADDITION, 0, 1],
                     [CONSTANT, 0, 0],
                     [SIN, 2, 3],
                     [CONSTANT, 0, 0],
                     [ADDITION, 4, 4]])


def test_utilized_commands(simp_backend, sample_command_array):
    util = simp_backend.get_utilized_commands(sample_command_array)
    expected_util = [True, True, True, False, True, False, True]
    np.testing.assert_array_equal(util, expected_util)


def test_reduce_stack(simp_backend, sample_command_array):
    reduction = simp_backend.reduce_stack(sample_command_array)
    expected_reduction = np.array([[VARIABLE, 0, 0],
                                   [VARIABLE, 1, 1],
                                   [ADDITION, 0, 1],
                                   [SIN, 2, 2],
                                   [ADDITION, 3, 3]])
    np.testing.assert_array_equal(reduction, expected_reduction)