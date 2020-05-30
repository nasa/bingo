# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
from collections import namedtuple
import pytest
import numpy as np

from bingo.symbolic_regression.agraph.evaluation_backend import \
    evaluation_backend as PythonBackend

try:
    from bingocpp.build import symbolic_regression as CppBackend
    CPP_LOADED = True
except ImportError:
    CppBackend = None
    CPP_LOADED = False


@pytest.fixture(params=[
    PythonBackend,
    pytest.param(CppBackend,
                 marks=pytest.mark.skipif(not CPP_LOADED,
                                          reason='BingoCpp import failure'))
])
def backend(request):
    return request.param


@pytest.fixture
def sample_agraph_values():
    values = namedtuple('Point', ['x', 'constants'])
    x = np.vstack((np.linspace(-1.0, 0.0, 11),
                   np.linspace(0.0, 1.0, 11))).transpose()
    constants = [10, 3.14]
    return values(x, constants)





@pytest.fixture
def sample_stack():
    test_stack = np.array([[0, 0, 0],  # sin(X_0) + 1.0
                           [1, 0, 0],
                           [2, 0, 1],
                           [6, 0, 2],
                           [2, 3, 1]])
    return test_stack


@pytest.fixture
def all_funcs_stack():
    test_stack = np.array([[0, 0, 0],
                           [1, 0, 0],
                           [2, 1, 0],
                           [3, 2, 0],
                           [4, 3, 0],
                           [5, 4, 0],
                           [6, 5, 0],
                           [7, 6, 0],
                           [8, 7, 0],
                           [9, 8, 0],
                           [10, 9, 0],
                           [11, 10, 0],
                           [12, 11, 0]])
    return test_stack


@pytest.fixture(params=['all_funcs_stack', 'sample_stack'])
def expected_stack_util(request):
    prop = {'stack': request.getfixturevalue(request.param)}
    if request.param == "all_funcs_stack":
        prop["util"] = np.ones(13, bool)
    elif request.param == "sample_stack":
        prop["util"] = [True, True, False, True, True]
    return prop


def test_cpp_backend_could_be_imported():
    assert CPP_LOADED





def test_agraph_get_utilized_commands(backend, expected_stack_util):
    np.testing.assert_array_equal(
        backend.get_utilized_commands(expected_stack_util["stack"]),
        expected_stack_util["util"])


@pytest.mark.parametrize("the_backend, expected", [
    (PythonBackend, False),
    pytest.param(CppBackend, True,
                 marks=pytest.mark.skipif(not CPP_LOADED,
                                          reason='BingoCpp import failure'))
])
def test_agraph_backend_identifiers(the_backend, expected):
    assert the_backend.is_cpp() == expected
