import pytest
import numpy as np

from bingo.symbolic_regression.agraph import backend as PythonBackend
from bingo.symbolic_regression.agraph.agraph import AGraph

try:
    from bingocpp.build import bingocpp as CppBackend
    CPP_LOADED = True
except ImportError:
    CppBackend = None
    CPP_LOADED = False


@pytest.fixture(params=[
    PythonBackend,
    # pytest.param(CppBackend,
    #              marks=pytest.mark.skipif(not CPP_LOADED,
    #                                       reason='BingoCpp import failure'))
])
def backend(request):
    return request.param


def test_simplification_1(backend):
    stack = np.array([[1, -1, -1],
                      [4, 0, 0],
                      [2, 1, 0],
                      [3, 1, 0],
                      [1, -1, -1],
                      [0, 0, 0],
                      [2, 2, 4],
                      [2, 6, 3],
                      [3, 7, 5],
                      [2, 8, 2],
                      [3, 4, 9],
                      ])
    agraph = AGraph()
    agraph.command_array = backend.simplify_stack(stack)
    agraph_str = str(agraph)
    print(agraph_str)
    assert agraph_str == "1.0 + X_0"


def test_simplification_2(backend):
    stack = np.array([[1, -1, -1],
                      [0, 0, 0],
                      [1, -1, -1],
                      [2, 1, 1],
                      [3, 3, 3],
                      [3, 2, 0],
                      [4, 5, 4],
                      [3, 6, 5],
                      ])
    agraph = AGraph()
    agraph.command_array = backend.simplify_stack(stack)
    agraph_str = str(agraph)
    print(agraph_str)
    assert agraph_str == "1.0"


def test_simplification_3(backend):
    stack = np.array([[1, -1, -1],
                      [0, 0, 0],
                      [3, 0, 0],
                      [2, 2, 2],
                      [4, 3, 2],
                      [3, 4, 0],
                      [3, 1, 3],
                      [4, 6, 2],
                      [4, 5, 7],
                      [2, 6, 2],
                      [2, 8, 9],
                      [3, 10, 0],
                      [3, 11, 4],
                      [1, -1, -1],
                      [2, 12, 13],
                      ])
    agraph = AGraph()
    agraph.command_array = backend.simplify_stack(stack)
    agraph_str = str(agraph)
    print(agraph_str)
    assert agraph_str == "1.0 + X_0"
