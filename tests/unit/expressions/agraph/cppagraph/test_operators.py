"""Tests for cppagraph operator constants — must match pyagraph exactly."""

import pytest

from bingo.expressions.agraph.cppagraph import (
    VARIABLE,
    CONSTANT,
    INTEGER,
    ADDITION,
    SUBTRACTION,
    MULTIPLICATION,
    DIVISION,
    POWER,
    SAFE_POWER,
    SQUARE,
    CUBE,
    SQRT,
    ABS,
    EXPONENTIAL,
    LOGARITHM,
    SIN,
    COS,
    TAN,
    ARCSIN,
    ARCCOS,
    ARCTAN,
    SINH,
    COSH,
    TANH,
    TERMINAL_IDS,
    ARITY_2_IDS,
    IS_TERMINAL_ARRAY,
    IS_ARITY_2_ARRAY,
    OPERATOR_NAMES,
)

from bingo.expressions.agraph.pyagraph import operators as py_ops

ALL_OPS = [
    VARIABLE,
    CONSTANT,
    INTEGER,
    ADDITION,
    SUBTRACTION,
    MULTIPLICATION,
    DIVISION,
    POWER,
    SAFE_POWER,
    SQUARE,
    CUBE,
    SQRT,
    ABS,
    EXPONENTIAL,
    LOGARITHM,
    SIN,
    COS,
    TAN,
    ARCSIN,
    ARCCOS,
    ARCTAN,
    SINH,
    COSH,
    TANH,
]


# ================================================================== #
#  Cross-check: every constant matches pyagraph                       #
# ================================================================== #


@pytest.mark.parametrize(
    "name",
    [
        "VARIABLE",
        "CONSTANT",
        "INTEGER",
        "ADDITION",
        "SUBTRACTION",
        "MULTIPLICATION",
        "DIVISION",
        "POWER",
        "SAFE_POWER",
        "SQUARE",
        "CUBE",
        "SQRT",
        "ABS",
        "EXPONENTIAL",
        "LOGARITHM",
        "SIN",
        "COS",
        "TAN",
        "ARCSIN",
        "ARCCOS",
        "ARCTAN",
        "SINH",
        "COSH",
        "TANH",
    ],
)
def test_operator_id_matches_pyagraph(name):
    cpp_val = globals()[name]
    py_val = getattr(py_ops, name)
    assert cpp_val == py_val, f"{name}: cpp={cpp_val} != py={py_val}"


# ================================================================== #
#  Basic operator properties (same tests as pyagraph)                 #
# ================================================================== #


def test_all_ids_are_non_negative():
    for op in ALL_OPS:
        assert op >= 0, f"Operator {op} is negative"


def test_all_ids_fit_in_uint8():
    for op in ALL_OPS:
        assert op <= 255, f"Operator {op} exceeds uint8 range"


def test_no_duplicate_ids():
    assert len(ALL_OPS) == len(set(ALL_OPS))


def test_maps_cover_all_operators():
    for op in ALL_OPS:
        assert (
            IS_TERMINAL_ARRAY.shape[0] > op
        ), f"{op} out of range for IS_TERMINAL_ARRAY"
        assert IS_ARITY_2_ARRAY.shape[0] > op, f"{op} out of range for IS_ARITY_2_ARRAY"
        assert op in OPERATOR_NAMES, f"{op} missing from OPERATOR_NAMES"


def test_terminals_identified_correctly():
    terminals = {VARIABLE, CONSTANT, INTEGER}
    assert TERMINAL_IDS == terminals
    for op in ALL_OPS:
        if op in terminals:
            assert IS_TERMINAL_ARRAY[op], f"{op} should be terminal in array"
        else:
            assert not IS_TERMINAL_ARRAY[op], f"{op} should not be terminal in array"


def test_arity_2_operators():
    binary_ops = {ADDITION, SUBTRACTION, MULTIPLICATION, DIVISION, POWER, SAFE_POWER}
    assert ARITY_2_IDS == binary_ops
    for op in ALL_OPS:
        if op in binary_ops:
            assert IS_ARITY_2_ARRAY[op], f"{op} should be arity 2 in array"
        else:
            assert not IS_ARITY_2_ARRAY[op], f"{op} should not be arity 2 in array"


def test_operator_names_are_non_empty():
    for op, names in OPERATOR_NAMES.items():
        assert len(names) > 0, f"Operator {op} has no names"


# ================================================================== #
#  Cross-check: sets and arrays match pyagraph                        #
# ================================================================== #


def test_terminal_ids_match_pyagraph():
    assert TERMINAL_IDS == py_ops.TERMINAL_IDS


def test_arity_2_ids_match_pyagraph():
    assert ARITY_2_IDS == py_ops.ARITY_2_IDS


def test_is_terminal_array_matches_pyagraph():
    import numpy as np

    np.testing.assert_array_equal(IS_TERMINAL_ARRAY, py_ops.IS_TERMINAL_ARRAY)


def test_is_arity_2_array_matches_pyagraph():
    import numpy as np

    np.testing.assert_array_equal(IS_ARITY_2_ARRAY, py_ops.IS_ARITY_2_ARRAY)


def test_operator_names_match_pyagraph():
    assert OPERATOR_NAMES == py_ops.OPERATOR_NAMES
