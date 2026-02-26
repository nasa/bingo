"""Tests for bingo.expressions.agraph.operators"""

import pytest

from bingo.expressions.agraph.operators import (
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
    IS_TERMINAL_MAP,
    IS_ARITY_2_MAP,
    OPERATOR_NAMES,
)

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
        assert op in IS_TERMINAL_MAP, f"{op} missing from IS_TERMINAL_MAP"
        assert op in IS_ARITY_2_MAP, f"{op} missing from IS_ARITY_2_MAP"
        assert op in OPERATOR_NAMES, f"{op} missing from OPERATOR_NAMES"


def test_terminals_identified_correctly():
    terminals = {VARIABLE, CONSTANT, INTEGER}
    for op, is_terminal in IS_TERMINAL_MAP.items():
        if op in terminals:
            assert is_terminal, f"{op} should be terminal"
        else:
            assert not is_terminal, f"{op} should not be terminal"


def test_arity_2_operators():
    binary_ops = {ADDITION, SUBTRACTION, MULTIPLICATION, DIVISION, POWER, SAFE_POWER}
    for op, is_binary in IS_ARITY_2_MAP.items():
        if op in binary_ops:
            assert is_binary, f"{op} should be arity 2"
        else:
            assert not is_binary, f"{op} should not be arity 2"


def test_operator_names_are_non_empty():
    for op, names in OPERATOR_NAMES.items():
        assert len(names) > 0, f"Operator {op} has no names"
