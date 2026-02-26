"""Tests for bingo.expressions.agraph.simplification"""

import numpy as np
import pytest

from bingo.expressions.agraph.operators import (
    VARIABLE,
    CONSTANT,
    INTEGER,
    ADDITION,
    MULTIPLICATION,
    SIN,
)
from bingo.expressions.agraph.simplification import (
    get_utilized_commands,
    reduce_stack,
)


class TestGetUtilizedCommands:
    def test_single_terminal(self):
        stack = np.array([[VARIABLE, 0, 0]], dtype=np.uint8)
        util = get_utilized_commands(stack)
        assert util == bytearray([1])

    def test_unused_command(self):
        # Row 0: X0, Row 1: X1 (unused), Row 2: sin(X0)
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [VARIABLE, 1, 1],
                [SIN, 0, 0],
            ],
            dtype=np.uint8,
        )
        util = get_utilized_commands(stack)
        assert util == bytearray([1, 0, 1])

    def test_binary_op_marks_both_children(self):
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [CONSTANT, 0, 0],
                [ADDITION, 0, 1],
            ],
            dtype=np.uint8,
        )
        util = get_utilized_commands(stack)
        assert util == bytearray([1, 1, 1])

    def test_chain_of_dependencies(self):
        # X0, X1, X0+X1, sin(X0+X1)
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [VARIABLE, 1, 1],
                [ADDITION, 0, 1],
                [SIN, 2, 2],
            ],
            dtype=np.uint8,
        )
        util = get_utilized_commands(stack)
        assert util == bytearray([1, 1, 1, 1])


class TestReduceStack:
    def test_no_reduction_needed(self):
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [CONSTANT, 0, 0],
                [ADDITION, 0, 1],
            ],
            dtype=np.uint8,
        )
        reduced = reduce_stack(stack)
        np.testing.assert_array_equal(reduced, stack)

    def test_removes_unused_row(self):
        # Row 0: X0, Row 1: X1 (unused), Row 2: sin(X0)
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [VARIABLE, 1, 1],
                [SIN, 0, 0],
            ],
            dtype=np.uint8,
        )
        reduced = reduce_stack(stack)
        expected = np.array(
            [
                [VARIABLE, 0, 0],
                [SIN, 0, 0],
            ],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(reduced, expected)

    def test_remaps_references(self):
        # Row 0: X0, Row 1: X1 (unused), Row 2: X1, Row 3: X0 + X1
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [VARIABLE, 1, 1],  # unused
                [VARIABLE, 1, 1],
                [ADDITION, 0, 2],
            ],
            dtype=np.uint8,
        )
        reduced = reduce_stack(stack)
        expected = np.array(
            [
                [VARIABLE, 0, 0],
                [VARIABLE, 1, 1],
                [ADDITION, 0, 1],
            ],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(reduced, expected)

    def test_keeps_terminal_params_unchanged(self):
        stack = np.array(
            [
                [CONSTANT, 5, 5],
                [VARIABLE, 0, 0],
                [INTEGER, 2, 2],
                [MULTIPLICATION, 0, 1],
            ],
            dtype=np.uint8,
        )
        reduced = reduce_stack(stack)
        # INTEGER row (index 2) is unused
        expected = np.array(
            [
                [CONSTANT, 5, 5],
                [VARIABLE, 0, 0],
                [MULTIPLICATION, 0, 1],
            ],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(reduced, expected)

    def test_preserves_dtype(self):
        stack = np.array([[VARIABLE, 0, 0]], dtype=np.uint8)
        reduced = reduce_stack(stack)
        assert reduced.dtype == np.uint8
