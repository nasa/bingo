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
    reduce,
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


class TestReduce:
    def test_no_reduction_needed(self):
        # All rows utilized; constants and integers pass through unchanged.
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [CONSTANT, 0, 0],
                [ADDITION, 0, 1],
            ],
            dtype=np.uint8,
        )
        reduced, consts, ints = reduce(stack, (3.14,), ())
        np.testing.assert_array_equal(
            reduced,
            np.array(
                [[VARIABLE, 0, 0], [CONSTANT, 0, 0], [ADDITION, 0, 1]],
                dtype=np.uint8,
            ),
        )
        assert consts == (3.14,)
        assert ints == ()

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
        reduced, consts, ints = reduce(stack, (), ())
        expected = np.array(
            [[VARIABLE, 0, 0], [SIN, 0, 0]],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(reduced, expected)
        assert consts == ()
        assert ints == ()

    def test_remaps_operator_references(self):
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
        reduced, consts, ints = reduce(stack, (), ())
        expected = np.array(
            [
                [VARIABLE, 0, 0],
                [VARIABLE, 1, 1],
                [ADDITION, 0, 1],
            ],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(reduced, expected)
        assert consts == ()
        assert ints == ()

    def test_renumbers_constant_indices(self):
        # Row 0: C0, Row 1: C1 (unused), Row 2: C2, Row 3: C0 * C2
        stack = np.array(
            [
                [CONSTANT, 0, 0],
                [CONSTANT, 1, 1],  # unused
                [CONSTANT, 2, 2],
                [MULTIPLICATION, 0, 2],
            ],
            dtype=np.uint8,
        )
        raw_consts = (1.0, 99.0, 2.0)
        reduced, consts, ints = reduce(stack, raw_consts, ())
        # C1 is dropped; C0 -> index 0, C2 -> index 1
        expected = np.array(
            [
                [CONSTANT, 0, 0],
                [CONSTANT, 1, 1],
                [MULTIPLICATION, 0, 1],
            ],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(reduced, expected)
        assert consts == (1.0, 2.0)
        assert ints == ()

    def test_renumbers_integer_indices(self):
        # Row 0: I0, Row 1: I1 (unused), Row 2: I2, Row 3: I0 + I2
        stack = np.array(
            [
                [INTEGER, 0, 0],
                [INTEGER, 1, 1],  # unused
                [INTEGER, 2, 2],
                [ADDITION, 0, 2],
            ],
            dtype=np.uint8,
        )
        raw_ints = (10, 99, 20)
        reduced, consts, ints = reduce(stack, (), raw_ints)
        expected = np.array(
            [
                [INTEGER, 0, 0],
                [INTEGER, 1, 1],
                [ADDITION, 0, 1],
            ],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(reduced, expected)
        assert consts == ()
        assert ints == (10, 20)

    def test_preserves_dtype(self):
        stack = np.array([[VARIABLE, 0, 0]], dtype=np.uint8)
        reduced, _, _ = reduce(stack, (), ())
        assert reduced.dtype == np.uint8

    def test_empty_stack(self):
        stack = np.empty((0, 3), dtype=np.uint8)
        reduced, consts, ints = reduce(stack, (), ())
        assert reduced.shape == (0, 3)
        assert consts == ()
        assert ints == ()
