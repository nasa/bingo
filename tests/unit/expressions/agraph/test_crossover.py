"""Tests for AGraphCrossover."""

import numpy as np
import pytest

from bingo.expressions.agraph.crossover import AGraphCrossover
from bingo.expressions.agraph.expression import AGraphExpression
from bingo.expressions.agraph.operators import (
    VARIABLE,
    CONSTANT,
    INTEGER,
    ADDITION,
    MULTIPLICATION,
    SIN,
)
from bingo.expressions.agraph.evolvable import EvolvableExpression


def _make_evolvable(command_rows, constants=(), integers=()):
    """Helper to build an EvolvableExpression from raw rows."""
    expr = AGraphExpression()
    expr.raw_command_array = np.array(command_rows, dtype=np.uint8)
    expr.raw_integers = integers
    expr.raw_constants = constants
    return EvolvableExpression(expr)


@pytest.fixture
def parent_a():
    """5-row expression:  C0 + X0"""
    return _make_evolvable(
        [
            [VARIABLE, 0, 0],
            [CONSTANT, 0, 0],
            [ADDITION, 0, 1],
            [SIN, 2, 2],
            [MULTIPLICATION, 2, 3],
        ],
        constants=(3.14,),
    )


@pytest.fixture
def parent_b():
    """4-row expression:  X1 * C0"""
    return _make_evolvable(
        [
            [VARIABLE, 1, 1],
            [CONSTANT, 0, 0],
            [MULTIPLICATION, 0, 1],
            [ADDITION, 0, 2],
        ],
        constants=(2.71,),
    )


@pytest.fixture
def crossover():
    return AGraphCrossover(min_size=1, max_size=10, random_state=0)


class TestInit:
    def test_min_size_too_small(self):
        with pytest.raises(ValueError, match="min_size"):
            AGraphCrossover(0, 10)

    def test_max_less_than_min(self):
        with pytest.raises(ValueError, match="max_size"):
            AGraphCrossover(5, 3)

    def test_equal_min_max(self):
        xover = AGraphCrossover(5, 5)
        assert xover._min_size == 5 and xover._max_size == 5


class TestCrossoverBasics:
    def test_returns_two_children(self, crossover, parent_a, parent_b):
        c1, c2 = crossover(parent_a, parent_b)
        assert isinstance(c1, EvolvableExpression)
        assert isinstance(c2, EvolvableExpression)

    def test_children_are_different_objects(self, crossover, parent_a, parent_b):
        c1, c2 = crossover(parent_a, parent_b)
        assert c1 is not parent_a
        assert c2 is not parent_b

    def test_genetic_age_is_max(self, crossover, parent_a, parent_b):
        parent_a.genetic_age = 3
        parent_b.genetic_age = 7
        c1, c2 = crossover(parent_a, parent_b)
        assert c1.genetic_age == 7
        assert c2.genetic_age == 7

    def test_fit_set_cleared(self, crossover, parent_a, parent_b):
        parent_a.fitness = 1.0
        parent_b.fitness = 2.0
        c1, c2 = crossover(parent_a, parent_b)
        assert not c1.fit_set
        assert not c2.fit_set

    def test_crossover_types_set(self, crossover, parent_a, parent_b):
        crossover(parent_a, parent_b)
        assert crossover.last_crossover_types == ("default", "default")


class TestSizeConstraints:
    def test_children_within_bounds(self):
        """Children must always stay within [min_size, max_size]."""
        xover = AGraphCrossover(min_size=3, max_size=7, random_state=0)
        for _ in range(100):
            p1 = _make_evolvable(
                [
                    [VARIABLE, 0, 0],
                    [CONSTANT, 0, 0],
                    [ADDITION, 0, 1],
                    [SIN, 2, 2],
                    [MULTIPLICATION, 2, 3],
                ],
                constants=(1.0,),
            )
            p2 = _make_evolvable(
                [
                    [VARIABLE, 1, 1],
                    [CONSTANT, 0, 0],
                    [MULTIPLICATION, 0, 1],
                    [ADDITION, 0, 2],
                ],
                constants=(2.0,),
            )
            c1, c2 = xover(p1, p2)
            assert (
                xover._min_size
                <= c1.expression.raw_command_array.shape[0]
                <= xover._max_size
            ), f"child 1 size {c1.expression.raw_command_array.shape[0]} out of bounds"
            assert (
                xover._min_size
                <= c2.expression.raw_command_array.shape[0]
                <= xover._max_size
            ), f"child 2 size {c2.expression.raw_command_array.shape[0]} out of bounds"

    def test_equal_min_max_fixed_size(self):
        """With min_size == max_size both children must have exactly that size."""
        size = 4
        xover = AGraphCrossover(min_size=size, max_size=size, random_state=42)
        for _ in range(50):
            p1 = _make_evolvable(
                [
                    [VARIABLE, 0, 0],
                    [CONSTANT, 0, 0],
                    [ADDITION, 0, 1],
                    [SIN, 2, 2],
                ],
                constants=(1.0,),
            )
            p2 = _make_evolvable(
                [
                    [VARIABLE, 1, 1],
                    [VARIABLE, 0, 0],
                    [MULTIPLICATION, 0, 1],
                    [ADDITION, 0, 2],
                ]
            )
            c1, c2 = xover(p1, p2)
            assert c1.expression.raw_command_array.shape[0] == size
            assert c2.expression.raw_command_array.shape[0] == size


class TestVariableSize:
    def test_different_size_parents(self, crossover, parent_a, parent_b):
        """parent_a has 5 rows, parent_b has 4 rows."""
        c1, c2 = crossover(parent_a, parent_b)
        # Children should have valid command arrays
        assert c1.expression.raw_command_array.ndim == 2
        assert c2.expression.raw_command_array.ndim == 2
        assert c1.expression.raw_command_array.shape[1] == 3
        assert c2.expression.raw_command_array.shape[1] == 3

    def test_same_size_parents(self, crossover):
        p1 = _make_evolvable(
            [
                [VARIABLE, 0, 0],
                [CONSTANT, 0, 0],
                [ADDITION, 0, 1],
            ],
            constants=(1.0,),
        )
        p2 = _make_evolvable(
            [
                [VARIABLE, 1, 1],
                [CONSTANT, 0, 0],
                [MULTIPLICATION, 0, 1],
            ],
            constants=(2.0,),
        )
        c1, c2 = crossover(p1, p2)
        # Both parents size 3 → children also valid
        assert c1.expression.raw_command_array.shape[0] >= 1
        assert c2.expression.raw_command_array.shape[0] >= 1


class TestPreserveConstants:
    """All tests go through the public __call__ interface.

    Using min_size == max_size == parent_size forces cp1 == cp2 for every
    crossover, which makes the constant-merge behaviour deterministic across
    all possible crossover points.
    """

    def test_child_constants_come_from_parents(self):
        """Every constant in a child must have come from one of the parents."""
        xover = AGraphCrossover(min_size=3, max_size=3, random_state=0)
        p1 = _make_evolvable(
            [[CONSTANT, 0, 0], [VARIABLE, 0, 0], [ADDITION, 0, 1]],
            constants=(42.0,),
        )
        p2 = _make_evolvable(
            [[CONSTANT, 0, 0], [VARIABLE, 1, 1], [MULTIPLICATION, 0, 1]],
            constants=(99.0,),
        )
        parent_pool = {42.0, 99.0}
        for _ in range(20):
            c1, c2 = xover(p1, p2)
            assert all(v in parent_pool for v in c1.expression.constants)
            assert all(v in parent_pool for v in c2.expression.constants)

    def test_only_head_parent_has_constants(self):
        """When the tail parent has no constants, children only carry head constants."""
        # p1 has a constant; p2 has none.
        # With fixed size 3 (cp1==cp2): c1's head comes from p1, tail from p2.
        # c2's head comes from p2, tail from p1.
        # In all cases, any constant present must be 42.0.
        xover = AGraphCrossover(min_size=3, max_size=3, random_state=0)
        p1 = _make_evolvable(
            [[CONSTANT, 0, 0], [VARIABLE, 0, 0], [ADDITION, 0, 1]],
            constants=(42.0,),
        )
        p2 = _make_evolvable(
            [[VARIABLE, 0, 0], [VARIABLE, 1, 1], [MULTIPLICATION, 0, 1]],
        )
        for _ in range(20):
            c1, c2 = xover(p1, p2)
            assert all(v == 42.0 for v in c1.expression.constants)
            assert all(v == 42.0 for v in c2.expression.constants)

    def test_head_and_tail_constants_independent(self):
        """With equal-size parents, c1 gets p1's constants and c2 gets p2's.

        With min_size == max_size == n, cp2 == cp1 for every draw.
        c1 = p1[:k] + p2[k:]:  the CONSTANT rows in p2's tail (rows >= k)
        index into p2.constants.
        c2 = p2[:k] + p1[k:]:  symmetric.
        For the stack layout below, the single CONSTANT row appears at row 0
        so it always ends up in the head (row 0 is always in the head for k>=1).
        Therefore c1.constants == (42.0,) and c2.constants == (99.0,) for all k.
        """
        xover = AGraphCrossover(min_size=3, max_size=3, random_state=0)
        p1 = _make_evolvable(
            [[CONSTANT, 0, 0], [VARIABLE, 0, 0], [ADDITION, 0, 1]],
            constants=(42.0,),
        )
        p2 = _make_evolvable(
            [[CONSTANT, 0, 0], [VARIABLE, 1, 1], [MULTIPLICATION, 0, 1]],
            constants=(99.0,),
        )
        for _ in range(20):
            c1, c2 = xover(p1, p2)
            assert c1.expression.constants == (42.0,)
            assert c2.expression.constants == (99.0,)


class TestIntegerMerging:
    def test_integers_preserved(self, crossover):
        p1 = _make_evolvable(
            [
                [INTEGER, 0, 0],
                [VARIABLE, 0, 0],
                [ADDITION, 0, 1],
            ],
            integers=(5,),
        )
        p2 = _make_evolvable(
            [
                [INTEGER, 0, 0],
                [VARIABLE, 1, 1],
                [MULTIPLICATION, 0, 1],
            ],
            integers=(10,),
        )

        c1, c2 = crossover(p1, p2)
        # Children should have integer values from parents
        all_ints = set(c1.expression.integers + c2.expression.integers)
        assert all_ints.intersection({5, 10})


class TestEdgeCases:
    def test_size_one_parent(self, crossover):
        """A parent with a single row — crossover should still work."""
        p1 = _make_evolvable([[VARIABLE, 0, 0]])
        p2 = _make_evolvable(
            [
                [VARIABLE, 1, 1],
                [CONSTANT, 0, 0],
                [ADDITION, 0, 1],
            ],
            constants=(1.0,),
        )
        c1, c2 = crossover(p1, p2)
        assert c1.expression.raw_command_array.shape[0] >= 1
        assert c2.expression.raw_command_array.shape[0] >= 1

    def test_repeated_crossover_no_crash(self, crossover, parent_a, parent_b):
        for _ in range(50):
            crossover(parent_a, parent_b)
