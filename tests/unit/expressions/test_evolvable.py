"""Tests for EvolvableExpression."""

import copy

import numpy as np
import pytest

from bingo.expressions.agraph.pyagraph.expression import AGraphExpression
from bingo.expressions.agraph.pyagraph.operators import (
    VARIABLE,
    CONSTANT,
    ADDITION,
    SIN,
)
from bingo.expressions.agraph.evolvable import EvolvableExpression
from bingo.chromosomes.chromosome import Chromosome


def _make_expr(command_rows, constants=(), integers=(), simplification="reduce"):
    expr = AGraphExpression(simplification=simplification)
    expr.raw_command_array = np.array(command_rows, dtype=np.uint8)
    expr.raw_integers = integers
    expr.raw_constants = constants
    return expr


def _make_evolvable(command_rows, constants=(), integers=()):
    return EvolvableExpression(_make_expr(command_rows, constants, integers))


class TestIsChromosome:
    def test_inherits_chromosome(self):
        indv = _make_evolvable([[VARIABLE, 0, 0]])
        assert isinstance(indv, Chromosome)


class TestChromosomeAttributes:
    def test_default_genetic_age(self):
        indv = _make_evolvable([[VARIABLE, 0, 0]])
        assert indv.genetic_age == 0

    def test_set_genetic_age(self):
        indv = _make_evolvable([[VARIABLE, 0, 0]])
        indv.genetic_age = 5
        assert indv.genetic_age == 5

    def test_default_fitness_not_set(self):
        indv = _make_evolvable([[VARIABLE, 0, 0]])
        assert not indv.fit_set

    def test_set_fitness(self):
        indv = _make_evolvable([[VARIABLE, 0, 0]])
        indv.fitness = 42.0
        assert indv.fitness == 42.0
        assert indv.fit_set

    def test_fit_set_toggle(self):
        indv = _make_evolvable([[VARIABLE, 0, 0]])
        indv.fitness = 1.0
        assert indv.fit_set
        indv.fit_set = False
        assert not indv.fit_set


class TestStr:
    def test_str_delegates_to_expression(self):
        indv = _make_evolvable(
            [
                [VARIABLE, 0, 0],
                [CONSTANT, 0, 0],
                [ADDITION, 0, 1],
            ],
            constants=(1.0,),
        )
        s = str(indv)
        assert "X0" in s or "x" in s.lower()


class TestDistance:
    def test_identical_distance_zero(self):
        rows = [[VARIABLE, 0, 0], [CONSTANT, 0, 0], [ADDITION, 0, 1]]
        a = _make_evolvable(rows, constants=(1.0,))
        b = _make_evolvable(rows, constants=(1.0,))
        assert a.distance(b) == 0

    def test_different_arrays_positive_distance(self):
        a = _make_evolvable([[VARIABLE, 0, 0]])
        b = _make_evolvable([[VARIABLE, 1, 1]])
        assert a.distance(b) > 0


class TestNoLocalOptimizationAdapter:
    def test_does_not_implement_local_optimization_methods(self):
        assert "needs_local_optimization" not in EvolvableExpression.__dict__
        assert "get_number_local_optimization_params" not in EvolvableExpression.__dict__
        assert "set_local_optimization_params" not in EvolvableExpression.__dict__


class TestDelegation:
    def test_command_array(self):
        """command_array on evolvable returns the simplified (evaluation-ready) stack."""
        rows = [[VARIABLE, 0, 0], [SIN, 0, 0]]
        indv = _make_evolvable(rows)
        np.testing.assert_array_equal(indv.command_array, indv.expression.command_array)

    def test_mutable_raw_command_array(self):
        """Genetic ops access the raw writable stack via expression directly."""
        rows = [[VARIABLE, 0, 0], [SIN, 0, 0]]
        indv = _make_evolvable(rows)
        m = indv.expression.mutable_raw_command_array
        assert m.flags.writeable

    def test_get_utilized_commands(self):
        rows = [[VARIABLE, 0, 0], [SIN, 0, 0]]
        indv = _make_evolvable(rows)
        util = indv.get_utilized_commands()
        assert isinstance(util, bytearray)
        assert len(util) == 2

    def test_tree_complexity_delegates(self):
        rows = [[VARIABLE, 0, 0], [CONSTANT, 0, 0], [ADDITION, 0, 1]]
        indv = _make_evolvable(rows, constants=(1.0,))
        assert indv.tree_complexity == indv.expression.tree_complexity


class TestCopy:
    def test_copy_returns_new_object(self):
        indv = _make_evolvable([[VARIABLE, 0, 0]])
        c = indv.copy()
        assert c is not indv
        assert c.expression is not indv.expression

    def test_copy_preserves_fitness(self):
        indv = _make_evolvable([[VARIABLE, 0, 0]])
        indv.fitness = 3.14
        indv.genetic_age = 7
        c = indv.copy()
        assert c.fitness == 3.14
        assert c.genetic_age == 7

    def test_deepcopy(self):
        indv = _make_evolvable([[VARIABLE, 0, 0]])
        indv.fitness = 2.0
        c = copy.deepcopy(indv)
        assert c is not indv
        assert c.fitness == 2.0

    def test_copy_independence(self):
        indv = _make_evolvable(
            [
                [VARIABLE, 0, 0],
                [CONSTANT, 0, 0],
                [ADDITION, 0, 1],
            ],
            constants=(1.0,),
        )
        c = indv.copy()
        c.expression.mutable_raw_command_array[0, 1] = 99
        # Original should be unchanged
        assert indv.command_array[0, 1] != 99
