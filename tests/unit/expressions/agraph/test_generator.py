"""Tests for AGraphGenerator."""

import numpy as np
import pytest

from bingo.expressions.agraph.generator import AGraphGenerator
from bingo.expressions.agraph.component_generator import ComponentGenerator
from bingo.expressions.agraph.operators import ADDITION, MULTIPLICATION, SIN, IS_TERMINAL_MAP
from bingo.expressions.agraph.evolvable import EvolvableExpression


@pytest.fixture
def component_gen():
    gen = ComponentGenerator(input_x_dimension=3)
    gen.add_operator(ADDITION)
    gen.add_operator(MULTIPLICATION)
    gen.add_operator(SIN)
    return gen


@pytest.fixture
def generator(component_gen):
    return AGraphGenerator(min_size=5, max_size=10, component_generator=component_gen)


class TestInit:
    def test_min_size_too_small(self, component_gen):
        with pytest.raises(ValueError, match="min_size"):
            AGraphGenerator(0, 10, component_gen)

    def test_max_less_than_min(self, component_gen):
        with pytest.raises(ValueError, match="max_size"):
            AGraphGenerator(5, 3, component_gen)

    def test_equal_min_max(self, component_gen):
        gen = AGraphGenerator(5, 5, component_gen)
        assert gen.min_size == 5 and gen.max_size == 5


class TestGeneration:
    def test_returns_evolvable_expression(self, generator):
        np.random.seed(42)
        indv = generator()
        assert isinstance(indv, EvolvableExpression)

    def test_size_in_range(self, generator):
        np.random.seed(0)
        for _ in range(30):
            indv = generator()
            n = indv.expression.raw_command_array.shape[0]
            assert generator.min_size <= n <= generator.max_size

    def test_fixed_size(self, component_gen):
        gen = AGraphGenerator(7, 7, component_gen)
        np.random.seed(0)
        for _ in range(10):
            indv = gen()
            assert indv.expression.raw_command_array.shape[0] == 7

    def test_command_array_has_correct_shape(self, generator):
        np.random.seed(0)
        indv = generator()
        assert indv.expression.raw_command_array.ndim == 2
        assert indv.expression.raw_command_array.shape[1] == 3

    def test_command_array_dtype(self, generator):
        np.random.seed(0)
        indv = generator()
        assert indv.expression.raw_command_array.dtype == np.uint8

    def test_row_zero_is_terminal(self, generator):
        np.random.seed(0)
        for _ in range(10):
            indv = generator()
            assert IS_TERMINAL_MAP[int(indv.expression.raw_command_array[0, 0])]

    def test_fresh_individual_has_zero_genetic_age(self, generator):
        np.random.seed(0)
        indv = generator()
        assert indv.genetic_age == 0

    def test_fresh_individual_fitness_not_set(self, generator):
        np.random.seed(0)
        indv = generator()
        assert not indv.fit_set

    def test_constant_values_populated_in_raw_constants(self):
        """Every CONSTANT node in the generated stack should have its
        value recorded in raw_constants at the correct index."""
        from bingo.expressions.agraph.operators import CONSTANT
        gen = ComponentGenerator(
            input_x_dimension=1,
            constant_probability=1.0,  # force all terminals to be CONSTANT
            constant_distribution="normal",
            constant_scale=2.0,
            random_state=0,
        )
        gen.add_operator(ADDITION)
        agraph_gen = AGraphGenerator(3, 3, gen, random_state=0)
        for _ in range(20):
            indv = agraph_gen()
            ca = indv.expression.raw_command_array
            rc = indv.expression.raw_constants
            for row in ca:
                if int(row[0]) == CONSTANT:
                    idx = int(row[1])
                    assert idx < len(rc), "CONSTANT index out of range in raw_constants"
                    assert isinstance(rc[idx], float)


class TestSimplificationParam:
    def test_default_simplification_is_cas(self, component_gen):
        gen = AGraphGenerator(5, 5, component_gen)
        np.random.seed(0)
        indv = gen()
        assert indv.expression._simplification == "cas"

    def test_reduce_simplification(self, component_gen):
        gen = AGraphGenerator(5, 5, component_gen, simplification="reduce")
        np.random.seed(0)
        indv = gen()
        assert indv.expression._simplification == "reduce"

    def test_cas_simplification_explicit(self, component_gen):
        gen = AGraphGenerator(5, 5, component_gen, simplification="cas")
        np.random.seed(0)
        indv = gen()
        assert indv.expression._simplification == "cas"
