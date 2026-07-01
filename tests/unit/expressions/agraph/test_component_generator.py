"""Tests for ComponentGenerator."""

import numpy as np
import pytest

from bingo.expressions.agraph.component_generator import ComponentGenerator
from bingo.expressions.agraph.pyagraph.operators import (
    VARIABLE,
    CONSTANT,
    ADDITION,
    SUBTRACTION,
    MULTIPLICATION,
    SIN,
    SQRT,
    TERMINAL_IDS,
    ARITY_2_IDS,
)


@pytest.fixture
def cgen():
    """ComponentGenerator with 3 input variables and basic operators."""
    gen = ComponentGenerator(input_x_dimension=3, random_state=0)
    gen.add_operator(ADDITION)
    gen.add_operator(MULTIPLICATION)
    gen.add_operator(SIN)
    return gen


class TestComponentGeneratorInit:
    def test_basic_creation(self):
        gen = ComponentGenerator(input_x_dimension=2)
        assert gen.input_x_dimension == 2

    def test_negative_dimension_raises(self):
        with pytest.raises(ValueError, match="input_x_dimension"):
            ComponentGenerator(input_x_dimension=-1)

    def test_bad_initial_load_raises(self):
        with pytest.raises(ValueError, match="num_initial_load_statements"):
            ComponentGenerator(input_x_dimension=2, num_initial_load_statements=0)

    def test_bad_terminal_prob_raises(self):
        with pytest.raises(ValueError, match="terminal_probability"):
            ComponentGenerator(input_x_dimension=2, terminal_probability=1.5)

    def test_bad_constant_distribution_raises(self):
        with pytest.raises(ValueError, match="constant_distribution"):
            ComponentGenerator(input_x_dimension=2, constant_distribution="cauchy")

    def test_bad_constant_scale_raises(self):
        with pytest.raises(ValueError, match="constant_scale"):
            ComponentGenerator(input_x_dimension=2, constant_scale=-1.0)

    def test_zero_constant_scale_raises(self):
        with pytest.raises(ValueError, match="constant_scale"):
            ComponentGenerator(input_x_dimension=2, constant_scale=0.0)

    def test_explicit_constant_probability(self):
        gen = ComponentGenerator(
            input_x_dimension=3, constant_probability=0.5, random_state=0
        )
        gen.add_operator(ADDITION)
        # Just check it doesn't crash
        cmd = gen.random_terminal_command()
        assert cmd.shape == (3,)

    def test_random_state_seed(self):
        gen1 = ComponentGenerator(input_x_dimension=2, random_state=42)
        gen2 = ComponentGenerator(input_x_dimension=2, random_state=42)
        gen1.add_operator(ADDITION)
        gen2.add_operator(ADDITION)
        cmd1 = gen1.random_command(5)
        cmd2 = gen2.random_command(5)
        np.testing.assert_array_equal(cmd1, cmd2)


class TestAddOperator:
    def test_add_by_id(self):
        gen = ComponentGenerator(input_x_dimension=2)
        gen.add_operator(ADDITION)
        assert gen.get_number_of_operators() == 1

    def test_add_by_name(self):
        gen = ComponentGenerator(input_x_dimension=2)
        gen.add_operator("+")
        assert gen.get_number_of_operators() == 1

    def test_add_multiple(self):
        gen = ComponentGenerator(input_x_dimension=2)
        gen.add_operator(ADDITION)
        gen.add_operator(MULTIPLICATION)
        gen.add_operator(SIN)
        assert gen.get_number_of_operators() == 3

    def test_unknown_name_raises(self):
        gen = ComponentGenerator(input_x_dimension=2)
        with pytest.raises(ValueError, match="Unknown operator"):
            gen.add_operator("nonexistent_op")


class TestRandomCommand:
    def test_early_rows_are_terminals(self, cgen):
        for _ in range(20):
            cmd = cgen.random_command(0)
            assert int(cmd[0]) in TERMINAL_IDS, "Row 0 should always be a terminal"

    def test_command_shape_and_dtype(self, cgen):
        cmd = cgen.random_command(5)
        assert cmd.shape == (3,)
        assert cmd.dtype == np.uint8

    def test_later_rows_can_be_operators(self, cgen):
        found_operator = False
        for _ in range(100):
            cmd = cgen.random_command(5)
            if int(cmd[0]) not in TERMINAL_IDS:
                found_operator = True
                break
        assert found_operator

    def test_operator_params_less_than_location(self, cgen):
        for _ in range(50):
            loc = 5
            cmd = cgen.random_command(loc)
            if int(cmd[0]) not in TERMINAL_IDS:
                assert int(cmd[1]) < loc
                assert int(cmd[2]) < loc


class TestRandomConstantValue:
    def test_returns_float(self):
        gen = ComponentGenerator(input_x_dimension=2, random_state=0)
        val = gen.random_constant_value()
        assert isinstance(val, float)

    def test_normal_distribution_scale(self):
        """Values drawn with a large scale should have larger spread."""
        gen_small = ComponentGenerator(
            input_x_dimension=1,
            constant_distribution="normal",
            constant_scale=0.01,
            random_state=42,
        )
        gen_large = ComponentGenerator(
            input_x_dimension=1,
            constant_distribution="normal",
            constant_scale=100.0,
            random_state=42,
        )
        vals_small = [abs(gen_small.random_constant_value()) for _ in range(50)]
        vals_large = [abs(gen_large.random_constant_value()) for _ in range(50)]
        assert max(vals_small) < max(vals_large)

    def test_uniform_distribution_bounded(self):
        """Uniform values must lie within (-scale, +scale)."""
        scale = 3.0
        gen = ComponentGenerator(
            input_x_dimension=1,
            constant_distribution="uniform",
            constant_scale=scale,
            random_state=7,
        )
        for _ in range(100):
            val = gen.random_constant_value()
            assert -scale <= val <= scale

    def test_normal_and_uniform_differ(self):
        """The two distributions should produce different sequences."""
        gen_n = ComponentGenerator(
            input_x_dimension=1,
            constant_distribution="normal",
            constant_scale=1.0,
            random_state=0,
        )
        gen_u = ComponentGenerator(
            input_x_dimension=1,
            constant_distribution="uniform",
            constant_scale=1.0,
            random_state=0,
        )
        vals_n = [gen_n.random_constant_value() for _ in range(20)]
        vals_u = [gen_u.random_constant_value() for _ in range(20)]
        assert vals_n != vals_u


class TestRandomTerminalCommand:
    def test_produces_variable_or_constant(self, cgen):
        for _ in range(50):
            cmd = cgen.random_terminal_command()
            assert int(cmd[0]) in (VARIABLE, CONSTANT)

    def test_variable_param_in_range(self, cgen):
        for _ in range(50):
            cmd = cgen.random_terminal_command()
            if int(cmd[0]) == VARIABLE:
                assert 0 <= int(cmd[1]) < cgen.input_x_dimension

    def test_constant_param_is_placeholder(self, cgen):
        for _ in range(50):
            cmd = cgen.random_terminal_command()
            if int(cmd[0]) == CONSTANT:
                assert int(cmd[1]) == 0


class TestRandomOperatorCommand:
    def test_operator_is_not_terminal(self, cgen):
        for _ in range(20):
            cmd = cgen.random_operator_command(5)
            assert int(cmd[0]) not in TERMINAL_IDS


class TestIntrospection:
    def test_number_of_terminals(self, cgen):
        assert cgen.get_number_of_terminals() == 2  # VARIABLE, CONSTANT

    def test_number_of_operators(self, cgen):
        assert cgen.get_number_of_operators() == 3


class TestCustomLoadStatements:
    def test_two_initial_load_statements(self):
        gen = ComponentGenerator(
            input_x_dimension=2, num_initial_load_statements=2, random_state=0
        )
        gen.add_operator(ADDITION)
        for _ in range(20):
            cmd0 = gen.random_command(0)
            cmd1 = gen.random_command(1)
            assert int(cmd0[0]) in TERMINAL_IDS
            assert int(cmd1[0]) in TERMINAL_IDS


class TestNoOperatorsError:
    """Descriptive error when no operators have been added (issue #90)."""

    def test_random_operator_raises_without_operators(self):
        gen = ComponentGenerator(input_x_dimension=2, random_state=0)
        with pytest.raises(ValueError, match="No operators have been added"):
            gen.random_operator()

    def test_random_operator_command_raises_without_operators(self):
        gen = ComponentGenerator(input_x_dimension=2, random_state=0)
        with pytest.raises(ValueError, match="No operators have been added"):
            gen.random_operator_command(5)

    def test_generator_call_raises_without_operators(self):
        """Replicates the bug-report usage (with min_size bumped to ensure an operator is requested)."""
        from bingo.expressions.agraph.generator import AGraphGenerator

        rng = np.random.default_rng(seed=42)
        component_generator = ComponentGenerator(
            input_x_dimension=2, random_state=rng
        )
        generator = AGraphGenerator(
            min_size=3,
            max_size=11,
            component_generator=component_generator,
            random_state=rng,
        )
        with pytest.raises(ValueError, match="No operators have been added"):
            generator()
