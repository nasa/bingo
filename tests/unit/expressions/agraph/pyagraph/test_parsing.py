"""Tests for bingo.expressions.agraph.parsing"""

import numpy as np
import pytest

from bingo.expressions.agraph.pyagraph.operators import (
    VARIABLE,
    CONSTANT,
    INTEGER,
    ADDITION,
    SUBTRACTION,
    MULTIPLICATION,
    DIVISION,
    SIN,
    POWER,
)
from bingo.expressions.agraph.pyagraph.parsing import (
    eq_string_to_command_array_and_constants,
)


class TestBasicParsing:
    def test_single_variable_underscore(self):
        cmd, consts, ints = eq_string_to_command_array_and_constants("X_0")
        assert cmd[0, 0] == VARIABLE
        assert cmd[0, 1] == 0

    def test_single_variable_no_underscore(self):
        cmd, consts, ints = eq_string_to_command_array_and_constants("X0")
        assert cmd[0, 0] == VARIABLE
        assert cmd[0, 1] == 0

    def test_single_constant(self):
        cmd, consts, ints = eq_string_to_command_array_and_constants("3.14")
        assert cmd[0, 0] == CONSTANT
        assert len(consts) == 1
        assert consts[0] == pytest.approx(3.14)

    def test_integer_literal(self):
        cmd, consts, ints = eq_string_to_command_array_and_constants("7")
        assert cmd[0, 0] == INTEGER
        assert len(ints) == 1
        assert ints[0] == 7

    def test_addition(self):
        cmd, consts, ints = eq_string_to_command_array_and_constants("X0 + X1")
        assert cmd[-1, 0] == ADDITION

    def test_subtraction(self):
        cmd, consts, ints = eq_string_to_command_array_and_constants("X0 - X1")
        assert cmd[-1, 0] == SUBTRACTION

    def test_multiplication(self):
        cmd, consts, ints = eq_string_to_command_array_and_constants("X0 * X1")
        assert cmd[-1, 0] == MULTIPLICATION

    def test_division(self):
        cmd, consts, ints = eq_string_to_command_array_and_constants("X0 / X1")
        assert cmd[-1, 0] == DIVISION


class TestFunctionParsing:
    def test_sin(self):
        cmd, consts, ints = eq_string_to_command_array_and_constants("sin(X0)")
        assert cmd[-1, 0] == SIN

    def test_power(self):
        cmd, consts, ints = eq_string_to_command_array_and_constants("X0 ^ X1")
        assert cmd[-1, 0] == POWER

    def test_double_star_power(self):
        cmd, consts, ints = eq_string_to_command_array_and_constants("X0 ** X1")
        assert cmd[-1, 0] == POWER


class TestDtype:
    def test_command_array_is_uint8(self):
        cmd, _, _ = eq_string_to_command_array_and_constants("X0 + X1")
        assert cmd.dtype == np.uint8


class TestConstantsAndIntegers:
    def test_mixed_expression(self):
        cmd, consts, ints = eq_string_to_command_array_and_constants("X0 + 3.5 + 2")
        assert len(consts) == 1
        assert consts[0] == pytest.approx(3.5)
        assert len(ints) == 1
        assert ints[0] == 2

    def test_constant_named_c0(self):
        cmd, consts, ints = eq_string_to_command_array_and_constants("C_0")
        assert cmd[0, 0] == CONSTANT
        assert cmd[0, 1] == 0

    def test_constant_named_c0_no_underscore(self):
        cmd, consts, ints = eq_string_to_command_array_and_constants("C0")
        assert cmd[0, 0] == CONSTANT
        assert cmd[0, 1] == 0


class TestComplexExpressions:
    def test_nested_functions(self):
        cmd, consts, ints = eq_string_to_command_array_and_constants(
            "sin(X0) + X1 * 2.0"
        )
        assert cmd.dtype == np.uint8
        assert len(consts) == 1
        assert consts[0] == pytest.approx(2.0)

    def test_parenthesized_expression(self):
        cmd, consts, ints = eq_string_to_command_array_and_constants("(X0 + X1) * X0")
        assert cmd[-1, 0] == MULTIPLICATION


class TestNamedParsing:
    def test_named_variables_parse_to_variable_indices(self):
        cmd, consts, ints = eq_string_to_command_array_and_constants(
            "a + sin(x)", variables=("a", "x")
        )
        assert cmd[0, 0] == VARIABLE
        assert cmd[0, 1] == 0
        assert any(row[0] == VARIABLE and row[1] == 1 for row in cmd)

    def test_named_parameter_reuse_shares_constant_index(self):
        cmd, consts, ints = eq_string_to_command_array_and_constants(
            "a + C*sin(x) + C",
            variables=("a", "x"),
            parameters=("C",),
        )
        constant_rows = cmd[cmd[:, 0] == CONSTANT]
        assert len(consts) == 1
        assert consts == (1.0,)
        assert {int(row[1]) for row in constant_rows} == {0}

    def test_named_parameters_follow_first_appearance_order(self):
        _, consts, _ = eq_string_to_command_array_and_constants(
            "a + B*sin(x) + C",
            variables=("a", "x"),
            parameters=("B", "C"),
        )
        assert consts == (1.0, 1.0)

    def test_named_mode_allows_legacy_shaped_custom_names(self):
        cmd, _, _ = eq_string_to_command_array_and_constants(
            "X0 + sin(x)", variables=("X0", "x")
        )
        assert cmd[0, 0] == VARIABLE
        assert cmd[0, 1] == 0


class TestRoundTrip:
    def test_console_round_trip(self):
        """Parse -> console format -> parse should produce equivalent arrays."""
        from bingo.expressions.agraph.pyagraph.formatting import get_formatted_string

        original = "X0 + X1"
        cmd1, c1, i1 = eq_string_to_command_array_and_constants(original)
        console_str = get_formatted_string("console", cmd1, c1, i1)
        cmd2, c2, i2 = eq_string_to_command_array_and_constants(console_str)
        np.testing.assert_array_equal(cmd1, cmd2)


class TestErrors:
    def test_mismatched_parens(self):
        with pytest.raises(RuntimeError, match="parenthesis"):
            eq_string_to_command_array_and_constants("(X0 + X1")

    def test_inf_complex(self):
        with pytest.raises(RuntimeError, match="inf/complex"):
            eq_string_to_command_array_and_constants("oo")

    def test_named_mode_rejects_unknown_variable(self):
        with pytest.raises(RuntimeError, match="Unknown token y"):
            eq_string_to_command_array_and_constants(
                "a + y", variables=("a", "x")
            )

    def test_named_mode_rejects_unknown_parameter(self):
        with pytest.raises(RuntimeError, match="Unknown token D"):
            eq_string_to_command_array_and_constants(
                "a + C + D",
                variables=("a",),
                parameters=("C",),
            )

    def test_named_mode_rejects_legacy_tokens_outside_symbol_table(self):
        with pytest.raises(RuntimeError, match="Unknown token X0"):
            eq_string_to_command_array_and_constants(
                "X0 + sin(x)", variables=("a", "x")
            )

    def test_parameters_require_unique_non_overlapping_names(self):
        with pytest.raises(ValueError, match="must not overlap"):
            eq_string_to_command_array_and_constants(
                "a + C", variables=("a",), parameters=("a",)
            )
