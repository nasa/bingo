"""Tests for bingo.expressions.agraph.formatting"""

import numpy as np
import pytest

from bingo.expressions.agraph.pyagraph.operators import (
    VARIABLE,
    CONSTANT,
    INTEGER,
    ADDITION,
    SUBTRACTION,
    MULTIPLICATION,
    SIN,
    SQUARE,
    CUBE,
    DIVISION,
    POWER,
    SAFE_POWER,
)
from bingo.expressions.agraph.pyagraph.formatting import get_formatted_string


@pytest.fixture
def x0_plus_c0_stack():
    """X0 + C0  where C0 = 3.0"""
    stack = np.array(
        [
            [VARIABLE, 0, 0],
            [CONSTANT, 0, 0],
            [ADDITION, 0, 1],
        ],
        dtype=np.uint8,
    )
    return stack, (3.0,), ()


@pytest.fixture
def sin_x0_stack():
    """sin(X0)"""
    stack = np.array(
        [
            [VARIABLE, 0, 0],
            [SIN, 0, 0],
        ],
        dtype=np.uint8,
    )
    return stack, (), ()


@pytest.fixture
def integer_stack():
    """X0 + 7  where 7 is in the integers tuple"""
    stack = np.array(
        [
            [VARIABLE, 0, 0],
            [INTEGER, 0, 0],
            [ADDITION, 0, 1],
        ],
        dtype=np.uint8,
    )
    return stack, (), (7,)


# ------------------------------------------------------------------ #
#  Console format                                                     #
# ------------------------------------------------------------------ #


class TestConsoleFormat:
    def test_x0_plus_c0(self, x0_plus_c0_stack):
        stack, consts, ints = x0_plus_c0_stack
        s = get_formatted_string("console", stack, consts, ints)
        assert "X0" in s
        assert "3.0" in s
        assert "+" in s

    def test_sin_x0(self, sin_x0_stack):
        stack, consts, ints = sin_x0_stack
        s = get_formatted_string("console", stack, consts, ints)
        assert s == "sin(X0)"

    def test_variable_name_no_underscore(self):
        stack = np.array([[VARIABLE, 1, 1]], dtype=np.uint8)
        s = get_formatted_string("console", stack, (), ())
        assert s == "X1"
        assert "_" not in s


# ------------------------------------------------------------------ #
#  Console — precedence-aware parenthesization                        #
# ------------------------------------------------------------------ #


class TestConsolePrecedence:
    """Verify that terminals never get parenthesised and that nested
    operations at the same precedence level omit unnecessary parens."""

    def test_subtraction_of_terminals(self):
        """X0 - X1  (not X0 - (X1))"""
        stack = np.array(
            [[VARIABLE, 0, 0], [VARIABLE, 1, 1], [SUBTRACTION, 0, 1]],
            dtype=np.uint8,
        )
        assert get_formatted_string("console", stack, (), ()) == "X0 - X1"

    def test_subtraction_of_addition_on_right(self):
        """X0 - (X1 + X2)  — right ADD child of SUB needs parens."""
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [VARIABLE, 1, 1],
                [VARIABLE, 2, 2],
                [ADDITION, 1, 2],
                [SUBTRACTION, 0, 3],
            ],
            dtype=np.uint8,
        )
        assert get_formatted_string("console", stack, (), ()) == "X0 - (X1 + X2)"

    def test_addition_of_subtraction_on_left(self):
        """(X0 - X1) + X2  — left SUB child of ADD: no parens needed."""
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [VARIABLE, 1, 1],
                [SUBTRACTION, 0, 1],
                [VARIABLE, 2, 2],
                [ADDITION, 2, 3],
            ],
            dtype=np.uint8,
        )
        assert get_formatted_string("console", stack, (), ()) == "X0 - X1 + X2"

    def test_multiplication_of_terminals(self):
        """X0*X1  (not (X0)*(X1))"""
        stack = np.array(
            [[VARIABLE, 0, 0], [VARIABLE, 1, 1], [MULTIPLICATION, 0, 1]],
            dtype=np.uint8,
        )
        assert get_formatted_string("console", stack, (), ()) == "X0*X1"

    def test_nested_multiplication(self):
        """X0*X1*X2  (not (X0*X1)*X2)"""
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [VARIABLE, 1, 1],
                [MULTIPLICATION, 0, 1],
                [VARIABLE, 2, 2],
                [MULTIPLICATION, 2, 3],
            ],
            dtype=np.uint8,
        )
        assert get_formatted_string("console", stack, (), ()) == "X0*X1*X2"

    def test_multiplication_wraps_lower_prec_child(self):
        """(X0 + X1)*X2"""
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [VARIABLE, 1, 1],
                [ADDITION, 0, 1],
                [VARIABLE, 2, 2],
                [MULTIPLICATION, 2, 3],
            ],
            dtype=np.uint8,
        )
        assert get_formatted_string("console", stack, (), ()) == "(X0 + X1)*X2"

    def test_division_of_terminals(self):
        """X0/X1  (not (X0)/(X1))"""
        stack = np.array(
            [[VARIABLE, 0, 0], [VARIABLE, 1, 1], [DIVISION, 0, 1]],
            dtype=np.uint8,
        )
        assert get_formatted_string("console", stack, (), ()) == "X0/X1"

    def test_division_right_child_mul(self):
        """X0/(X1*X2)  — right MUL child of DIV needs parens."""
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [VARIABLE, 1, 1],
                [VARIABLE, 2, 2],
                [MULTIPLICATION, 1, 2],
                [DIVISION, 0, 3],
            ],
            dtype=np.uint8,
        )
        assert get_formatted_string("console", stack, (), ()) == "X0/(X1*X2)"

    def test_mul_right_child_div(self):
        """X0*(X1/X2)  — right DIV child of MUL needs parens."""
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [VARIABLE, 1, 1],
                [VARIABLE, 2, 2],
                [DIVISION, 1, 2],
                [MULTIPLICATION, 0, 3],
            ],
            dtype=np.uint8,
        )
        assert get_formatted_string("console", stack, (), ()) == "X0*(X1/X2)"

    def test_power_of_terminals(self):
        """X0**X1  (not (X0)**(X1))"""
        stack = np.array(
            [[VARIABLE, 0, 0], [VARIABLE, 1, 1], [POWER, 0, 1]],
            dtype=np.uint8,
        )
        assert get_formatted_string("console", stack, (), ()) == "X0**X1"

    def test_power_wraps_lower_prec_base(self):
        """(X0 + X1)**X2"""
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [VARIABLE, 1, 1],
                [ADDITION, 0, 1],
                [VARIABLE, 2, 2],
                [POWER, 2, 3],
            ],
            dtype=np.uint8,
        )
        assert get_formatted_string("console", stack, (), ()) == "(X0 + X1)**X2"

    def test_power_wraps_lower_prec_exponent(self):
        """X0**(X1 + X2)"""
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [VARIABLE, 1, 1],
                [VARIABLE, 2, 2],
                [ADDITION, 1, 2],
                [POWER, 0, 3],
            ],
            dtype=np.uint8,
        )
        assert get_formatted_string("console", stack, (), ()) == "X0**(X1 + X2)"

    def test_power_right_associative(self):
        """X0**X1**X2  — right-assoc: no parens on right child."""
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [VARIABLE, 1, 1],
                [VARIABLE, 2, 2],
                [POWER, 1, 2],
                [POWER, 0, 3],
            ],
            dtype=np.uint8,
        )
        assert get_formatted_string("console", stack, (), ()) == "X0**X1**X2"

    def test_power_left_child_power_needs_parens(self):
        """(X0**X1)**X2  — left POW child of POW needs parens."""
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [VARIABLE, 1, 1],
                [POWER, 0, 1],
                [VARIABLE, 2, 2],
                [POWER, 2, 3],
            ],
            dtype=np.uint8,
        )
        assert get_formatted_string("console", stack, (), ()) == "(X0**X1)**X2"

    def test_square_of_terminal(self):
        """X0**2  (not (X0)**2)"""
        stack = np.array(
            [[VARIABLE, 0, 0], [SQUARE, 0, 0]],
            dtype=np.uint8,
        )
        assert get_formatted_string("console", stack, (), ()) == "X0**2"

    def test_square_of_sum(self):
        """(X0 + X1)**2"""
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [VARIABLE, 1, 1],
                [ADDITION, 0, 1],
                [SQUARE, 2, 2],
            ],
            dtype=np.uint8,
        )
        assert get_formatted_string("console", stack, (), ()) == "(X0 + X1)**2"

    def test_square_of_sin(self):
        """sin(X0)**2  — function child, no parens."""
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [SIN, 0, 0],
                [SQUARE, 1, 1],
            ],
            dtype=np.uint8,
        )
        assert get_formatted_string("console", stack, (), ()) == "sin(X0)**2"

    def test_cube_of_product(self):
        """(X0*X1)**3"""
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [VARIABLE, 1, 1],
                [MULTIPLICATION, 0, 1],
                [CUBE, 2, 2],
            ],
            dtype=np.uint8,
        )
        assert get_formatted_string("console", stack, (), ()) == "(X0*X1)**3"

    def test_safe_power_terminals(self):
        """|X0|**X1  — left inside |…|, right unwrapped."""
        stack = np.array(
            [[VARIABLE, 0, 0], [VARIABLE, 1, 1], [SAFE_POWER, 0, 1]],
            dtype=np.uint8,
        )
        assert get_formatted_string("console", stack, (), ()) == "|X0|**X1"

    def test_nested_division_left_assoc(self):
        """X0/X1/X2  — left-assoc, no parens for left DIV child of DIV."""
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [VARIABLE, 1, 1],
                [DIVISION, 0, 1],
                [VARIABLE, 2, 2],
                [DIVISION, 2, 3],
            ],
            dtype=np.uint8,
        )
        assert get_formatted_string("console", stack, (), ()) == "X0/X1/X2"


# ------------------------------------------------------------------ #
#  Sympy srepr format                                                 #
# ------------------------------------------------------------------ #


class TestSympySreprFormat:
    def test_variable(self):
        stack = np.array([[VARIABLE, 0, 0]], dtype=np.uint8)
        s = get_formatted_string("sympy", stack, (), ())
        assert s == "Symbol('X0')"

    def test_constant(self):
        stack = np.array([[CONSTANT, 0, 0]], dtype=np.uint8)
        s = get_formatted_string("sympy", stack, (3.14,), ())
        assert s == "Float(3.14)"

    def test_integer(self, integer_stack):
        stack, consts, ints = integer_stack
        s = get_formatted_string("sympy", stack, consts, ints)
        assert "Integer(7)" in s

    def test_addition(self, x0_plus_c0_stack):
        stack, consts, ints = x0_plus_c0_stack
        s = get_formatted_string("sympy", stack, consts, ints)
        assert s == "Add(Symbol('X0'), Float(3.0))"

    def test_sin(self, sin_x0_stack):
        stack, consts, ints = sin_x0_stack
        s = get_formatted_string("sympy", stack, consts, ints)
        assert s == "sin(Symbol('X0'))"

    def test_subtraction(self):
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [VARIABLE, 1, 1],
                [SUBTRACTION, 0, 1],
            ],
            dtype=np.uint8,
        )
        s = get_formatted_string("sympy", stack, (), ())
        assert s == "Add(Symbol('X0'), Mul(Integer(-1), Symbol('X1')))"

    def test_multiplication(self):
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [VARIABLE, 1, 1],
                [MULTIPLICATION, 0, 1],
            ],
            dtype=np.uint8,
        )
        s = get_formatted_string("sympy", stack, (), ())
        assert s == "Mul(Symbol('X0'), Symbol('X1'))"

    def test_division(self):
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [VARIABLE, 1, 1],
                [DIVISION, 0, 1],
            ],
            dtype=np.uint8,
        )
        s = get_formatted_string("sympy", stack, (), ())
        assert s == "Mul(Symbol('X0'), Pow(Symbol('X1'), Integer(-1)))"

    def test_power(self):
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [VARIABLE, 1, 1],
                [POWER, 0, 1],
            ],
            dtype=np.uint8,
        )
        s = get_formatted_string("sympy", stack, (), ())
        assert s == "Pow(Symbol('X0'), Symbol('X1'))"

    def test_square(self):
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [SQUARE, 0, 0],
            ],
            dtype=np.uint8,
        )
        s = get_formatted_string("sympy", stack, (), ())
        assert s == "Pow(Symbol('X0'), Integer(2))"

    def test_x0_plus_sin_x1(self):
        """User's example: X0 + sin(X1)"""
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [VARIABLE, 1, 1],
                [SIN, 1, 1],
                [ADDITION, 0, 2],
            ],
            dtype=np.uint8,
        )
        s = get_formatted_string("sympy", stack, (), ())
        assert s == "Add(Symbol('X0'), sin(Symbol('X1')))"

    def test_sympify_roundtrip(self, x0_plus_c0_stack):
        """srepr output can be parsed by sympy.sympify."""
        import sympy

        stack, consts, ints = x0_plus_c0_stack
        s = get_formatted_string("sympy", stack, consts, ints)
        expr = sympy.sympify(s)
        assert isinstance(expr, sympy.Basic)


# ------------------------------------------------------------------ #
#  LaTeX format                                                       #
# ------------------------------------------------------------------ #


class TestLatexFormat:
    def test_x0_plus_c0(self, x0_plus_c0_stack):
        stack, consts, ints = x0_plus_c0_stack
        s = get_formatted_string("latex", stack, consts, ints)
        assert "X0" in s
        assert "3.0" in s

    def test_sin_x0(self, sin_x0_stack):
        stack, consts, ints = sin_x0_stack
        s = get_formatted_string("latex", stack, consts, ints)
        assert "sin" in s

    def test_subtraction_of_terminals(self):
        """X0 - X1  — no parens around terminals."""
        stack = np.array(
            [[VARIABLE, 0, 0], [VARIABLE, 1, 1], [SUBTRACTION, 0, 1]],
            dtype=np.uint8,
        )
        assert get_formatted_string("latex", stack, (), ()) == "X0 - X1"

    def test_multiplication_of_terminals(self):
        r"""X0 \cdot X1  — no parens around terminals."""
        stack = np.array(
            [[VARIABLE, 0, 0], [VARIABLE, 1, 1], [MULTIPLICATION, 0, 1]],
            dtype=np.uint8,
        )
        s = get_formatted_string("latex", stack, (), ())
        assert s == r"X0 \cdot X1"


# ------------------------------------------------------------------ #
#  Integer / edge-case formatting                                     #
# ------------------------------------------------------------------ #


class TestIntegerFormatting:
    def test_console_integer(self, integer_stack):
        stack, consts, ints = integer_stack
        s = get_formatted_string("console", stack, consts, ints)
        assert "7" in s

    def test_sympy_integer(self, integer_stack):
        stack, consts, ints = integer_stack
        s = get_formatted_string("sympy", stack, consts, ints)
        assert "Integer(7)" in s

    def test_missing_constant_shows_question_mark(self):
        stack = np.array([[CONSTANT, 5, 5]], dtype=np.uint8)
        s = get_formatted_string("console", stack, (), ())
        assert s == "?"

    def test_missing_integer_shows_question_mark(self):
        stack = np.array([[INTEGER, 5, 5]], dtype=np.uint8)
        s = get_formatted_string("console", stack, (), ())
        assert s == "?"
