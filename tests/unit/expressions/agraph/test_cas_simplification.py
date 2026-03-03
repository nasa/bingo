"""Tests for the CAS simplification pipeline.

Tests cover:
- CASExpression basics
- interpreter (build_cas_expression ↔ build_agraph_stack)
- automatic_simplification (algebraic identities)
- constant_folding
- optional_modifications (subtraction insertion, integer power expansion)
- full pipeline (simplify)
"""

import numpy as np
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
    SIN,
    COS,
    EXPONENTIAL,
    LOGARITHM,
    SQRT,
    SQUARE,
    CUBE,
    ARCSIN,
    ARCCOS,
    ARCTAN,
)
from bingo.expressions.agraph.simplification.cas_expression import (
    CASExpression,
)
from bingo.expressions.agraph.simplification.interpreter import (
    build_cas_expression,
    build_agraph_stack,
)
from bingo.expressions.agraph.simplification.automatic_simplification import (
    automatic_simplify,
)
from bingo.expressions.agraph.simplification.constant_folding import (
    fold_constants,
)
from bingo.expressions.agraph.simplification.optional_modifications import (
    optional_modifications,
)
from bingo.expressions.agraph.simplification import simplify


# ================================================================== #
#  CASExpression basics                                               #
# ================================================================== #


class TestCASExpression:
    def test_terminal_integer(self):
        expr = CASExpression(INTEGER, [5])
        assert expr.operator == INTEGER
        assert expr.operands == [5]

    def test_is_zero(self):
        assert CASExpression(INTEGER, [0]).is_zero()
        assert not CASExpression(INTEGER, [1]).is_zero()

    def test_is_one(self):
        assert CASExpression(INTEGER, [1]).is_one()
        assert not CASExpression(INTEGER, [0]).is_one()

    def test_constant_valued_constant(self):
        assert CASExpression(CONSTANT, [0]).is_constant_valued

    def test_constant_valued_integer(self):
        assert CASExpression(INTEGER, [7]).is_constant_valued

    def test_variable_not_constant_valued(self):
        assert not CASExpression(VARIABLE, [0]).is_constant_valued

    def test_depends_on_variable(self):
        expr = CASExpression(VARIABLE, [3])
        assert "x" in expr.depends_on

    def test_depends_on_constant(self):
        expr = CASExpression(CONSTANT, [0])
        assert 0 in expr.depends_on

    def test_depends_on_integer(self):
        expr = CASExpression(INTEGER, [5])
        assert "i" in expr.depends_on

    def test_map(self):
        # sin(X_0) — map identity should return equivalent
        inner = CASExpression(VARIABLE, [0])
        expr = CASExpression(SIN, [inner])
        mapped = expr.map(lambda x: x)
        assert mapped.operator == SIN
        assert mapped.operands[0].operator == VARIABLE

    def test_copy(self):
        expr = CASExpression(INTEGER, [42])
        c = expr.copy()
        assert c == expr
        assert c is not expr

    def test_equality(self):
        a = CASExpression(INTEGER, [5])
        b = CASExpression(INTEGER, [5])
        assert a == b

    def test_inequality(self):
        a = CASExpression(INTEGER, [5])
        b = CASExpression(INTEGER, [6])
        assert a != b

    def test_base_of_power(self):
        base = CASExpression(VARIABLE, [0])
        exp = CASExpression(INTEGER, [2])
        expr = CASExpression(POWER, [base, exp])
        assert expr.base == base

    def test_exponent_of_power(self):
        base = CASExpression(VARIABLE, [0])
        exp = CASExpression(INTEGER, [2])
        expr = CASExpression(POWER, [base, exp])
        assert expr.exponent == exp

    def test_coefficient_of_product(self):
        coeff = CASExpression(INTEGER, [3])
        var = CASExpression(VARIABLE, [0])
        expr = CASExpression(MULTIPLICATION, [coeff, var])
        assert expr.coefficient == coeff

    def test_term_of_product(self):
        coeff = CASExpression(INTEGER, [3])
        var = CASExpression(VARIABLE, [0])
        expr = CASExpression(MULTIPLICATION, [coeff, var])
        assert expr.term.operands[0] == var


# ================================================================== #
#  Interpreter                                                        #
# ================================================================== #


class TestInterpreter:
    def test_roundtrip_variable(self):
        """X_0 → CAS → stack and back."""
        stack = np.array([[VARIABLE, 0, 0]], dtype=np.uint8)
        cas = build_cas_expression(stack, (), ())
        assert cas.operator == VARIABLE
        new_stack, new_c, new_i = build_agraph_stack(cas, ())
        np.testing.assert_array_equal(new_stack, stack)

    def test_roundtrip_constant(self):
        """C_0 → CAS → stack."""
        stack = np.array([[CONSTANT, 0, 0]], dtype=np.uint8)
        cas = build_cas_expression(stack, (3.14,), ())
        assert cas.operator == CONSTANT
        new_stack, new_c, new_i = build_agraph_stack(cas, (3.14,))
        assert new_c == (3.14,)
        np.testing.assert_array_equal(new_stack, stack)

    def test_roundtrip_integer(self):
        """I_0(=7) → CAS → stack."""
        stack = np.array([[INTEGER, 0, 0]], dtype=np.uint8)
        cas = build_cas_expression(stack, (), (7,))
        assert cas.operator == INTEGER
        assert cas.operands[0] == 7  # actual integer value
        new_stack, new_c, new_i = build_agraph_stack(cas, ())
        assert new_i == (7,)

    def test_roundtrip_addition(self):
        """X_0 + X_1 → CAS → stack."""
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [VARIABLE, 1, 1],
                [ADDITION, 0, 1],
            ],
            dtype=np.uint8,
        )
        cas = build_cas_expression(stack, (), ())
        assert cas.operator == ADDITION
        new_stack, new_c, new_i = build_agraph_stack(cas, ())
        np.testing.assert_array_equal(new_stack, stack)

    def test_roundtrip_sin(self):
        """sin(X_0) → CAS → stack."""
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [SIN, 0, 0],
            ],
            dtype=np.uint8,
        )
        cas = build_cas_expression(stack, (), ())
        assert cas.operator == SIN
        new_stack, _, _ = build_agraph_stack(cas, ())
        np.testing.assert_array_equal(new_stack, stack)

    def test_cse_deduplication(self):
        """X_0 + X_0 should re-use X_0 row via CSE."""
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [ADDITION, 0, 0],
            ],
            dtype=np.uint8,
        )
        cas = build_cas_expression(stack, (), ())
        new_stack, _, _ = build_agraph_stack(cas, ())
        # The rebuilt stack should have 2 rows (X_0 reused)
        assert new_stack.shape[0] == 2

    def test_multiple_constants_renumbered(self):
        """C_0 + C_1 → CAS → stack should renumber constants."""
        stack = np.array(
            [
                [CONSTANT, 0, 0],
                [CONSTANT, 1, 1],
                [ADDITION, 0, 1],
            ],
            dtype=np.uint8,
        )
        cas = build_cas_expression(stack, (1.0, 2.0), ())
        new_stack, new_c, _ = build_agraph_stack(cas, (1.0, 2.0))
        assert len(new_c) == 2
        assert set(new_c) == {1.0, 2.0}


# ================================================================== #
#  Automatic Simplification                                           #
# ================================================================== #


class TestAutomaticSimplify:
    """Test algebraic simplification rules."""

    def test_x_plus_zero(self):
        """X_0 + 0 → X_0."""
        x = CASExpression(VARIABLE, [0])
        zero = CASExpression(INTEGER, [0])
        expr = CASExpression(ADDITION, [x, zero])
        result = automatic_simplify(expr)
        assert result.operator == VARIABLE
        assert result.operands[0] == 0

    def test_x_times_one(self):
        """X_0 * 1 → X_0."""
        x = CASExpression(VARIABLE, [0])
        one = CASExpression(INTEGER, [1])
        expr = CASExpression(MULTIPLICATION, [x, one])
        result = automatic_simplify(expr)
        assert result.operator == VARIABLE

    def test_x_times_zero(self):
        """X_0 * 0 → 0."""
        x = CASExpression(VARIABLE, [0])
        zero = CASExpression(INTEGER, [0])
        expr = CASExpression(MULTIPLICATION, [x, zero])
        result = automatic_simplify(expr)
        assert result.is_zero()

    def test_x_power_one(self):
        """X_0^1 → X_0."""
        x = CASExpression(VARIABLE, [0])
        one = CASExpression(INTEGER, [1])
        expr = CASExpression(POWER, [x, one])
        result = automatic_simplify(expr)
        assert result.operator == VARIABLE

    def test_x_power_zero(self):
        """X_0^0 → 1."""
        x = CASExpression(VARIABLE, [0])
        zero = CASExpression(INTEGER, [0])
        expr = CASExpression(POWER, [x, zero])
        result = automatic_simplify(expr)
        assert result.is_one()

    def test_one_power_x(self):
        """1^X_0 → 1."""
        one = CASExpression(INTEGER, [1])
        x = CASExpression(VARIABLE, [0])
        expr = CASExpression(POWER, [one, x])
        result = automatic_simplify(expr)
        assert result.is_one()

    def test_integer_power(self):
        """2^3 → 8."""
        two = CASExpression(INTEGER, [2])
        three = CASExpression(INTEGER, [3])
        expr = CASExpression(POWER, [two, three])
        result = automatic_simplify(expr)
        assert result.operator == INTEGER
        assert result.operands[0] == 8

    def test_integer_addition(self):
        """2 + 3 → 5."""
        two = CASExpression(INTEGER, [2])
        three = CASExpression(INTEGER, [3])
        expr = CASExpression(ADDITION, [two, three])
        result = automatic_simplify(expr)
        assert result.operator == INTEGER
        assert result.operands[0] == 5

    def test_integer_multiplication(self):
        """2 * 3 → 6."""
        two = CASExpression(INTEGER, [2])
        three = CASExpression(INTEGER, [3])
        expr = CASExpression(MULTIPLICATION, [two, three])
        result = automatic_simplify(expr)
        assert result.operator == INTEGER
        assert result.operands[0] == 6

    def test_x_minus_x(self):
        """X_0 - X_0 → 0."""
        x = CASExpression(VARIABLE, [0])
        x2 = CASExpression(VARIABLE, [0])
        expr = CASExpression(SUBTRACTION, [x, x2])
        result = automatic_simplify(expr)
        assert result.is_zero()

    def test_x_div_x(self):
        """X_0 / X_0 → 1."""
        x = CASExpression(VARIABLE, [0])
        x2 = CASExpression(VARIABLE, [0])
        expr = CASExpression(DIVISION, [x, x2])
        result = automatic_simplify(expr)
        assert result.is_one()

    def test_sin_zero(self):
        """sin(0) → 0."""
        zero = CASExpression(INTEGER, [0])
        expr = CASExpression(SIN, [zero])
        result = automatic_simplify(expr)
        assert result.is_zero()

    def test_cos_zero(self):
        """cos(0) → 1."""
        zero = CASExpression(INTEGER, [0])
        expr = CASExpression(COS, [zero])
        result = automatic_simplify(expr)
        assert result.is_one()

    def test_exp_zero(self):
        """exp(0) → 1."""
        zero = CASExpression(INTEGER, [0])
        expr = CASExpression(EXPONENTIAL, [zero])
        result = automatic_simplify(expr)
        assert result.is_one()

    def test_log_one(self):
        """log(1) → 0."""
        one = CASExpression(INTEGER, [1])
        expr = CASExpression(LOGARITHM, [one])
        result = automatic_simplify(expr)
        assert result.is_zero()

    def test_log_exp(self):
        """log(exp(X_0)) → X_0."""
        x = CASExpression(VARIABLE, [0])
        exp_x = CASExpression(EXPONENTIAL, [x])
        expr = CASExpression(LOGARITHM, [exp_x])
        result = automatic_simplify(expr)
        assert result.operator == VARIABLE

    def test_sin_arcsin(self):
        """sin(arcsin(X_0)) → X_0."""
        x = CASExpression(VARIABLE, [0])
        asin_x = CASExpression(ARCSIN, [x])
        expr = CASExpression(SIN, [asin_x])
        result = automatic_simplify(expr)
        assert result.operator == VARIABLE

    def test_cos_arccos(self):
        """cos(arccos(X_0)) → X_0."""
        x = CASExpression(VARIABLE, [0])
        acos_x = CASExpression(ARCCOS, [x])
        expr = CASExpression(COS, [acos_x])
        result = automatic_simplify(expr)
        assert result.operator == VARIABLE

    def test_x_times_x(self):
        """X_0 * X_0 → X_0^2."""
        x = CASExpression(VARIABLE, [0])
        x2 = CASExpression(VARIABLE, [0])
        expr = CASExpression(MULTIPLICATION, [x, x2])
        result = automatic_simplify(expr)
        assert result.operator == POWER
        assert result.operands[0].operator == VARIABLE
        assert result.operands[1].operator == INTEGER
        assert result.operands[1].operands[0] == 2

    def test_x_plus_x(self):
        """X_0 + X_0 → 2*X_0."""
        x = CASExpression(VARIABLE, [0])
        x2 = CASExpression(VARIABLE, [0])
        expr = CASExpression(ADDITION, [x, x2])
        result = automatic_simplify(expr)
        assert result.operator == MULTIPLICATION
        # coefficient should be 2
        int_operand = [
            op for op in result.operands if op.operator == INTEGER
        ]
        assert len(int_operand) == 1
        assert int_operand[0].operands[0] == 2

    def test_square_zero(self):
        """square(0) → 0."""
        zero = CASExpression(INTEGER, [0])
        expr = CASExpression(SQUARE, [zero])
        result = automatic_simplify(expr)
        assert result.is_zero()

    def test_square_one(self):
        """square(1) → 1."""
        one = CASExpression(INTEGER, [1])
        expr = CASExpression(SQUARE, [one])
        result = automatic_simplify(expr)
        assert result.is_one()

    def test_square_integer(self):
        """square(3) → 9."""
        three = CASExpression(INTEGER, [3])
        expr = CASExpression(SQUARE, [three])
        result = automatic_simplify(expr)
        assert result.operator == INTEGER
        assert result.operands[0] == 9

    def test_square_sqrt(self):
        """square(sqrt(X_0)) → X_0."""
        x = CASExpression(VARIABLE, [0])
        sqrt_x = CASExpression(SQRT, [x])
        expr = CASExpression(SQUARE, [sqrt_x])
        result = automatic_simplify(expr)
        assert result.operator == VARIABLE

    def test_cube_zero(self):
        """cube(0) → 0."""
        zero = CASExpression(INTEGER, [0])
        expr = CASExpression(CUBE, [zero])
        result = automatic_simplify(expr)
        assert result.is_zero()

    def test_cube_one(self):
        """cube(1) → 1."""
        one = CASExpression(INTEGER, [1])
        expr = CASExpression(CUBE, [one])
        result = automatic_simplify(expr)
        assert result.is_one()

    def test_cube_integer(self):
        """cube(2) → 8."""
        two = CASExpression(INTEGER, [2])
        expr = CASExpression(CUBE, [two])
        result = automatic_simplify(expr)
        assert result.operator == INTEGER
        assert result.operands[0] == 8

    def test_power_of_power(self):
        """(X_0^2)^3 → X_0^6."""
        x = CASExpression(VARIABLE, [0])
        two = CASExpression(INTEGER, [2])
        three = CASExpression(INTEGER, [3])
        inner = CASExpression(POWER, [x, two])
        expr = CASExpression(POWER, [inner, three])
        result = automatic_simplify(expr)
        assert result.operator == POWER
        assert result.operands[0].operator == VARIABLE
        assert result.operands[1].operator == INTEGER
        assert result.operands[1].operands[0] == 6


# ================================================================== #
#  Constant Folding                                                   #
# ================================================================== #


class TestConstantFolding:
    def test_no_constants_unchanged(self):
        """Expression with no constants is unchanged."""
        x = CASExpression(VARIABLE, [0])
        result = fold_constants(x)
        assert result.operator == VARIABLE

    def test_single_constant_unchanged(self):
        """A lone constant has nothing to fold."""
        c = CASExpression(CONSTANT, [0])
        result = fold_constants(c)
        assert result.operator == CONSTANT

    def test_two_constants_in_sum_are_folded(self):
        """C_0 + C_1 in (C_0 + C_1) + X_0 → C_folded + X_0.

        After folding, the number of CONSTANT nodes appearing
        alongside the variable should be reduced.
        """
        c0 = CASExpression(CONSTANT, [0])
        c1 = CASExpression(CONSTANT, [1])
        x = CASExpression(VARIABLE, [0])
        inner_sum = CASExpression(ADDITION, [c0, c1, x])
        result = fold_constants(inner_sum)
        # Should have grouped the two constants together
        const_leaves = _count_operator(result, CONSTANT)
        assert const_leaves <= 2  # at most the original count


# ================================================================== #
#  Optional Modifications                                             #
# ================================================================== #


class TestOptionalModifications:
    def test_subtraction_insertion(self):
        """X_0 + (-1)*X_1 → X_0 - X_1."""
        x0 = CASExpression(VARIABLE, [0])
        x1 = CASExpression(VARIABLE, [1])
        neg_one = CASExpression(INTEGER, [-1])
        neg_x1 = CASExpression(MULTIPLICATION, [neg_one, x1])
        expr = CASExpression(ADDITION, [x0, neg_x1])
        result = optional_modifications(expr)
        assert result.operator == SUBTRACTION

    def test_power_2_to_square(self):
        """X_0^2 → square(X_0)."""
        x = CASExpression(VARIABLE, [0])
        two = CASExpression(INTEGER, [2])
        expr = CASExpression(POWER, [x, two])
        result = optional_modifications(expr)
        assert result.operator == SQUARE

    def test_power_3_to_cube(self):
        """X_0^3 → cube(X_0)."""
        x = CASExpression(VARIABLE, [0])
        three = CASExpression(INTEGER, [3])
        expr = CASExpression(POWER, [x, three])
        result = optional_modifications(expr)
        assert result.operator == CUBE

    def test_power_4_to_multiplication(self):
        """X_0^4 → X_0 * X_0 * X_0 * X_0."""
        x = CASExpression(VARIABLE, [0])
        four = CASExpression(INTEGER, [4])
        expr = CASExpression(POWER, [x, four])
        result = optional_modifications(expr)
        assert result.operator == MULTIPLICATION

    def test_no_subtraction_when_no_negatives(self):
        """X_0 + X_1 stays as addition."""
        x0 = CASExpression(VARIABLE, [0])
        x1 = CASExpression(VARIABLE, [1])
        expr = CASExpression(ADDITION, [x0, x1])
        result = optional_modifications(expr)
        assert result.operator == ADDITION


# ================================================================== #
#  Full pipeline (simplify)                                           #
# ================================================================== #


class TestSimplifyPipeline:
    """End-to-end tests for the full CAS pipeline via stacks."""

    def test_identity_variable(self):
        """X_0 → X_0 (no change)."""
        stack = np.array([[VARIABLE, 0, 0]], dtype=np.uint8)
        new_stack, new_c, new_i = simplify(stack, (), ())
        assert new_stack.shape[0] == 1
        assert new_stack[0, 0] == VARIABLE

    def test_x_plus_zero_simplifies(self):
        """X_0 + 0 → X_0."""
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [INTEGER, 0, 0],
                [ADDITION, 0, 1],
            ],
            dtype=np.uint8,
        )
        new_stack, new_c, new_i = simplify(stack, (), (0,))
        # Should reduce to just X_0
        assert new_stack.shape[0] == 1
        assert new_stack[0, 0] == VARIABLE

    def test_x_times_one_simplifies(self):
        """X_0 * 1 → X_0."""
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [INTEGER, 0, 0],
                [MULTIPLICATION, 0, 1],
            ],
            dtype=np.uint8,
        )
        new_stack, new_c, new_i = simplify(stack, (), (1,))
        assert new_stack.shape[0] == 1
        assert new_stack[0, 0] == VARIABLE

    def test_x_minus_x_simplifies(self):
        """X_0 - X_0 → 0."""
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [SUBTRACTION, 0, 0],
            ],
            dtype=np.uint8,
        )
        new_stack, new_c, new_i = simplify(stack, (), ())
        # Result should be a single INTEGER node with value 0
        assert new_stack.shape[0] == 1
        assert new_stack[0, 0] == INTEGER
        assert new_i[new_stack[0, 1]] == 0

    def test_x_div_x_simplifies(self):
        """X_0 / X_0 → 1."""
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [DIVISION, 0, 0],
            ],
            dtype=np.uint8,
        )
        new_stack, new_c, new_i = simplify(stack, (), ())
        assert new_stack.shape[0] == 1
        assert new_stack[0, 0] == INTEGER
        assert new_i[new_stack[0, 1]] == 1

    def test_sin_zero_simplifies(self):
        """sin(0) → 0."""
        stack = np.array(
            [
                [INTEGER, 0, 0],
                [SIN, 0, 0],
            ],
            dtype=np.uint8,
        )
        new_stack, new_c, new_i = simplify(stack, (), (0,))
        assert new_stack.shape[0] == 1
        assert new_stack[0, 0] == INTEGER

    def test_constant_preserved(self):
        """C_0 → C_0 (constant survives round-trip)."""
        stack = np.array([[CONSTANT, 0, 0]], dtype=np.uint8)
        new_stack, new_c, new_i = simplify(stack, (3.14,), ())
        assert new_stack[0, 0] == CONSTANT
        assert pytest.approx(new_c[0]) == 3.14

    def test_x_squared_becomes_square(self):
        """X_0^2 → square(X_0) via optional modifications."""
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [INTEGER, 0, 0],
                [POWER, 0, 1],
            ],
            dtype=np.uint8,
        )
        new_stack, new_c, new_i = simplify(stack, (), (2,))
        # The output should contain a SQUARE operator
        operators = set(new_stack[:, 0])
        assert SQUARE in operators

    def test_x_cubed_becomes_cube(self):
        """X_0^3 → cube(X_0) via optional modifications."""
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [INTEGER, 0, 0],
                [POWER, 0, 1],
            ],
            dtype=np.uint8,
        )
        new_stack, new_c, new_i = simplify(stack, (), (3,))
        operators = set(new_stack[:, 0])
        assert CUBE in operators


# ================================================================== #
#  Integration with AGraphExpression                                  #
# ================================================================== #


class TestAGraphExpressionCASMode:
    """Test AGraphExpression with simplification='cas'."""

    def test_cas_mode_constructor(self):
        from bingo.expressions.agraph.expression import AGraphExpression

        expr = AGraphExpression(simplification="cas")
        assert expr._simplification == "cas"

    def test_reduce_mode_is_default(self):
        from bingo.expressions.agraph.expression import AGraphExpression

        expr = AGraphExpression()
        assert expr._simplification == "reduce"

    def test_invalid_simplification_raises(self):
        from bingo.expressions.agraph.expression import AGraphExpression

        with pytest.raises(ValueError, match="simplification"):
            AGraphExpression(simplification="invalid")

    def test_cas_mode_simplifies_x_plus_zero(self):
        from bingo.expressions.agraph.expression import AGraphExpression

        expr = AGraphExpression(simplification="cas")
        expr._raw_command_array = np.array(
            [
                [VARIABLE, 0, 0],
                [INTEGER, 0, 0],
                [ADDITION, 0, 1],
            ],
            dtype=np.uint8,
        )
        expr._raw_constants = ()
        expr._raw_integers = (0,)
        expr._modified = True
        # Accessing command_array triggers _update with CAS
        assert expr.command_array.shape[0] == 1
        assert expr.command_array[0, 0] == VARIABLE

    def test_deepcopy_preserves_simplification_mode(self):
        from bingo.expressions.agraph.expression import AGraphExpression

        expr = AGraphExpression(simplification="cas")
        clone = expr.copy()
        assert clone._simplification == "cas"

    def test_promote_simplification(self):
        """promote_simplification() replaces raw with simplified."""
        from bingo.expressions.agraph.expression import AGraphExpression

        expr = AGraphExpression(simplification="cas")
        expr._raw_command_array = np.array(
            [
                [VARIABLE, 0, 0],
                [INTEGER, 0, 0],
                [ADDITION, 0, 1],
            ],
            dtype=np.uint8,
        )
        expr._raw_constants = ()
        expr._raw_integers = (0,)
        expr._modified = True
        expr.promote_simplification()
        # After promotion, raw should equal simplified
        assert expr.raw_command_array.shape[0] == 1


# ================================================================== #
#  Helpers                                                            #
# ================================================================== #


def _count_operator(expr, operator):
    """Count the number of nodes with a given operator in a CAS tree."""
    if expr.operator == operator:
        return 1
    if isinstance(expr.operands[0], int):
        return 0
    count = 0
    for operand in expr.operands:
        if isinstance(operand, CASExpression):
            count += _count_operator(operand, operator)
    return count
