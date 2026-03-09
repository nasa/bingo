"""Tests for bingo.expressions.agraph.evaluation.cached_evaluation"""

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
    EXPONENTIAL,
    SQRT,
)
from bingo.expressions.agraph.pyagraph.evaluation import evaluation
from bingo.expressions.agraph.pyagraph.evaluation.cached_evaluation import (
    CachedEvaluator,
    _build_dependency_mask,
    _build_reverse_variant_mask,
)


# ------------------------------------------------------------------ #
#  Fixtures                                                            #
# ------------------------------------------------------------------ #


@pytest.fixture
def simple_x():
    return np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


@pytest.fixture
def x0_plus_c0_stack():
    """Stack: [X0, C0, X0 + C0]  —  row 0 independent, rows 1–2 dependent."""
    return np.array(
        [
            [VARIABLE, 0, 0],
            [CONSTANT, 0, 0],
            [ADDITION, 0, 1],
        ],
        dtype=np.uint8,
    )


@pytest.fixture
def c0_times_x0_stack():
    """Stack: [C0, X0, C0 * X0]"""
    return np.array(
        [
            [CONSTANT, 0, 0],
            [VARIABLE, 0, 0],
            [MULTIPLICATION, 0, 1],
        ],
        dtype=np.uint8,
    )


@pytest.fixture
def pure_variable_stack():
    """Stack: [X0, X1, X0 + X1]  —  no constants, nothing dependent."""
    return np.array(
        [
            [VARIABLE, 0, 0],
            [VARIABLE, 1, 1],
            [ADDITION, 0, 1],
        ],
        dtype=np.uint8,
    )


@pytest.fixture
def mixed_stack():
    """Stack: [X0, C0, X1, X0 * C0, X1 + I0, row3 - row4]

    Dependencies:
      row 0: X0         -> independent
      row 1: C0         -> dependent
      row 2: X1         -> independent
      row 3: X0 * C0    -> dependent (via row 1)
      row 4: X1 + I0    -> independent (INTEGER is not CONSTANT)
      row 5: row3 - row4 -> dependent (via row 3)
    """
    return np.array(
        [
            [VARIABLE, 0, 0],
            [CONSTANT, 0, 0],
            [VARIABLE, 1, 1],
            [MULTIPLICATION, 0, 1],
            [ADDITION, 2, 2],  # X1 + X1 (simplified, don't need INTEGER for this)
            [SUBTRACTION, 3, 4],
        ],
        dtype=np.uint8,
    )


@pytest.fixture
def two_constant_stack():
    """Stack: [C0, C1, X0, C0 * X0, C1 + row3]

    f(x) = C1 + C0 * X0
    df/dC0 = X0, df/dC1 = 1
    """
    return np.array(
        [
            [CONSTANT, 0, 0],
            [CONSTANT, 1, 1],
            [VARIABLE, 0, 0],
            [MULTIPLICATION, 0, 2],
            [ADDITION, 1, 3],
        ],
        dtype=np.uint8,
    )


# ------------------------------------------------------------------ #
#  _build_dependency_mask                                              #
# ------------------------------------------------------------------ #


class TestBuildDependencyMask:
    def test_single_variable_not_dependent(self):
        stack = np.array([[VARIABLE, 0, 0]], dtype=np.uint8)
        mask = _build_dependency_mask(stack)
        assert mask.tolist() == [False]

    def test_single_constant_dependent(self):
        stack = np.array([[CONSTANT, 0, 0]], dtype=np.uint8)
        mask = _build_dependency_mask(stack)
        assert mask.tolist() == [True]

    def test_single_integer_not_dependent(self):
        stack = np.array([[INTEGER, 0, 0]], dtype=np.uint8)
        mask = _build_dependency_mask(stack)
        assert mask.tolist() == [False]

    def test_x0_plus_c0(self, x0_plus_c0_stack):
        mask = _build_dependency_mask(x0_plus_c0_stack)
        assert mask.tolist() == [False, True, True]

    def test_pure_variable_none_dependent(self, pure_variable_stack):
        mask = _build_dependency_mask(pure_variable_stack)
        assert mask.tolist() == [False, False, False]

    def test_mixed_transitive_dependency(self, mixed_stack):
        mask = _build_dependency_mask(mixed_stack)
        # row0=X0(F), row1=C0(T), row2=X1(F), row3=X0*C0(T),
        # row4=X1+X1(F), row5=row3-row4(T)
        assert mask.tolist() == [False, True, False, True, False, True]


# ------------------------------------------------------------------ #
#  _build_reverse_variant_mask                                         #
# ------------------------------------------------------------------ #


class TestBuildReverseVariantMask:
    def test_pure_variables_all_invariant(self, pure_variable_stack):
        fwd = _build_dependency_mask(pure_variable_stack)
        rev_var, row_var = _build_reverse_variant_mask(pure_variable_stack, fwd)
        assert rev_var.tolist() == [False, False, False]
        assert row_var.tolist() == [False, False, False]

    def test_add_with_constant_stays_invariant(self, x0_plus_c0_stack):
        """ADD reverse does not read forward values. f = X0+C0, df/dC0 = 1.
        The entire reverse pass is invariant."""
        fwd = _build_dependency_mask(x0_plus_c0_stack)
        rev_var, row_var = _build_reverse_variant_mask(x0_plus_c0_stack, fwd)
        assert rev_var.tolist() == [False, False, False]
        assert row_var.tolist() == [False, False, False]

    def test_mul_with_constant_is_variant(self, c0_times_x0_stack):
        """MUL reads forward values. f = C0*X0, reverse for MUL is variant."""
        fwd = _build_dependency_mask(c0_times_x0_stack)
        # fwd = [T, F, T]
        rev_var, row_var = _build_reverse_variant_mask(c0_times_x0_stack, fwd)
        # row 2: MUL, reverse_variant[2]=F, forward_depends[2]=T → variant
        # → reverse_variant[0]=T, reverse_variant[1]=T
        assert rev_var.tolist() == [True, True, False]
        assert row_var.tolist() == [False, False, True]

    def test_add_chain_preserves_invariance(self):
        """Pure ADD/SUB chain above constants has fully invariant reverse.
        f = (X0 + C0) + (X0 + C0) = 2*X0 + 2*C0."""
        stack = np.array(
            [
                [VARIABLE, 0, 0],  # 0: X0
                [CONSTANT, 0, 0],  # 1: C0
                [ADDITION, 0, 1],  # 2: X0 + C0
                [ADDITION, 2, 2],  # 3: (X0+C0)+(X0+C0)
            ],
            dtype=np.uint8,
        )
        fwd = _build_dependency_mask(stack)  # [F, T, T, T]
        rev_var, row_var = _build_reverse_variant_mask(stack, fwd)
        assert rev_var.tolist() == [False, False, False, False]
        assert row_var.tolist() == [False, False, False, False]

    def test_sub_preserves_invariance(self):
        """SUB reverse also does not read forward."""
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [CONSTANT, 0, 0],
                [SUBTRACTION, 0, 1],
            ],
            dtype=np.uint8,
        )
        fwd = _build_dependency_mask(stack)
        rev_var, row_var = _build_reverse_variant_mask(stack, fwd)
        assert rev_var.tolist() == [False, False, False]
        assert row_var.tolist() == [False, False, False]

    def test_sin_of_constant_is_variant(self):
        """Unary SIN reads forward values → variant."""
        stack = np.array(
            [
                [CONSTANT, 0, 0],
                [SIN, 0, 0],
            ],
            dtype=np.uint8,
        )
        fwd = _build_dependency_mask(stack)  # [T, T]
        rev_var, row_var = _build_reverse_variant_mask(stack, fwd)
        # row 1: SIN, fwd_dep=T → variant → reverse_variant[0]=T
        assert rev_var.tolist() == [True, False]
        assert row_var.tolist() == [False, True]

    def test_mixed_mul_under_add(self):
        """ADD on top of MUL: ADD stays invariant, MUL is variant.
        f = C0 + C0*X0 = C0*(1+X0), df/dC0 = 1+X0."""
        stack = np.array(
            [
                [CONSTANT, 0, 0],  # 0: C0
                [VARIABLE, 0, 0],  # 1: X0
                [MULTIPLICATION, 0, 1],  # 2: C0*X0
                [ADDITION, 0, 2],  # 3: C0 + C0*X0
            ],
            dtype=np.uint8,
        )
        fwd = _build_dependency_mask(stack)  # [T, F, T, T]
        rev_var, row_var = _build_reverse_variant_mask(stack, fwd)
        # row 3: ADD, reverse_variant[3]=F → invariant
        # row 2: MUL, fwd_dep[2]=T → variant → rev[0]=T, rev[1]=T
        assert rev_var.tolist() == [True, True, False, False]
        assert row_var.tolist() == [False, False, True, False]


# ------------------------------------------------------------------ #
#  CachedEvaluator.forward_eval — matches uncached evaluate()          #
# ------------------------------------------------------------------ #


class TestCachedForwardEval:
    def test_matches_uncached_single_variable(self, simple_x):
        stack = np.array([[VARIABLE, 0, 0]], dtype=np.uint8)
        cached = CachedEvaluator(stack, simple_x, ())
        expected = evaluation.evaluate(stack, simple_x, (), ())
        result = cached.forward_eval(())
        np.testing.assert_array_almost_equal(result, expected)

    def test_matches_uncached_x0_plus_c0(self, simple_x, x0_plus_c0_stack):
        constants = (10.0,)
        cached = CachedEvaluator(x0_plus_c0_stack, simple_x, ())
        expected = evaluation.evaluate(x0_plus_c0_stack, simple_x, constants, ())
        result = cached.forward_eval(constants)
        np.testing.assert_array_almost_equal(result, expected)

    def test_matches_after_constant_change(self, simple_x, x0_plus_c0_stack):
        cached = CachedEvaluator(x0_plus_c0_stack, simple_x, ())

        for c_val in [10.0, 20.0, -5.0]:
            constants = (c_val,)
            expected = evaluation.evaluate(x0_plus_c0_stack, simple_x, constants, ())
            result = cached.forward_eval(constants)
            np.testing.assert_array_almost_equal(result, expected)

    def test_matches_two_constants(self, simple_x, two_constant_stack):
        constants = (3.0, 7.0)
        cached = CachedEvaluator(two_constant_stack, simple_x, ())
        expected = evaluation.evaluate(two_constant_stack, simple_x, constants, ())
        result = cached.forward_eval(constants)
        np.testing.assert_array_almost_equal(result, expected)

    def test_matches_with_integers(self, simple_x):
        # I0 + X0  where I0 = 7
        stack = np.array(
            [
                [INTEGER, 0, 0],
                [VARIABLE, 0, 0],
                [ADDITION, 0, 1],
            ],
            dtype=np.uint8,
        )
        integers = (7,)
        cached = CachedEvaluator(stack, simple_x, integers)
        expected = evaluation.evaluate(stack, simple_x, (), integers)
        result = cached.forward_eval(())
        np.testing.assert_array_almost_equal(result, expected)


# ------------------------------------------------------------------ #
#  CachedEvaluator.forward_eval_with_const_derivative                  #
# ------------------------------------------------------------------ #


class TestCachedDerivative:
    def test_matches_uncached_c0_times_x0(self, simple_x, c0_times_x0_stack):
        # f(x) = C0 * X0   =>  df/dC0 = X0
        constants = (2.0,)
        cached = CachedEvaluator(c0_times_x0_stack, simple_x, ())
        expected_f, expected_jac = evaluation.evaluate_with_derivative(
            c0_times_x0_stack, simple_x, constants, (), False
        )
        result_f, result_jac = cached.forward_eval_with_const_derivative(constants)
        np.testing.assert_array_almost_equal(result_f, expected_f)
        np.testing.assert_array_almost_equal(result_jac, expected_jac)

    def test_matches_two_constants(self, simple_x, two_constant_stack):
        # f = C1 + C0 * X0  =>  df/dC0 = X0, df/dC1 = 1
        constants = (3.0, 7.0)
        cached = CachedEvaluator(two_constant_stack, simple_x, ())
        expected_f, expected_jac = evaluation.evaluate_with_derivative(
            two_constant_stack, simple_x, constants, (), False
        )
        result_f, result_jac = cached.forward_eval_with_const_derivative(constants)
        np.testing.assert_array_almost_equal(result_f, expected_f)
        np.testing.assert_array_almost_equal(result_jac, expected_jac)

    def test_derivative_changes_with_constants(self, simple_x, two_constant_stack):
        cached = CachedEvaluator(two_constant_stack, simple_x, ())
        for c0, c1 in [(1.0, 2.0), (5.0, -3.0), (0.0, 0.0)]:
            constants = (c0, c1)
            expected_f, expected_jac = evaluation.evaluate_with_derivative(
                two_constant_stack, simple_x, constants, (), False
            )
            result_f, result_jac = cached.forward_eval_with_const_derivative(constants)
            np.testing.assert_array_almost_equal(result_f, expected_f)
            np.testing.assert_array_almost_equal(result_jac, expected_jac)


# ------------------------------------------------------------------ #
#  Fused residual/jacobian: forward buffer reuse                       #
# ------------------------------------------------------------------ #


class TestFusedResidualJacobian:
    def test_jacobian_reuses_forward_buffer_after_residual(
        self, simple_x, c0_times_x0_stack
    ):
        """After forward_eval, forward_eval_with_const_derivative should
        reuse the cached forward buffer when params match."""
        cached = CachedEvaluator(c0_times_x0_stack, simple_x, ())
        constants = (2.0,)

        # Call forward_eval (as residuals would)
        cached.forward_eval(constants)
        saved_forward = cached._last_forward

        # Call derivative (as jacobian would)  — should reuse
        cached.forward_eval_with_const_derivative(constants)
        assert cached._last_forward is saved_forward

    def test_jacobian_does_not_reuse_on_different_params(
        self, simple_x, c0_times_x0_stack
    ):
        cached = CachedEvaluator(c0_times_x0_stack, simple_x, ())

        cached.forward_eval((2.0,))
        saved_forward = cached._last_forward

        # Different constants — must recompute
        cached.forward_eval_with_const_derivative((3.0,))
        assert cached._last_forward is not saved_forward


# ------------------------------------------------------------------ #
#  Partial re-evaluation: caching of static rows                       #
# ------------------------------------------------------------------ #


class TestPartialReevaluation:
    def test_static_cache_populated_after_first_eval(self, simple_x, x0_plus_c0_stack):
        cached = CachedEvaluator(x0_plus_c0_stack, simple_x, ())
        assert cached._static_forward is None

        cached.forward_eval((10.0,))
        assert cached._static_forward is not None

        # Row 0 (X0) should be cached, row 1 (C0) should be None
        assert cached._static_forward[0] is not None  # X0 is static
        assert cached._static_forward[1] is None  # C0 is dynamic
        assert cached._static_forward[2] is None  # X0+C0 is dynamic

    def test_static_rows_unchanged_across_evals(self, simple_x, x0_plus_c0_stack):
        cached = CachedEvaluator(x0_plus_c0_stack, simple_x, ())

        cached.forward_eval((10.0,))
        static_row0 = cached._static_forward[0]

        cached.forward_eval((20.0,))
        # Static row 0 should be the exact same object
        assert cached._static_forward[0] is static_row0

    def test_all_static_when_no_constants(self, simple_x, pure_variable_stack):
        """When there are no constants, all rows should be cached."""
        cached = CachedEvaluator(pure_variable_stack, simple_x, ())
        cached.forward_eval(())
        for i, entry in enumerate(cached._static_forward):
            assert entry is not None, f"Row {i} should be static"


# ------------------------------------------------------------------ #
#  Partial reverse re-evaluation                                       #
# ------------------------------------------------------------------ #


class TestPartialReverseReevaluation:
    def test_reverse_cache_populated_after_first_derivative(
        self, simple_x, x0_plus_c0_stack
    ):
        cached = CachedEvaluator(x0_plus_c0_stack, simple_x, ())
        assert cached._static_reverse is None
        assert cached._static_derivative is None

        cached.forward_eval_with_const_derivative((10.0,))
        assert cached._static_reverse is not None
        assert cached._static_derivative is not None

    def test_add_chain_fully_cached(self, simple_x, x0_plus_c0_stack):
        """For f = X0+C0, the reverse pass is fully invariant.
        The static derivative should equal the full derivative."""
        constants = (5.0,)
        cached = CachedEvaluator(x0_plus_c0_stack, simple_x, ())

        _, jac1 = cached.forward_eval_with_const_derivative(constants)
        # Second call with different constants — should still be correct
        _, jac2 = cached.forward_eval_with_const_derivative((99.0,))

        expected_f2, expected_jac2 = evaluation.evaluate_with_derivative(
            x0_plus_c0_stack, simple_x, (99.0,), (), False
        )
        np.testing.assert_array_almost_equal(jac2, expected_jac2)

    def test_mul_variant_rows_recomputed(self, simple_x, c0_times_x0_stack):
        """For f = C0*X0, the MUL reverse is variant.
        Derivatives must update when constants change."""
        cached = CachedEvaluator(c0_times_x0_stack, simple_x, ())

        for c_val in [2.0, 5.0, -1.0]:
            constants = (c_val,)
            _, jac = cached.forward_eval_with_const_derivative(constants)
            _, expected_jac = evaluation.evaluate_with_derivative(
                c0_times_x0_stack, simple_x, constants, (), False
            )
            np.testing.assert_array_almost_equal(jac, expected_jac)

    def test_correctness_two_constants(self, simple_x, two_constant_stack):
        """f = C1 + C0*X0. Derivative changes with constants."""
        cached = CachedEvaluator(two_constant_stack, simple_x, ())

        for c0, c1 in [(1.0, 2.0), (5.0, -3.0), (0.0, 0.0)]:
            constants = (c0, c1)
            _, jac = cached.forward_eval_with_const_derivative(constants)
            _, expected_jac = evaluation.evaluate_with_derivative(
                two_constant_stack, simple_x, constants, (), False
            )
            np.testing.assert_array_almost_equal(jac, expected_jac)

    def test_static_reverse_not_corrupted_across_calls(
        self, simple_x, c0_times_x0_stack
    ):
        """The static reverse cache should not be mutated by subsequent
        cached calls."""
        cached = CachedEvaluator(c0_times_x0_stack, simple_x, ())

        # First call populates cache
        cached.forward_eval_with_const_derivative((2.0,))
        static_copy = [
            v.copy() if isinstance(v, np.ndarray) else v for v in cached._static_reverse
        ]
        static_deriv_copy = cached._static_derivative.copy()

        # Several more calls
        for c in [5.0, -1.0, 100.0]:
            cached.forward_eval_with_const_derivative((c,))

        # Verify cache is unchanged
        for i, (orig, cur) in enumerate(zip(static_copy, cached._static_reverse)):
            if isinstance(orig, np.ndarray):
                np.testing.assert_array_equal(
                    orig, cur, err_msg=f"static_reverse[{i}] was mutated"
                )
            else:
                assert orig == cur, f"static_reverse[{i}] was mutated"
        np.testing.assert_array_equal(static_deriv_copy, cached._static_derivative)

    def test_complex_expression_correctness(self, simple_x):
        """SIN(C0*X0) + C1: MUL and SIN are variant, ADD is invariant."""
        stack = np.array(
            [
                [CONSTANT, 0, 0],  # 0: C0
                [VARIABLE, 0, 0],  # 1: X0
                [MULTIPLICATION, 0, 1],  # 2: C0*X0
                [SIN, 2, 2],  # 3: SIN(C0*X0)
                [CONSTANT, 1, 1],  # 4: C1
                [ADDITION, 3, 4],  # 5: SIN(C0*X0) + C1
            ],
            dtype=np.uint8,
        )

        cached = CachedEvaluator(stack, simple_x, ())
        for c0, c1 in [(1.0, 2.0), (3.0, -1.0), (0.5, 0.5)]:
            constants = (c0, c1)
            _, jac = cached.forward_eval_with_const_derivative(constants)
            _, expected_jac = evaluation.evaluate_with_derivative(
                stack, simple_x, constants, (), False
            )
            np.testing.assert_array_almost_equal(jac, expected_jac)


# ------------------------------------------------------------------ #
#  Integration: fit() with CachedEvaluator                            #
# ------------------------------------------------------------------ #


class TestFitIntegration:
    def _make_expression(self, stack, constants, integers=()):
        """Build an AGraphExpression from raw stack/constants."""
        from bingo.expressions.agraph.pyagraph.expression import AGraphExpression

        expr = AGraphExpression(simplification="reduce")
        expr._raw_command_array = stack.copy()
        expr._raw_constants = tuple(constants)
        expr._raw_integers = tuple(integers)
        expr._modified = True
        expr._is_fitted = False
        return expr

    def test_fit_recovers_linear_coefficient(self):
        """fit() should find C0 ≈ 3  for  f(x) = C0 * X0."""
        stack = np.array(
            [
                [CONSTANT, 0, 0],
                [VARIABLE, 0, 0],
                [MULTIPLICATION, 0, 1],
            ],
            dtype=np.uint8,
        )
        expr = self._make_expression(stack, (1.0,))

        X = np.linspace(0, 5, 50).reshape(-1, 1)
        y = (3.0 * X).ravel()

        expr.fit(X, y)
        np.testing.assert_almost_equal(expr.constants[0], 3.0, decimal=4)

    def test_fit_recovers_affine(self):
        """fit() should find C0 ≈ 2, C1 ≈ 5  for  f(x) = C1 + C0 * X0."""
        stack = np.array(
            [
                [CONSTANT, 0, 0],
                [CONSTANT, 1, 1],
                [VARIABLE, 0, 0],
                [MULTIPLICATION, 0, 2],
                [ADDITION, 1, 3],
            ],
            dtype=np.uint8,
        )
        expr = self._make_expression(stack, (1.0, 1.0))

        X = np.linspace(-3, 3, 100).reshape(-1, 1)
        y = (5.0 + 2.0 * X).ravel()

        expr.fit(X, y)
        np.testing.assert_almost_equal(expr.constants[0], 2.0, decimal=4)
        np.testing.assert_almost_equal(expr.constants[1], 5.0, decimal=4)

    def test_fit_no_constants_is_noop(self):
        """fit() returns self immediately when there are no constants."""
        stack = np.array(
            [
                [VARIABLE, 0, 0],
                [VARIABLE, 1, 1],
                [ADDITION, 0, 1],
            ],
            dtype=np.uint8,
        )
        expr = self._make_expression(stack, ())
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([3.0, 7.0])
        result = expr.fit(X, y)
        assert result is expr
