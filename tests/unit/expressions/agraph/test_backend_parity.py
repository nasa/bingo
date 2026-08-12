"""Backend-parametrized parity tests for AGraphExpression.

Runs the same fixed-fixture assertions against both the pure-Python backend
and the C++ backend.  Python cases always execute; C++ cases skip only when
the compiled extension is unavailable locally (CI builds it and requires the
cases to pass).
"""

import warnings

import numpy as np
import pytest

import bingo.expressions.agraph as agraph
from bingo.expressions.agraph.evolvable import EvolvableExpression
from bingo.expressions.agraph.pyagraph.operators import (
    VARIABLE,
    CONSTANT,
    ADDITION,
    MULTIPLICATION,
    DIVISION,
    POWER,
    LOGARITHM,
    SQUARE,
)


def _set_backend_or_skip(name):
    try:
        agraph.set_backend(name)
    except ImportError:
        pytest.skip("C++ backend unavailable locally")


@pytest.fixture(params=["python", "cpp"])
def backend(request):
    if request.param == "cpp":
        _set_backend_or_skip("cpp")
    else:
        agraph.set_backend("python")
    yield request.param
    agraph.set_backend("auto")


@pytest.fixture
def simple_x():
    return np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


def test_predictions_match_contract(backend, simple_x):
    Expr = agraph.get_expression_class()
    expr = Expr(equation="X_0 + 10.0")
    preds = expr.predict(simple_x)
    np.testing.assert_allclose(preds, simple_x[:, 0] + 10.0)


def test_temporary_constant_prediction_contract(backend, simple_x):
    Expr = agraph.get_expression_class()
    expr = Expr(simplification="reduce")
    expr.raw_command_array = np.array(
        [
            [VARIABLE, 0, 0],
            [CONSTANT, 0, 0],
            [MULTIPLICATION, 0, 1],
            [CONSTANT, 1, 0],
            [ADDITION, 2, 3],
        ],
        dtype=np.uint8,
    )
    expr.raw_constants = (2.0, 3.0)
    original_constants = expr.constants
    original_raw_constants = expr.raw_constants
    original_fittedness = expr.is_fitted

    constants = np.array([[4.0, 5.0], [10.0, 20.0]])
    predictions = expr.predict(simple_x, constants=constants)

    expected = simple_x[:, :1] * constants[0] + constants[1]
    assert predictions.shape == (len(simple_x), 2)
    np.testing.assert_allclose(predictions, expected)
    assert expr.constants == original_constants
    assert expr.raw_constants == original_raw_constants
    assert expr.is_fitted == original_fittedness


def test_temporary_constant_prediction_preserves_input_rank(backend, simple_x):
    Expr = agraph.get_expression_class()
    expr = Expr(equation="X_0 + 10.0")

    vector_result = expr.predict(simple_x, constants=np.array([4.0]))
    matrix_result = expr.predict(simple_x, constants=np.array([[4.0]]))

    assert vector_result.shape == (len(simple_x),)
    assert matrix_result.shape == (len(simple_x), 1)
    np.testing.assert_allclose(vector_result, matrix_result[:, 0])


def test_empty_constant_batch_contract(backend, simple_x):
    Expr = agraph.get_expression_class()
    expr = Expr(equation="X_0 + 10.0")

    predictions = expr.predict(simple_x, constants=np.empty((1, 0)))

    assert predictions.shape == (len(simple_x), 0)


def test_empty_sample_batch_contract(backend, simple_x):
    Expr = agraph.get_expression_class()
    expr = Expr(equation="X_0 + 10.0")

    predictions = expr.predict(
        simple_x[:0], constants=np.array([[2.0, 3.0]])
    )

    assert predictions.shape == (0, 2)


def test_batched_prediction_accepts_numeric_array_like(backend, simple_x):
    Expr = agraph.get_expression_class()
    expr = Expr(equation="X_0 + 10.0")

    predictions = expr.predict(simple_x, constants=[[2.0, 3.0]])

    np.testing.assert_allclose(
        predictions, simple_x[:, :1] + np.array([[2.0, 3.0]])
    )


def test_constant_free_batched_prediction_contract(backend, simple_x):
    Expr = agraph.get_expression_class()
    expr = Expr(equation="X_0")

    predictions = expr.predict(simple_x, constants=np.empty((0, 3)))

    assert predictions.shape == (len(simple_x), 3)
    np.testing.assert_allclose(predictions, np.repeat(simple_x[:, :1], 3, axis=1))


def test_constant_only_batched_prediction_contract(backend, simple_x):
    Expr = agraph.get_expression_class()
    expr = Expr(equation="10.0")

    predictions = expr.predict(simple_x, constants=np.array([[2.0, 3.0]]))

    np.testing.assert_allclose(predictions, [[2.0, 3.0]] * len(simple_x))


def test_nonfinite_batched_prediction_is_column_local(backend, simple_x):
    Expr = agraph.get_expression_class()
    expr = Expr(simplification="reduce")
    expr.raw_command_array = np.array(
        [
            [VARIABLE, 0, 0],
            [CONSTANT, 0, 0],
            [DIVISION, 0, 1],
        ],
        dtype=np.uint8,
    )
    expr.raw_constants = (1.0,)

    with warnings.catch_warnings(record=True) as caught_warnings:
        predictions = expr.predict(
            simple_x, constants=np.array([[1.0, 0.0, 2.0]])
        )

    assert not caught_warnings
    np.testing.assert_allclose(predictions[:, 0], simple_x[:, 0])
    assert np.isinf(predictions[:, 1]).all()
    np.testing.assert_allclose(predictions[:, 2], simple_x[:, 0] / 2.0)


def test_nonfinite_temporary_constants_propagate_by_column(backend, simple_x):
    Expr = agraph.get_expression_class()
    expr = Expr(equation="X_0 + 10.0")

    predictions = expr.predict(
        simple_x, constants=np.array([[np.nan, np.inf, -np.inf]])
    )

    assert np.isnan(predictions[:, 0]).all()
    assert np.isposinf(predictions[:, 1]).all()
    assert np.isneginf(predictions[:, 2]).all()


def test_temporary_constants_are_keyword_only(backend, simple_x):
    Expr = agraph.get_expression_class()
    expr = Expr(equation="X_0 + 10.0")

    with pytest.raises(TypeError):
        expr.predict(simple_x, np.array([2.0]))


def test_evolvable_expression_forwards_batched_constants(backend, simple_x):
    Expr = agraph.get_expression_class()
    individual = EvolvableExpression(Expr(equation="X_0 + 10.0"))

    predictions = individual.predict(
        simple_x, constants=np.array([[2.0, 3.0]])
    )

    np.testing.assert_allclose(
        predictions, simple_x[:, :1] + np.array([[2.0, 3.0]])
    )


@pytest.mark.parametrize(
    "constants",
    [np.array(1.0), np.zeros((1, 1, 1)), np.zeros(2), np.zeros((2, 3))],
    ids=["scalar", "rank-3", "wrong-vector-size", "wrong-matrix-rows"],
)
def test_temporary_constant_prediction_validates_shape(
    backend, simple_x, constants
):
    Expr = agraph.get_expression_class()
    expr = Expr(equation="X_0 + 10.0")

    with pytest.raises(ValueError):
        expr.predict(simple_x, constants=constants)


def test_gradient_contract(backend, simple_x):
    Expr = agraph.get_expression_class()
    expr = Expr(equation="X_0 + 10.0")
    f, df_dx = expr.gradient(simple_x)
    np.testing.assert_allclose(f, simple_x[:, 0] + 10.0)
    expected = np.zeros_like(simple_x)
    expected[:, 0] = 1.0
    np.testing.assert_allclose(df_dx, expected)


def test_const_hessian_contract(backend, simple_x):
    # f(C0, C1) = C0 * C1 + C0**2
    Expr = agraph.get_expression_class()
    expr = Expr(simplification="reduce")
    expr.raw_command_array = np.array(
        [
            [CONSTANT, 0, 0],
            [CONSTANT, 1, 0],
            [MULTIPLICATION, 0, 1],
            [SQUARE, 0, 0],
            [ADDITION, 2, 3],
        ],
        dtype=np.uint8,
    )
    expr.raw_constants = (2.0, 3.0)

    value, gradient, hessian = expr._evaluate_with_const_hessian(simple_x)
    np.testing.assert_allclose(value, 10.0)
    np.testing.assert_allclose(gradient, [[7.0, 2.0]] * len(simple_x))
    np.testing.assert_allclose(
        hessian, [np.array([[2.0, 1.0], [1.0, 0.0]])] * len(simple_x)
    )


@pytest.mark.parametrize(
    "stack, constants, value, gradient, hessian",
    [
        (
            [[CONSTANT, 0, 0], [LOGARITHM, 0, 0]],
            (0.0,),
            -np.inf,
            np.inf,
            np.nan,
        ),
        (
            [[CONSTANT, 0, 0], [CONSTANT, 1, 0], [POWER, 0, 1]],
            (0.0, 2.0),
            0.0,
            np.nan,
            np.nan,
        ),
    ],
)
def test_const_hessian_nonfinite_contract(
    backend, simple_x, stack, constants, value, gradient, hessian
):
    Expr = agraph.get_expression_class()
    expr = Expr(simplification="reduce")
    expr.raw_command_array = np.array(stack, dtype=np.uint8)
    expr.raw_constants = constants

    actual_value, actual_gradient, actual_hessian = expr._evaluate_with_const_hessian(
        simple_x
    )
    np.testing.assert_allclose(actual_value, value)
    np.testing.assert_allclose(actual_gradient, gradient)
    np.testing.assert_allclose(actual_hessian, hessian)


def test_loss_vocabulary_and_values(backend, simple_x):
    Expr = agraph.get_expression_class()
    expr = Expr(equation="X_0 + 10.0")
    y = simple_x[:, 0] + 10.0
    assert expr.loss(simple_x, y, kind="mse") == pytest.approx(0.0, abs=1e-10)
    assert expr.loss(simple_x, y, kind="mae") == pytest.approx(0.0, abs=1e-10)
    assert expr.loss(simple_x, y, kind="rmse") == pytest.approx(0.0, abs=1e-10)


def test_score_vocabulary_and_values(backend, simple_x):
    Expr = agraph.get_expression_class()
    expr = Expr(equation="X_0 + 10.0")
    y = simple_x[:, 0] + 10.0
    assert expr.score(simple_x, y) == pytest.approx(1.0)  # r2, perfect fit
    # ``score`` rejects loss-only kinds.
    with pytest.raises(ValueError):
        expr.score(simple_x, y, kind="mse")


def test_explicit_methods_reject_mismatched_samples(backend, simple_x):
    Expr = agraph.get_expression_class()
    expr = Expr(equation="X_0 + 1.0")
    y = np.ones(simple_x.shape[0] - 1)

    for method in (expr.fit, expr.loss, expr.score):
        with pytest.raises(ValueError, match="same number of samples"):
            method(simple_x, y)


def test_implicit_methods_reject_mismatched_shapes(backend, simple_x):
    Expr = agraph.get_expression_class()
    expr = Expr(equation="X_0 + X_1")
    dx_dt = np.ones((simple_x.shape[0] - 1, simple_x.shape[1]))

    for method in (expr.fit_implicit, expr.implicit_loss, expr.implicit_score):
        with pytest.raises(ValueError, match="same shape"):
            method(simple_x, dx_dt)


def test_fit_tolerance_is_keyword_only(backend, simple_x):
    Expr = agraph.get_expression_class()
    expr = Expr(equation="X_0 * 1.0")
    y = 3.0 * simple_x[:, 0]
    with pytest.raises(TypeError):
        expr.fit(simple_x, y, 1e-6)
    expr.fit(simple_x, y, tolerance=1e-6)
    assert expr.constants[0] == pytest.approx(3.0, abs=0.1)


def test_is_fitted_lifecycle(backend, simple_x):
    Expr = agraph.get_expression_class()
    expr = Expr(equation="X_0 * 1.0")
    assert not expr.is_fitted
    assert not expr.__sklearn_is_fitted__()
    expr.fit(simple_x, 2.0 * simple_x[:, 0])
    assert expr.is_fitted
    assert expr.__sklearn_is_fitted__()
    # A raw structural change unsets the fitted state.
    cmd = expr.mutable_raw_command_array
    cmd[:] = np.array(
        [[VARIABLE, 0, 0], [CONSTANT, 0, 0], [ADDITION, 0, 1]], dtype=np.uint8
    )
    assert not expr.is_fitted


def test_implicit_loss_and_score(backend, simple_x):
    Expr = agraph.get_expression_class()
    expr = Expr(equation="X_0 + X_1")
    dx_dt = np.ones_like(simple_x)
    loss = expr.implicit_loss(simple_x, dx_dt)
    score = expr.implicit_score(simple_x, dx_dt)
    assert np.isfinite(loss)
    assert score == pytest.approx(-loss)


def test_nonfinite_normalization(backend):
    Expr = agraph.get_expression_class()
    expr = Expr(equation="1.0 / X_0")
    x = np.array([[0.0]])
    y = np.array([1.0])
    assert expr.loss(x, y, kind="mse") == float("inf")
    assert expr.score(x, y) == float("-inf")
