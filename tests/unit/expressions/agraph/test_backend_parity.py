"""Backend-parametrized parity tests for AGraphExpression.

Runs the same fixed-fixture assertions against both the pure-Python backend
and the C++ backend.  Python cases always execute; C++ cases skip only when
the compiled extension is unavailable locally (CI builds it and requires the
cases to pass).
"""

import numpy as np
import pytest

import bingo.expressions.agraph as agraph
from bingo.expressions.agraph.pyagraph.operators import (
    VARIABLE,
    CONSTANT,
    ADDITION,
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


def test_gradient_contract(backend, simple_x):
    Expr = agraph.get_expression_class()
    expr = Expr(equation="X_0 + 10.0")
    f, df_dx = expr.gradient(simple_x)
    np.testing.assert_allclose(f, simple_x[:, 0] + 10.0)
    expected = np.zeros_like(simple_x)
    expected[:, 0] = 1.0
    np.testing.assert_allclose(df_dx, expected)


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
