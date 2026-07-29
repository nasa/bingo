import numpy as np
import pytest

import bingo.expressions.agraph as agraph
from bingo.expressions.agraph.pyagraph.operators import VARIABLE, CONSTANT, ADDITION


def _set_backend_or_skip(name: str):
    try:
        agraph.set_backend(name)
    except ImportError:
        pytest.skip("C++ backend unavailable locally")


@pytest.fixture(params=["python", "cpp"], scope="module")
def backend(request):
    if request.param == "cpp":
        _set_backend_or_skip("cpp")
    else:
        agraph.set_backend("python")
    yield request.param
    # restore auto after module
    agraph.set_backend("auto")


@pytest.fixture
def simple_x():
    return np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


def test_predictions_match_python_contract(backend, simple_x):
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


def test_loss_score_vocab_and_values(backend, simple_x):
    Expr = agraph.get_expression_class()
    expr = Expr(equation="X0 + 10.0")
    y = simple_x[:, 0] + 10.0
    # losses
    assert expr.loss(simple_x, y, kind="mse") == pytest.approx(0.0, abs=1e-10)
    assert expr.loss(simple_x, y, kind="mae") == pytest.approx(0.0, abs=1e-10)
    assert expr.loss(simple_x, y, kind="rmse") == pytest.approx(0.0, abs=1e-10)
    # score
    assert expr.score(simple_x, y) == pytest.approx(1.0)


def test_fit_keyword_only_tolerance(backend, simple_x):
    Expr = agraph.get_expression_class()
    expr = Expr(equation="X0 * 1.0")
    y = 3.0 * simple_x[:, 0]
    # kw-only enforced
    with pytest.raises(TypeError):
        expr.fit(simple_x, y, 1e-6)
    expr.fit(simple_x, y, tolerance=1e-6)


def test_is_fitted_lifecycle(backend, simple_x):
    Expr = agraph.get_expression_class()
    expr = Expr(equation="X0 * 1.0")
    assert not expr.__sklearn_is_fitted__()
    expr.fit(simple_x, 2.0 * simple_x[:, 0])
    assert expr.__sklearn_is_fitted__()
    # raw mutation clears fitted
    cmd = expr.mutable_raw_command_array
    cmd[:] = np.array([[VARIABLE, 0, 0], [CONSTANT, 0, 0], [ADDITION, 0, 1]], dtype=np.uint8)
    # Some backends expose property only via sklearn hook; use that
    assert not expr.__sklearn_is_fitted__()


def test_nonfinite_normalization(backend):
    Expr = agraph.get_expression_class()
    expr = Expr(equation="1.0 / X0")
    x = np.array([[0.0]])
    assert expr.loss(x, [1.0]) == float("inf")
    assert expr.score(x, [1.0]) == float("-inf")
