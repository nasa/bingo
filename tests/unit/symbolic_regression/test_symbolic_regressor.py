"""Public estimator tests for expression-backed symbolic regression."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.metrics import r2_score

import bingo.expressions.agraph as agraph
from bingo.expressions import EvolvableExpression
from bingo.symbolic_regression.symbolic_regressor import SymbolicRegressor


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


def test_exposes_expression_generation_and_regression_parameters():
    estimator = SymbolicRegressor(
        min_stack_size=3,
        max_stack_size=7,
        simplification="reduce",
        loss="mae",
        fit_tolerance=1e-7,
    )

    params = estimator.get_params()
    assert params["min_stack_size"] == 3
    assert params["max_stack_size"] == 7
    assert params["simplification"] == "reduce"
    assert params["loss"] == "mae"
    assert params["fit_tolerance"] == 1e-7
    assert {"stack_size", "use_simplification", "metric", "clo_alg", "clo_threshold"}.isdisjoint(params)


def test_fit_validates_inputs_tracks_features_and_predict_validates_features():
    estimator = SymbolicRegressor(population_size=4, generations=1, random_state=0)
    X = np.arange(8.0).reshape(-1, 1)
    y = 2.0 * X.ravel() + 1.0

    estimator.fit(X, y)

    assert estimator.n_features_in_ == 1
    with pytest.raises(ValueError, match="features"):
        estimator.predict(np.ones((2, 2)))
    with pytest.raises(ValueError):
        SymbolicRegressor().fit(X, y[:-1])


def test_predict_before_fit_raises_not_fitted_error():
    with pytest.raises(NotFittedError):
        SymbolicRegressor().predict(np.ones((2, 1)))


def test_evolves_expression_and_uses_sklearn_score_contract(backend):
    X = np.linspace(-2.0, 2.0, 12).reshape(-1, 1)
    y = 2.0 * X.ravel() + 1.0
    estimator = SymbolicRegressor(
        population_size=12,
        min_stack_size=3,
        max_stack_size=5,
        operators=["+", "*"],
        simplification="reduce",
        loss="mae",
        fit_tolerance=1e-7,
        generations=2,
        random_state=0,
    )

    estimator.fit(X, y)

    predictions = estimator.predict(X)
    assert isinstance(estimator.get_best_individual(), EvolvableExpression)
    assert predictions.shape == y.shape
    assert np.all(np.isfinite(predictions))
    assert estimator.get_best_individual().fitness == pytest.approx(
        estimator.get_best_individual().expression.loss(X, y, kind="mae")
    )
    assert estimator.score(X, y) == pytest.approx(r2_score(y, predictions))
