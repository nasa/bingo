# pylint: disable=missing-docstring
import builtins
import multiprocessing

import numpy as np
import pytest

from bingo.expressions.agraph.pyagraph.expression import AGraphExpression
from bingo.expressions.agraph.pyagraph.operators import CONSTANT
from bingo.symbolic_regression import (
    EvidenceResult,
    FitResult,
    ObjectiveData,
    ResidualMeasure,
    SmcEvidenceEstimator,
    explicit_residuals,
)
from bingo.symbolic_regression.evidence import smc_nmll_loss

try:
    from bingo.expressions.agraph.cppagraph import AGraphExpression as CppAGraphExpression
except ImportError:
    CppAGraphExpression = None


def _constant_expression(value=0.0):
    expression = AGraphExpression()
    expression.raw_command_array = np.array([[CONSTANT, 0, 0]], dtype=np.uint8)
    expression.raw_constants = (value,)
    return expression


def _estimate_constant_evidence(_):
    result = SmcEvidenceEstimator(num_particles=20, mcmc_steps=2, seed=21).estimate(
        _constant_expression(1.0),
        ObjectiveData(np.arange(5.0).reshape(-1, 1), np.full(5, 2.0)),
        explicit_residuals(),
    )
    return result.smc_nmll, result.map_constants


def test_estimate_returns_unsuccessful_result_when_proposal_cannot_be_built():
    expression = _constant_expression(1.0)
    estimator = SmcEvidenceEstimator()

    result = estimator.estimate(
        expression,
        ObjectiveData(np.arange(3.0), np.full(3, 2.0)),
        lambda *_: np.full(3, np.inf),
    )

    assert isinstance(result, EvidenceResult)
    assert not result.success
    assert result.smc_nmll == -np.inf
    assert result.map_constants is None
    assert expression.constants == (1.0,)
    assert not expression.is_fitted


def test_estimate_installs_only_posterior_map_constants_after_sampling():
    expression = _constant_expression(1.0)
    estimator = SmcEvidenceEstimator(num_particles=20, mcmc_steps=2, seed=4)

    result = estimator.estimate(
        expression,
        ObjectiveData(np.arange(5.0).reshape(-1, 1), np.full(5, 2.0)),
        explicit_residuals(),
    )

    assert result.success
    assert np.isfinite(result.smc_nmll)
    assert result.map_constants == expression.constants
    assert expression.is_fitted
    assert result.posterior is None


def test_failed_estimation_maps_to_infinite_ranking_loss():
    result = SmcEvidenceEstimator().estimate(
        _constant_expression(),
        ObjectiveData(np.arange(3.0), np.full(3, 2.0)),
        lambda *_: np.full(3, np.inf),
    )

    assert smc_nmll_loss(result) == np.inf


def test_estimate_retries_invalid_proposal_components():
    expression = _constant_expression(1.0)
    calls = 0

    def fitter(*_):
        nonlocal calls
        calls += 1
        return FitResult([np.nan] if calls == 1 else [2.0], success=True)

    result = SmcEvidenceEstimator(
        num_particles=20, mcmc_steps=2, seed=4, fitter=fitter
    ).estimate(
        expression,
        ObjectiveData(np.arange(5.0).reshape(-1, 1), np.full(5, 2.0)),
        explicit_residuals(),
    )

    assert result.success
    assert calls == 2


def test_estimate_regularizes_indefinite_laplace_curvature():
    measure = ResidualMeasure(
        lambda *_: np.ones(5),
        jacobian=lambda *_: np.zeros((5, 1)),
        residual_hessian=lambda *_: -np.ones((5, 1, 1)),
    )
    result = SmcEvidenceEstimator(
        num_particles=20,
        mcmc_steps=2,
        seed=4,
        fitter=lambda *_: FitResult([2.0], success=True),
    ).estimate(
        _constant_expression(1.0),
        ObjectiveData(np.arange(5.0).reshape(-1, 1), np.full(5, 2.0)),
        measure,
    )

    assert result.success


def test_missing_smcpy_returns_unsuccessful_result(mocker):
    real_import = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name == "smcpy" or name.startswith("smcpy."):
            raise ImportError("SMCPy unavailable")
        return real_import(name, *args, **kwargs)

    mocker.patch("builtins.__import__", side_effect=unavailable)
    result = SmcEvidenceEstimator().estimate(
        _constant_expression(),
        ObjectiveData(np.arange(3.0), np.full(3, 2.0)),
        explicit_residuals(),
    )

    assert not result.success
    assert result.smc_nmll == -np.inf


def test_fixed_seed_reproduces_evidence_and_can_return_posterior():
    data = ObjectiveData(np.arange(5.0).reshape(-1, 1), np.full(5, 2.0))
    first = _constant_expression(1.0)
    second = _constant_expression(8.0)
    estimator = SmcEvidenceEstimator(
        num_particles=20,
        mcmc_steps=2,
        seed=11,
        return_posterior=True,
        fitter=lambda *_: FitResult([2.0], success=True),
    )

    first_result = estimator.estimate(first, data, explicit_residuals())
    second_result = estimator.estimate(second, data, explicit_residuals())

    assert first_result.success
    assert second_result.success
    assert first_result.smc_nmll == second_result.smc_nmll
    assert first_result.map_constants == second_result.map_constants
    assert first_result.posterior is not None
    assert not hasattr(first, "posterior")


@pytest.mark.skipif(CppAGraphExpression is None, reason="C++ expression unavailable")
def test_fixed_seed_is_backend_neutral():
    data = ObjectiveData(np.arange(5.0).reshape(-1, 1), np.full(5, 2.0))
    python_result = SmcEvidenceEstimator(num_particles=20, mcmc_steps=2, seed=8).estimate(
        _constant_expression(1.0), data, explicit_residuals()
    )
    cpp_result = SmcEvidenceEstimator(num_particles=20, mcmc_steps=2, seed=8).estimate(
        CppAGraphExpression(equation="1.0"), data, explicit_residuals()
    )

    assert python_result.success
    assert cpp_result.success
    assert python_result.smc_nmll == cpp_result.smc_nmll
    assert python_result.map_constants == cpp_result.map_constants


def test_fixed_seed_reproduces_evidence_in_multiprocessing():
    with multiprocessing.get_context("spawn").Pool(2) as pool:
        results = pool.map(_estimate_constant_evidence, range(2))

    assert results[0] == results[1]


def test_none_seed_permits_nonreproducible_sampling():
    data = ObjectiveData(np.arange(5.0).reshape(-1, 1), np.full(5, 2.0))
    first = SmcEvidenceEstimator(num_particles=20, mcmc_steps=2, seed=None).estimate(
        _constant_expression(1.0), data, explicit_residuals()
    )
    second = SmcEvidenceEstimator(num_particles=20, mcmc_steps=2, seed=None).estimate(
        _constant_expression(1.0), data, explicit_residuals()
    )

    assert (first.smc_nmll, first.map_constants) != (second.smc_nmll, second.map_constants)
