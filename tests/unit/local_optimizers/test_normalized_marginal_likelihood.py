import pytest
from bingo.local_optimizers.normalized_marginal_likelihood import (
    NormalizedMarginalLikelihood,
)


def test_nml_sets_up_smcpy_optimizer(mocker):
    objective_fn = mocker.MagicMock()
    deterministic_optimizer = mocker.Mock()
    smcpy_optimizer = mocker.patch(
        "bingo.local_optimizers.normalized_marginal_likelihood.SmcpyOptimizer",
        autospec=True,
    )

    nml = NormalizedMarginalLikelihood(
        objective_fn, deterministic_optimizer, log_scale=True, mcmc_steps=999
    )
    nml.training_data = 10
    nml.eval_count = 13

    smcpy_optimizer.assert_called_once_with(
        objective_fn, deterministic_optimizer, mcmc_steps=999
    )

    assert nml.optimizer.training_data == 10
    assert nml.optimizer.eval_count == 13
    assert nml.training_data == 10
    assert nml.eval_count == 13


@pytest.mark.parametrize(
    "log_scale,expected_nml", [(True, -1), (False, -2.718281828459045)]
)
def test_logscale(mocker, log_scale, expected_nml):
    individual = mocker.Mock()
    objective_fn = mocker.MagicMock()
    deterministic_optimizer = mocker.Mock()
    mocker.patch(
        "bingo.local_optimizers.normalized_marginal_likelihood.SmcpyOptimizer",
        autospec=True,
        return_value=mocker.Mock(return_value=(1, None, None)),
    )

    nml = NormalizedMarginalLikelihood(
        objective_fn, deterministic_optimizer, log_scale=log_scale
    )

    assert nml(individual) == expected_nml

