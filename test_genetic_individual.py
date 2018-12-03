# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest
import numpy as np

from bingo.GeneticIndividual import GeneticIndividual, EquationIndividual


class InvalidGeneticIndividualChild(GeneticIndividual):
    def __str__(self):
        super().__str__()

    def needs_local_optimization(self):
        super().needs_local_optimization()

    def get_number_local_optimization_params(self):
        super().get_number_local_optimization_params()

    def set_local_optimization_params(self, params):
        super().set_local_optimization_params(params)


class InvalidEquationIndividualChild(EquationIndividual,
                                     InvalidGeneticIndividualChild):
    def evaluate_equation_at(self, x):
        super().evaluate_equation_at(x)

    def evaluate_equation_with_x_gradient_at(self, x):
        super().evaluate_equation_with_x_gradient_at(x)

    def evaluate_equation_with_local_opt_gradient_at(self, x):
        super().evaluate_equation_with_local_opt_gradient_at(x)

    def get_latex_string(self):
        super().get_latex_string()

    def get_console_string(self):
        super().get_console_string()

    def get_complexity(self):
        super().get_complexity()


@pytest.fixture
def bad_gi():
    return InvalidGeneticIndividualChild()


@pytest.fixture
def bad_ei():
    return InvalidEquationIndividualChild()


def test_raises_error_construct_genetic_individual():
    with pytest.raises(TypeError):
        _ = GeneticIndividual()


def test_raises_error_using_super_on_gi_derived_classes(bad_gi):
    with pytest.raises(NotImplementedError):
        str(bad_gi)
    with pytest.raises(NotImplementedError):
        bad_gi.needs_local_optimization()
    with pytest.raises(NotImplementedError):
        bad_gi.get_number_local_optimization_params()
    with pytest.raises(NotImplementedError):
        bad_gi.set_local_optimization_params([1.0])


def test_raises_error_using_super_on_ei_derived_classes(bad_ei):
    x = np.ones((1, 1))
    with pytest.raises(NotImplementedError):
        bad_ei.evaluate_equation_at(x)
    with pytest.raises(NotImplementedError):
        bad_ei.evaluate_equation_with_x_gradient_at(x)
    with pytest.raises(NotImplementedError):
        bad_ei.evaluate_equation_with_local_opt_gradient_at(x)
    with pytest.raises(NotImplementedError):
        bad_ei.get_latex_string()
    with pytest.raises(NotImplementedError):
        bad_ei.get_console_string()
    with pytest.raises(NotImplementedError):
        bad_ei.get_complexity()
