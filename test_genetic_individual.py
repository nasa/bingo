# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest
import numpy as np

from bingo.GeneticIndividual import GeneticIndividual


class InvalidChild(GeneticIndividual):
    def __str__(self):
        super().__str__()

    def needs_local_optimization(self):
        super().needs_local_optimization()

    def get_number_local_optimization_params(self):
        super().get_number_local_optimization_params()

    def set_local_optimization_params(self, params=None):
        super().set_local_optimization_params(params)


@pytest.fixture
def bad_gi():
    return InvalidChild()


def test_raises_error_construct_genetic_individual():
    with pytest.raises(TypeError):
        _ = GeneticIndividual()


@pytest.mark.parametrize("func", ["__str__",
                                  "needs_local_optimization",
                                  "get_number_local_optimization_params",
                                  "set_local_optimization_params"])
def test_raises_error_using_super_on_derived_classes(bad_gi, func):
    with pytest.raises(NotImplementedError):
        getattr(bad_gi, func)()
