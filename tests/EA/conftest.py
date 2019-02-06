# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest

from SingleValue import SingleValueChromosome

@pytest.fixture
def single_value_population_of_4():
    return [SingleValueChromosome(),
            SingleValueChromosome(),
            SingleValueChromosome(),
            SingleValueChromosome()]

@pytest.fixture
def single_value_population_of_100():
    return [SingleValueChromosome() for _ in range(100)]
