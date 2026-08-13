# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
from bingo.chromosomes.multiple_floats import MultipleFloatChromosome,\
                                               MultipleFloatChromosomeGenerator

DUMMY_VALUE = 999


def test_generator():
    def dummy_function():
        return DUMMY_VALUE

    generator = MultipleFloatChromosomeGenerator(dummy_function,
                                                  values_per_chromosome=6)
    chromosome = generator()

    assert chromosome.values == [DUMMY_VALUE] * 6
