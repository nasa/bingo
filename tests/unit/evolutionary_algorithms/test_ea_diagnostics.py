# Ignoring some linting rules in tests
# pylint: disable=missing-docstring

import pytest
import numpy as np
from bingo.chromosomes.chromosome import Chromosome
from bingo.evolutionary_algorithms.ea_diagnostics \
    import EaDiagnostics, EaDiagnosticsSummary


@pytest.fixture
def dummy_chromosome(mocker):
    mocker.patch('bingo.chromosomes.chromosome.Chromosome', autospec=True)
    mocker.patch.object(Chromosome, "__abstractmethods__",
                        new_callable=set)
    return Chromosome


@pytest.fixture
def population_12(dummy_chromosome):
    return [dummy_chromosome(fitness=i) for i in [1, 2]]


@pytest.fixture
def population_0123_times_4(dummy_chromosome):
    return [dummy_chromosome(fitness=i) for i in [0, 1, 2, 3] * 4]


def test_correctly_updated_overall_summary(population_12,
                                           population_0123_times_4):
    offspring_parents = [[0, 1]] * 8 + [[0]] * 6 + [[]] * 2
    offspring_crossover_type = \
        np.array(["normal_c"] * 8 + [None] * 8, dtype=object)
    offspring_mutation_type = \
        np.array([None] * 4 + ["normal_m"] * 8 + [None] * 4, dtype=object)

    ead = EaDiagnostics()
    ead.update(population_12, population_0123_times_4, offspring_parents,
               offspring_crossover_type, offspring_mutation_type)

    expected_summary = EaDiagnosticsSummary(
        beneficial_crossover_rate=0.25,
        detrimental_crossover_rate=0.25,
        beneficial_mutation_rate=0.25,
        detrimental_mutation_rate=0.5,
        beneficial_crossover_mutation_rate=0.25,
        detrimental_crossover_mutation_rate=0.25)

    assert ead.summary == expected_summary


def test_correctly_updated_type_summaries(population_12,
                                         population_0123_times_4):
    offspring_parents = [[0, 1]] * 16
    offspring_crossover_type = \
        np.array(["c_n"] * 2 + ["c_s"] * 4 + ["c_n"] * 4 + [None] * 6,
                 dtype=object)
    offspring_mutation_type = \
        np.array([None] * 4 + ["m_n", "m_s"] * 5 + [None] * 2, dtype=object)

    ead = EaDiagnostics()
    ead.update(population_12, population_0123_times_4, offspring_parents,
               offspring_crossover_type, offspring_mutation_type)

    expected_cross_summary = {"c_n": (0.5, 0),
                              "c_s": (0, 0.5)}

    assert ead.crossover_type_summary == expected_cross_summary

    expected_mut_summary = {"m_n": (0.5, 0),
                            "m_s": (0, 0.5)}

    assert ead.mutation_type_summary == expected_mut_summary

    expected_cross_mut_summary = {"c_n": {"m_n": (0.5, 0),
                                          "m_s": (0, 0.5)},
                                  "c_S": {"m_n": (1, 0),
                                          "m_s": (0, 0)}}

    assert ead.crossover_mutation_summary == expected_cross_mut_summary


@pytest.mark.parametrize("num_subsets", [1, 2, 4, 8])
@pytest.mark.parametrize("crossover_type, mutation_type",
                         [("normal_c", 1), ("normal_m", 1)])
def test_sum(population_12, population_0123_times_4, num_subsets,
             crossover_type, mutation_type):
    offspring_parents = [[0, 1]] * 8 + [[0]] * 8
    offspring_crossover_type = \
        np.array([crossover_type] * 8 + [None] * 8, dtype=object)
    offspring_mutation_type = \
        np.array([None] * 4 + [mutation_type] * 8 + [None] * 4, dtype=object)

    num_subsets = 2
    ead_list = []
    for i in range(num_subsets):
        subset_inds = list(range(i, 16, num_subsets))
        offspring = [population_0123_times_4[i] for i in subset_inds]
        parents = [offspring_parents[i] for i in subset_inds]
        cross_type = offspring_crossover_type[subset_inds]
        mut_type = offspring_mutation_type[subset_inds]
        ead = EaDiagnostics()
        ead.update(population_12, offspring, parents, cross_type, mut_type)
        ead_list.append(ead)

    expected_summary = EaDiagnosticsSummary(
            beneficial_crossover_rate=0.25,
            detrimental_crossover_rate=0.25,
            beneficial_mutation_rate=0.25,
            detrimental_mutation_rate=0.5,
            beneficial_crossover_mutation_rate=0.25,
            detrimental_crossover_mutation_rate=0.25)

    assert sum(ead_list).summary == expected_summary
