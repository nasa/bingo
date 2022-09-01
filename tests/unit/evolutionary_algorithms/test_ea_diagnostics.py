# Ignoring some linting rules in tests
# pylint: disable=missing-docstring

import pytest
import numpy as np
from bingo.chromosomes.chromosome import Chromosome
from bingo.evolutionary_algorithms.ea_diagnostics import (
    EaDiagnostics,
    EaDiagnosticsSummary,
    GeneticOperatorSummary,
)


@pytest.fixture
def dummy_chromosome(mocker):
    mocker.patch("bingo.chromosomes.chromosome.Chromosome", autospec=True)
    mocker.patch.object(Chromosome, "__abstractmethods__", new_callable=set)
    return Chromosome


@pytest.fixture
def population_12(dummy_chromosome):
    return [dummy_chromosome(fitness=i) for i in [1, 2]]


@pytest.fixture
def population_0123_times_4(dummy_chromosome):
    return [dummy_chromosome(fitness=i) for i in [0, 1, 2, 3] * 4]


@pytest.fixture
def cross_type_simple():
    return np.array(["c_n"] * 8 + [None] * 8, dtype=object)


@pytest.fixture
def mut_type_simple():
    return np.array([None] * 4 + ["m_n"] * 8 + [None] * 4, dtype=object)


@pytest.fixture
def cross_type_complex():
    return np.array(["c_n"] * 4 + ["c_s"] * 4 + [None] * 8, dtype=object)


@pytest.fixture
def mut_type_complex():
    return np.array(
        ["m_n"] * 2
        + ["m_s"] * 2
        + ["m_n"] * 2
        + [None] * 2
        + ["m_n"] * 2
        + [None] * 6,
        dtype=object,
    )


def test_correctly_updated_overall_summary(
    population_12, population_0123_times_4, cross_type_simple, mut_type_simple
):
    offspring_parent_idx = [[0, 1]] * 8 + [[0]] * 6 + [[]] * 2
    ead = EaDiagnostics(["c_n"], ["m_n"])
    ead.update(
        population_12,
        population_0123_times_4,
        offspring_parent_idx,
        cross_type_simple,
        mut_type_simple,
    )

    expected_summary = EaDiagnosticsSummary(
        beneficial_crossover_rate=0.25,
        detrimental_crossover_rate=0.25,
        beneficial_mutation_rate=0.25,
        detrimental_mutation_rate=0.5,
        beneficial_crossover_mutation_rate=0.25,
        detrimental_crossover_mutation_rate=0.25,
    )

    assert ead.summary == expected_summary


def test_correctly_updated_type_summaries(
    population_12,
    population_0123_times_4,
    cross_type_complex,
    mut_type_complex,
):
    offspring_parent_idx = [[0, 1]] * 16
    ead = EaDiagnostics(["c_n", "c_s"], ["m_n", "m_s"])
    ead.update(
        population_12,
        population_0123_times_4,
        offspring_parent_idx,
        cross_type_complex,
        mut_type_complex,
    )

    expected_summary = EaDiagnosticsSummary(
        beneficial_crossover_rate=0,
        detrimental_crossover_rate=0.5,
        beneficial_mutation_rate=0.5,
        detrimental_mutation_rate=0,
        beneficial_crossover_mutation_rate=1.0 / 3.0,
        detrimental_crossover_mutation_rate=1.0 / 6.0,
    )

    assert ead.summary == expected_summary

    expected_cross_summary = {
        "c_n": GeneticOperatorSummary(
            beneficial_rate=np.nan, detrimental_rate=np.nan
        ),
        "c_s": GeneticOperatorSummary(beneficial_rate=0, detrimental_rate=0.5),
    }

    # using np.testing.assert_equal to deal with nan
    assert ead.crossover_type_summary.keys() == expected_cross_summary.keys()
    for cross_type in expected_cross_summary.keys():
        np.testing.assert_array_equal(
            ead.crossover_type_summary[cross_type],
            expected_cross_summary[cross_type],
        )

    expected_mut_summary = {
        "m_n": GeneticOperatorSummary(beneficial_rate=0.5, detrimental_rate=0),
        "m_s": GeneticOperatorSummary(
            beneficial_rate=np.nan, detrimental_rate=np.nan
        ),
    }

    assert ead.mutation_type_summary.keys() == expected_mut_summary.keys()
    for mut_type in expected_mut_summary.keys():
        np.testing.assert_array_equal(
            ead.mutation_type_summary[mut_type], expected_mut_summary[mut_type]
        )

    expected_cross_mut_summary = {
        ("c_n", "m_n"): GeneticOperatorSummary(
            beneficial_rate=0.5, detrimental_rate=0
        ),
        ("c_n", "m_s"): GeneticOperatorSummary(
            beneficial_rate=0, detrimental_rate=0.5
        ),
        ("c_s", "m_n"): GeneticOperatorSummary(
            beneficial_rate=0.5, detrimental_rate=0
        ),
        ("c_s", "m_s"): GeneticOperatorSummary(
            beneficial_rate=np.nan, detrimental_rate=np.nan
        ),
    }

    assert (
        ead.crossover_mutation_type_summary.keys()
        == expected_cross_mut_summary.keys()
    )
    for pair in expected_cross_mut_summary.keys():
        np.testing.assert_array_equal(
            ead.crossover_mutation_type_summary[pair],
            expected_cross_mut_summary[pair],
        )


def test_correct_log_headers():
    ead = EaDiagnostics(["c_n", "c_s"], ["m_n", "m_s"])
    expected_header = (
        "crossover_number, crossover_beneficial, crossover_detrimental, "
        + "mutation_number, mutation_beneficial, mutation_detrimental, "
        + "crossover_mutation_number, crossover_mutation_beneficial, "
        + "crossover_mutation_detrimental, "
        + "c_n_number, c_n_beneficial, c_n_detrimental, "
        + "c_s_number, c_s_beneficial, c_s_detrimental, "
        + "m_n_number, m_n_beneficial, m_n_detrimental, "
        + "m_s_number, m_s_beneficial, m_s_detrimental, "
        + "c_n_m_n_number, c_n_m_n_beneficial, c_n_m_n_detrimental, "
        + "c_n_m_s_number, c_n_m_s_beneficial, c_n_m_s_detrimental, "
        + "c_s_m_n_number, c_s_m_n_beneficial, c_s_m_n_detrimental, "
        + "c_s_m_s_number, c_s_m_s_beneficial, c_s_m_s_detrimental"
    )
    assert expected_header == ead.get_log_header()


def test_correct_log_stats(
    population_12,
    population_0123_times_4,
    cross_type_complex,
    mut_type_complex,
):
    offspring_parent_idx = [[0, 1]] * 16
    ead = EaDiagnostics(["c_n", "c_s"], ["m_n", "m_s"])

    ead.update(
        population_12,
        population_0123_times_4,
        offspring_parent_idx,
        cross_type_complex,
        mut_type_complex,
    )

    expected_stats = (
        [2, 0, 1, 2, 1, 0, 6, 2, 1]
        + [0, 0, 0, 2, 0, 1]
        + [2, 1, 0, 0, 0, 0]
        + [2, 1, 0, 2, 0, 1, 2, 1, 0, 0, 0, 0]
    )

    np.testing.assert_array_equal(expected_stats, ead.get_log_stats())


def test_correctly_updated_existing_type_summaries(
    population_12,
    population_0123_times_4,
    cross_type_complex,
    mut_type_complex,
):
    offspring_parents = [[0, 1]] * 16
    ead = EaDiagnostics(["c_n", "c_s"], ["m_n", "m_s"])
    ead.update(
        population_12,
        population_0123_times_4,
        offspring_parents,
        cross_type_complex,
        mut_type_complex,
    )

    crossover_offspring_type_2 = np.array(
        ["c_s"] * 3 + ["c_n"] + ["c_s"] + [None] * 2 + ["c_s"] + [None] * 8,
        dtype=object,
    )
    mutation_offspring_type_2 = np.array(
        ["m_s"]
        + ["m_n"] * 2
        + ["m_s"]
        + [None] * 4
        + ["m_n"]
        + [None] * 2
        + ["m_s"]
        + [None] * 3
        + ["m_s"],
        dtype=object,
    )

    ead.update(
        population_12,
        population_0123_times_4,
        offspring_parents,
        crossover_offspring_type_2,
        mutation_offspring_type_2,
    )

    expected_summary = EaDiagnosticsSummary(
        beneficial_crossover_rate=0.25,
        detrimental_crossover_rate=0.5,
        beneficial_mutation_rate=0.4,
        detrimental_mutation_rate=0.4,
        beneficial_crossover_mutation_rate=0.3,
        detrimental_crossover_mutation_rate=0.2,
    )

    assert ead.summary == expected_summary

    expected_cross_summary = {
        "c_n": GeneticOperatorSummary(
            beneficial_rate=np.nan, detrimental_rate=np.nan
        ),
        "c_s": GeneticOperatorSummary(
            beneficial_rate=0.25, detrimental_rate=0.5
        ),
    }

    # using np.testing.assert_equal to deal with nan
    assert ead.crossover_type_summary.keys() == expected_cross_summary.keys()
    for cross_type in expected_cross_summary.keys():
        np.testing.assert_array_equal(
            ead.crossover_type_summary[cross_type],
            expected_cross_summary[cross_type],
        )

    expected_mut_summary = {
        "m_n": GeneticOperatorSummary(
            beneficial_rate=2.0 / 3.0, detrimental_rate=0
        ),
        "m_s": GeneticOperatorSummary(beneficial_rate=0, detrimental_rate=1),
    }

    assert ead.mutation_type_summary == expected_mut_summary

    expected_cross_mut_summary = {
        ("c_n", "m_n"): GeneticOperatorSummary(
            beneficial_rate=0.5, detrimental_rate=0
        ),
        ("c_n", "m_s"): GeneticOperatorSummary(
            beneficial_rate=0, detrimental_rate=2.0 / 3.0
        ),
        ("c_s", "m_n"): GeneticOperatorSummary(
            beneficial_rate=0.25, detrimental_rate=0
        ),
        ("c_s", "m_s"): GeneticOperatorSummary(
            beneficial_rate=1, detrimental_rate=0
        ),
    }

    assert ead.crossover_mutation_type_summary == expected_cross_mut_summary


@pytest.mark.parametrize("num_subsets", [1, 2, 4, 8])
def test_sum(
    population_12,
    population_0123_times_4,
    num_subsets,
    cross_type_complex,
    mut_type_complex,
):
    offspring_parent_idx = [[0, 1]] * 8 + [[0]] * 8
    crossover_types = set(cross_type_complex) - {None}
    mutation_types = set(mut_type_complex) - {None}

    num_subsets = 2
    ead_list = []
    for i in range(num_subsets):
        subset_inds = list(range(i, 16, num_subsets))
        offspring = [population_0123_times_4[i] for i in subset_inds]
        parents = [offspring_parent_idx[i] for i in subset_inds]
        cross_type = cross_type_complex[subset_inds]
        mut_type = mut_type_complex[subset_inds]
        ead = EaDiagnostics(crossover_types, mutation_types)
        ead.update(population_12, offspring, parents, cross_type, mut_type)
        ead_list.append(ead)

    expected_summary = EaDiagnosticsSummary(
        beneficial_crossover_rate=0,
        detrimental_crossover_rate=0.5,
        beneficial_mutation_rate=0.5,
        detrimental_mutation_rate=0,
        beneficial_crossover_mutation_rate=1.0 / 3.0,
        detrimental_crossover_mutation_rate=1.0 / 6.0,
    )

    expected_cross_summary = {
        "c_n": GeneticOperatorSummary(
            beneficial_rate=np.nan, detrimental_rate=np.nan
        ),
        "c_s": GeneticOperatorSummary(beneficial_rate=0, detrimental_rate=0.5),
    }

    expected_mut_summary = {
        "m_n": GeneticOperatorSummary(beneficial_rate=0.5, detrimental_rate=0),
        "m_s": GeneticOperatorSummary(
            beneficial_rate=np.nan, detrimental_rate=np.nan
        ),
    }

    expected_cross_mut_summary = {
        ("c_n", "m_n"): GeneticOperatorSummary(
            beneficial_rate=0.5, detrimental_rate=0
        ),
        ("c_n", "m_s"): GeneticOperatorSummary(
            beneficial_rate=0, detrimental_rate=0.5
        ),
        ("c_s", "m_n"): GeneticOperatorSummary(
            beneficial_rate=0.5, detrimental_rate=0
        ),
        ("c_s", "m_s"): GeneticOperatorSummary(
            beneficial_rate=np.nan, detrimental_rate=np.nan
        ),
    }

    summed_ead = sum(ead_list)

    assert summed_ead.summary == expected_summary

    # using np.testing.assert_equal to deal with nan
    assert (
        summed_ead.crossover_type_summary.keys()
        == expected_cross_summary.keys()
    )
    for cross_type in expected_cross_summary.keys():
        np.testing.assert_array_equal(
            summed_ead.crossover_type_summary[cross_type],
            expected_cross_summary[cross_type],
        )

    assert (
        summed_ead.mutation_type_summary.keys() == expected_mut_summary.keys()
    )
    for mut_type in expected_mut_summary.keys():
        np.testing.assert_array_equal(
            summed_ead.mutation_type_summary[mut_type],
            expected_mut_summary[mut_type],
        )

    assert (
        summed_ead.crossover_mutation_type_summary.keys()
        == expected_cross_mut_summary.keys()
    )
    for pair in expected_cross_mut_summary.keys():
        np.testing.assert_array_equal(
            summed_ead.crossover_mutation_type_summary[pair],
            expected_cross_mut_summary[pair],
        )

