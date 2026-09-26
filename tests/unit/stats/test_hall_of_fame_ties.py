"""Keep the earliest equal-fitness individuals when the archive is full."""

from collections import namedtuple
from operator import attrgetter

import numpy as np
import pytest

from bingo.stats.hall_of_fame import HallOfFame


Individual = namedtuple("Individual", ["fitness", "gene", "score"])


@pytest.mark.parametrize("capacity", [1, 2, 5])
@pytest.mark.parametrize("batch_size", [1, 3, 10])
def test_full_archive_retains_earliest_equal_keys(capacity, batch_size):
    population = [Individual(1.0, i, 1.0) for i in range(10)]
    archive = HallOfFame(capacity)
    for start in range(0, len(population), batch_size):
        archive.update(population[start : start + batch_size])
    assert list(archive) == population[:capacity]
    assert len(archive) == capacity


@pytest.mark.parametrize("custom_key", [False, True])
def test_stream_matches_stable_best_n_reference(custom_key):
    key = attrgetter("score" if custom_key else "fitness")
    archive = HallOfFame(3, key_function=key)
    population = [
        Individual(value, i, 10 - value)
        for i, value in enumerate([5, 5, 3, 5, 2, 3, 2, 1, 1, 1, 1])
    ]
    seen = []
    for individual in population:
        seen.append(individual)
        archive.update([individual])
        assert list(archive) == sorted(seen, key=key)[:3]


def test_tied_cutoff_does_not_replace_an_existing_member():
    archive = HallOfFame(3)
    original = [
        Individual(0.0, "best", 0),
        Individual(1.0, "older", 0),
        Individual(1.0, "newer", 0),
    ]
    archive.update(original)
    stored = list(archive)
    archive.update([Individual(1.0, "latest", 0)])
    assert list(archive) == original
    assert all(actual is previous for actual, previous in zip(archive, stored))


def test_similarity_filter_and_strict_improvements_are_preserved():
    archive = HallOfFame(2, similarity_function=lambda a, b: a.gene == b.gene)
    archive.update([Individual(1.0, "first", 0), Individual(1.0, "second", 0)])
    archive.update([Individual(0.5, "first", 0), Individual(1.0, "third", 0)])
    assert [item.gene for item in archive] == ["first", "second"]
    archive.update([Individual(0.5, "better", 0)])
    assert [item.gene for item in archive] == ["better", "first"]


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), 0.0])
def test_nan_is_ignored_and_equal_boundary_keys_keep_the_first(value):
    archive = HallOfFame(1)
    first = Individual(value, "first", 0)
    archive.update(
        [Individual(np.nan, "invalid", 0), first, Individual(value, "later", 0)]
    )
    assert list(archive) == [first]


def test_unfilled_archive_keeps_ties_in_insertion_order():
    archive = HallOfFame(5)
    population = [Individual(1.0, i, 0) for i in range(3)]
    archive.update(population)
    assert list(archive) == population


def test_explicit_insert_still_accepts_ties_beyond_capacity():
    archive = HallOfFame(1)
    first, second = Individual(1.0, 0, 0), Individual(1.0, 1, 0)
    archive.insert(first)
    archive.insert(second)
    assert list(archive) == [first, second]


def test_archive_still_copies_new_members():
    archive = HallOfFame(1)
    individual = Individual(1.0, ["original"], 0)
    archive.update([individual])
    individual.gene.append("changed")
    assert archive[0].gene == ["original"]
