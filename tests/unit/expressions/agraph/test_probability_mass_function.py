"""Tests for the local ProbabilityMassFunction."""

import numpy as np
import pytest

from bingo.expressions.agraph.probability_mass_function import ProbabilityMassFunction


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestInit:
    def test_empty(self, rng):
        pmf = ProbabilityMassFunction(rng)
        assert pmf.items == []

    def test_with_items_and_weights(self, rng):
        pmf = ProbabilityMassFunction(rng, items=["a", "b"], weights=[1, 3])
        assert pmf.items == ["a", "b"]

    def test_mismatched_lengths_raises(self, rng):
        with pytest.raises(ValueError, match="same length"):
            ProbabilityMassFunction(rng, items=["a", "b"], weights=[1])

    def test_negative_weight_raises(self, rng):
        with pytest.raises(ValueError, match="non-negative"):
            ProbabilityMassFunction(rng, items=["a"], weights=[-1])


class TestDrawSample:
    def test_single_item_always_drawn(self, rng):
        pmf = ProbabilityMassFunction(rng, items=["only"], weights=[1.0])
        for _ in range(20):
            assert pmf.draw_sample() == "only"

    def test_respects_weights(self):
        rng = np.random.default_rng(0)
        pmf = ProbabilityMassFunction(
            rng, items=["rare", "common"], weights=[0.01, 0.99]
        )
        counts = {"rare": 0, "common": 0}
        for _ in range(1000):
            counts[pmf.draw_sample()] += 1
        assert counts["common"] > counts["rare"]

    def test_reproducible(self):
        pmf1 = ProbabilityMassFunction(
            np.random.default_rng(7), items=[1, 2, 3], weights=[1, 1, 1]
        )
        pmf2 = ProbabilityMassFunction(
            np.random.default_rng(7), items=[1, 2, 3], weights=[1, 1, 1]
        )
        results1 = [pmf1.draw_sample() for _ in range(50)]
        results2 = [pmf2.draw_sample() for _ in range(50)]
        assert results1 == results2


class TestAddItem:
    def test_add_increases_items(self, rng):
        pmf = ProbabilityMassFunction(rng, items=["a"], weights=[1.0])
        pmf.add_item("b", 1.0)
        assert len(pmf.items) == 2
        assert "b" in pmf.items

    def test_added_item_can_be_drawn(self, rng):
        pmf = ProbabilityMassFunction(rng, items=["a"], weights=[0.0001])
        pmf.add_item("b", 100.0)
        drawn = {pmf.draw_sample() for _ in range(100)}
        assert "b" in drawn

    def test_add_to_empty(self, rng):
        pmf = ProbabilityMassFunction(rng)
        pmf.add_item("x", 1.0)
        assert pmf.draw_sample() == "x"

    def test_default_weight(self, rng):
        pmf = ProbabilityMassFunction(rng, items=["a"], weights=[1.0])
        pmf.add_item("b")  # default weight
        assert len(pmf.items) == 2
