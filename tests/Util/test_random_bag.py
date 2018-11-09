import sys
import pytest
import numpy as np
sys.path.append("../..")

from bingo.Util.RandomBag import RandomBag

@pytest.fixture
def all_true_bag():
    return RandomBag(items=[True, False], weights=[1, 0])


@pytest.fixture
def all_false_bag():
    return RandomBag(items=[True, False], weights=[0, 1])


@pytest.fixture
def empty_bag():
    return RandomBag()


@pytest.fixture
def sample_bag():
    return RandomBag(items=[1.0, 2.0, 3.0, 4.0],
                     weights=[4.0, 3.0, 2.0, 1.0])


def test_raises_exception_for_uneven_init():
    with pytest.raises(ValueError):
        RandomBag(items=[1, 2, 3], weights=[1, ])


def test_raises_exception_for_non_listlike_init():
    with pytest.raises(ValueError):
        RandomBag(items="happy")


def test_raises_exception_for_non_numeric_weight():
    with pytest.raises(ValueError):
        RandomBag(items=[1, 2, 3], weights=[1, "a", 3])


def test_raises_exception_for_draw_from_empty_bag(empty_bag):
    with pytest.raises(ValueError):
        _ = empty_bag.draw_item()


@pytest.mark.parametrize("bag,expected_value", [
    (all_true_bag(), True),
    (all_false_bag(), False),
])
def test_constant_bags(bag, expected_value):
    for _ in range(10):
        assert bag.draw_item() == expected_value


@pytest.mark.parametrize("bag,expected_value", [
    (all_true_bag(), True),
    (all_false_bag(), False),
])
def test_expected_values_bags(bag, expected_value):
    for _ in range(10):
        assert bag.draw_item() == expected_value
