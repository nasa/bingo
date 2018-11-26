import pytest

from bingo.Util.ProbabilityMassFunction import ProbabilityMassFunction


@pytest.fixture
def all_true_pmf():
    return ProbabilityMassFunction(items=[True, False], weights=[1, 0])


@pytest.fixture
def all_false_pmf():
    return ProbabilityMassFunction(items=[True, False], weights=[0, 1])


@pytest.fixture
def empty_pmf():
    return ProbabilityMassFunction()


@pytest.fixture
def sample_pmf():
    return ProbabilityMassFunction(items=[1.0, 2.0, 3.0, 4.0],
                                   weights=[4.0, 3.0, 2.0, 1.0])


def test_raises_exception_for_uneven_init():
    with pytest.raises(ValueError):
        ProbabilityMassFunction(items=[1, 2, 3], weights=[1, ])


def test_raises_exception_for_non_listlike_init():
    with pytest.raises(ValueError):
        ProbabilityMassFunction(items="happy")


def test_raises_exception_for_non_numeric_weight():
    with pytest.raises(TypeError):
        ProbabilityMassFunction(items=[1, 2, 3], weights=[1, "a", 3])


def test_raises_exception_for_draw_from_empty_pmf(empty_pmf):
    with pytest.raises(ValueError):
        _ = empty_pmf.draw_sample()


@pytest.mark.parametrize("pmf,expected_value", [
    (all_true_pmf(), True),
    (all_false_pmf(), False),
])
def test_constant_pmfs(pmf, expected_value):
    for _ in range(10):
        assert pmf.draw_sample() == expected_value


@pytest.mark.parametrize("pmf,expected_value", [
    (all_true_pmf(), True),
    (all_false_pmf(), False),
])
def test_expected_values_pmfs(pmf, expected_value):
    for _ in range(10):
        assert pmf.draw_sample() == expected_value


@pytest.mark.parametrize("item", [
    True,
    "ABC",
    sum,
])
@pytest.mark.parametrize("weight", [
    None,
    1.0
])
def test_add_items_to_empty_pmf(empty_pmf, item, weight):
    empty_pmf.add_item(item, weight)
    assert empty_pmf.draw_sample() == item


def test_raises_exception_negative_weights(empty_pmf):
    empty_pmf.add_item("-", -1.0)
    with pytest.raises(ValueError):
        empty_pmf.add_item("+", 1.0)


def test_raises_exception_nan_weights(empty_pmf):
    with pytest.raises(ValueError):
        empty_pmf.add_item("a", 0.0)

