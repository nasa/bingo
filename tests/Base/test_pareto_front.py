# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest
from collections import namedtuple

from bingo.Base.ParetoFront import ParetoFront

DummyIndv = namedtuple('DummyIndv', ['fitness', 'gene', 'att1', 'att2'])


@pytest.fixture
def empty_pf():
    similar = lambda indv1, indv2 : indv1.gene == indv2.gene
    key_2 = lambda indv : indv.att1
    return ParetoFront(similarity_function=similar, secondary_key=key_2)


@pytest.fixture
def full_pf():
    similar = lambda indv1, indv2 : indv1.gene == indv2.gene
    key_2 = lambda indv : indv.att1
    hof = ParetoFront(similarity_function=similar, secondary_key=key_2)
    for i in range(5):
        hof.insert(DummyIndv(i, i, 4 - i, 4 - i))
    return hof


@pytest.fixture(params=["empty", "full"])
def all_pfs(request, empty_pf, full_pf):
    if request.param == "empty":
        return empty_pf
    return full_pf


@pytest.mark.parametrize("pop, new_len",
                        [([DummyIndv(-1, -1, 5, 3)], 6),
                         ([DummyIndv(-1, 0, 5, 3)], 5),
                         ([DummyIndv(0, -1, 5, 3)], 5),
                         ([DummyIndv(0, -1, 0, 3)], 1),
                         ([DummyIndv(-1, -1, 5, 3),
                           DummyIndv(-1, -2, -1, 3)], 1),
                         ])
def test_update_adds_indvs_properly(full_pf, pop, new_len):
    full_pf.update(pop)
    assert len(full_pf) == new_len



