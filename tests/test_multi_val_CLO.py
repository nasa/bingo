import pytest

from bingo.MultiValueContinuousLocalOptimization import MultiValueContinuousLocalOptimization as mvclo

def test_count_number_of_optimizable_params():
    indv = mvclo([i*1.0 for i in range(10)])
    