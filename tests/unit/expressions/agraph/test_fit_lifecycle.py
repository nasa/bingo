import copy
import pickle

import numpy as np
import pytest

from bingo.expressions.agraph.pyagraph import AGraphExpression as PyAGraphExpression

try:
    from bingo.expressions.agraph.cppagraph import AGraphExpression as CppAGraphExpression
except ImportError:
    EXPRESSION_TYPES = [PyAGraphExpression]
else:
    EXPRESSION_TYPES = [PyAGraphExpression, CppAGraphExpression]


@pytest.mark.parametrize("expression_type", EXPRESSION_TYPES)
def test_commit_fit_validates_atomically_and_clear_fit_resets(expression_type):
    expression = expression_type(equation="X0 + 1.0")

    with pytest.raises(ValueError, match="one entry per simplified expression constant"):
        expression.commit_fit([])
    assert expression.constants == (1.0,)
    assert not expression.is_fitted

    with pytest.raises(ValueError, match="finite"):
        expression.commit_fit([np.nan])
    assert expression.constants == (1.0,)
    assert not expression.is_fitted

    assert expression.commit_fit([2.0]) is expression
    assert expression.constants == (2.0,)
    assert expression.is_fitted

    assert expression.clear_fit() is expression
    assert not expression.is_fitted


@pytest.mark.parametrize("expression_type", EXPRESSION_TYPES)
def test_fit_commit_lifecycle_survives_copy_and_pickle(expression_type):
    expression = expression_type(equation="X0 + 1.0")
    expression.commit_fit([2.0])

    copied = copy.deepcopy(expression)
    restored = pickle.loads(pickle.dumps(expression))

    for clone in (copied, restored):
        assert clone.constants == (2.0,)
        assert clone.is_fitted

    expression.raw_constants = (3.0,)
    assert not expression.is_fitted


@pytest.mark.parametrize("expression_type", EXPRESSION_TYPES)
def test_clear_fit_is_a_noop_for_constant_free_expressions(expression_type):
    expression = expression_type(equation="X0")

    assert expression.is_fitted
    assert expression.clear_fit() is expression
    assert expression.is_fitted
