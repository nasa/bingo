"""Tests for AGraph utility helpers."""

import numpy as np
import pytest

from bingo.expressions.agraph import set_backend
from bingo.expressions.agraph.component_generator import ComponentGenerator
from bingo.expressions.agraph.generator import AGraphGenerator
from bingo.expressions.agraph.pyagraph.expression import AGraphExpression as PyAGraphExpression
from bingo.expressions.agraph.pyagraph.operators import (
    ADDITION,
    MULTIPLICATION,
    SIN,
    VARIABLE,
    CONSTANT,
)
from bingo.expressions.agraph.utils import pad_agraph_expression

try:
    from bingo.expressions.agraph.cppagraph import AGraphExpression as CppAGraphExpression

    CPP_AVAILABLE = True
except ImportError:
    CppAGraphExpression = None
    CPP_AVAILABLE = False


@pytest.fixture
def component_gen():
    gen = ComponentGenerator(input_x_dimension=3)
    gen.add_operator(ADDITION)
    gen.add_operator(MULTIPLICATION)
    gen.add_operator(SIN)
    return gen


@pytest.fixture(autouse=True)
def _restore_auto_backend():
    yield
    set_backend("auto")


class TestPadAGraphExpression:
    @staticmethod
    def _base_expr(expr_cls=PyAGraphExpression):
        expr = expr_cls(simplification="reduce")
        expr.raw_command_array = np.array(
            [
                [VARIABLE, 0, 0],
                [CONSTANT, 0, 0],
                [ADDITION, 0, 1],
            ],
            dtype=np.uint8,
        )
        expr.raw_constants = (3.5,)
        return expr

    def test_preserves_simplified_expression(self, component_gen):
        base_expr = self._base_expr()
        gen = AGraphGenerator(7, 7, component_gen, simplification="reduce", random_state=7)

        padded = pad_agraph_expression(base_expr, gen)

        np.testing.assert_array_equal(padded.command_array, base_expr.command_array)
        assert padded.constants == base_expr.constants
        assert str(padded) == str(base_expr)
        assert padded.raw_command_array.shape[0] == 7

    def test_intersperses_base_rows(self, component_gen):
        base_expr = self._base_expr()
        gen = AGraphGenerator(7, 7, component_gen, simplification="reduce", random_state=2)

        padded = pad_agraph_expression(base_expr, gen)
        raw = padded.raw_command_array

        utilized = list(padded.get_utilized_commands())
        utilized_indices = [i for i, is_used in enumerate(utilized) if is_used]
        assert len(utilized_indices) == base_expr.raw_command_array.shape[0]

        assert utilized_indices != list(range(len(utilized_indices)))
        assert utilized_indices != list(
            range(raw.shape[0] - len(utilized_indices), raw.shape[0])
        )

    def test_raises_if_generator_cannot_fit_base(self, component_gen):
        base_expr = self._base_expr()
        gen = AGraphGenerator(2, 2, component_gen, simplification="reduce", random_state=0)

        with pytest.raises(ValueError, match="generator.max_size"):
            pad_agraph_expression(base_expr, gen)

    def test_preserves_fitted_constants(self, component_gen):
        base_expr = PyAGraphExpression(equation="X0 * 1.0", simplification="reduce")
        x = np.array([[1.0], [2.0], [3.0]])
        y = 4.0 * x[:, 0]
        base_expr.fit(x, y)
        gen = AGraphGenerator(5, 5, component_gen, simplification="reduce", random_state=3)

        padded = pad_agraph_expression(base_expr, gen)

        assert padded.__sklearn_is_fitted__()
        assert padded.constants == pytest.approx(base_expr.constants)
        np.testing.assert_allclose(padded.predict(x), base_expr.predict(x))

    def test_returns_same_backend_type_for_python(self, component_gen):
        set_backend("python")
        base_expr = self._base_expr(PyAGraphExpression)
        gen = AGraphGenerator(6, 6, component_gen, simplification="reduce", random_state=5)

        padded = pad_agraph_expression(base_expr, gen)

        assert type(padded) is PyAGraphExpression

    @pytest.mark.skipif(not CPP_AVAILABLE, reason="cppagraph not built")
    def test_returns_same_backend_type_for_cpp(self, component_gen):
        set_backend("cpp")
        base_expr = self._base_expr(CppAGraphExpression)
        gen = AGraphGenerator(6, 6, component_gen, simplification="reduce", random_state=5)

        padded = pad_agraph_expression(base_expr, gen)

        assert type(padded) is CppAGraphExpression
