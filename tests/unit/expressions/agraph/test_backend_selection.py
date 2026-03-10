"""Tests for the backend selection API (set_backend / get_backend)."""

import pytest

import bingo.expressions.agraph as agraph_pkg
from bingo.expressions.agraph import (
    set_backend,
    get_backend,
    get_expression_class,
)
from bingo.expressions.agraph.pyagraph import (
    AGraphExpression as PyAGraphExpression,
)
from bingo.expressions.agraph.component_generator import ComponentGenerator
from bingo.expressions.agraph.generator import AGraphGenerator
from bingo.expressions.agraph.evolvable import EvolvableExpression
from bingo.expressions.agraph.pyagraph.operators import ADDITION, SIN

try:
    from bingo.expressions.agraph.cppagraph import (
        AGraphExpression as CppAGraphExpression,
    )

    CPP_AVAILABLE = True
except ImportError:
    CppAGraphExpression = None
    CPP_AVAILABLE = False


@pytest.fixture(autouse=True)
def _restore_auto_backend():
    """Reset backend to 'auto' after every test."""
    yield
    set_backend("auto")


# ------------------------------------------------------------------ #
#  set_backend / get_backend basics                                    #
# ------------------------------------------------------------------ #


class TestSetBackend:
    def test_default_is_auto(self):
        set_backend("auto")
        assert get_backend() == "auto"

    def test_set_python(self):
        set_backend("python")
        assert get_backend() == "python"

    @pytest.mark.skipif(not CPP_AVAILABLE, reason="cppagraph not built")
    def test_set_cpp(self):
        set_backend("cpp")
        assert get_backend() == "cpp"

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            set_backend("fortran")

    @pytest.mark.skipif(CPP_AVAILABLE, reason="cppagraph IS available")
    def test_cpp_unavailable_raises(self):
        with pytest.raises(ImportError, match="not available"):
            set_backend("cpp")


# ------------------------------------------------------------------ #
#  get_expression_class                                                #
# ------------------------------------------------------------------ #


class TestGetExpressionClass:
    def test_python_returns_pyagraph(self):
        set_backend("python")
        assert get_expression_class() is PyAGraphExpression

    @pytest.mark.skipif(not CPP_AVAILABLE, reason="cppagraph not built")
    def test_cpp_returns_cppagraph(self):
        set_backend("cpp")
        assert get_expression_class() is CppAGraphExpression

    @pytest.mark.skipif(not CPP_AVAILABLE, reason="cppagraph not built")
    def test_auto_prefers_cpp(self):
        set_backend("auto")
        assert get_expression_class() is CppAGraphExpression

    @pytest.mark.skipif(CPP_AVAILABLE, reason="cppagraph IS available")
    def test_auto_falls_back_to_python(self):
        set_backend("auto")
        assert get_expression_class() is PyAGraphExpression


# ------------------------------------------------------------------ #
#  Module-level AGraphExpression binding                               #
# ------------------------------------------------------------------ #


class TestModuleLevelBinding:
    def test_python_sets_module_attribute(self):
        set_backend("python")
        assert agraph_pkg.AGraphExpression is PyAGraphExpression

    @pytest.mark.skipif(not CPP_AVAILABLE, reason="cppagraph not built")
    def test_cpp_sets_module_attribute(self):
        set_backend("cpp")
        assert agraph_pkg.AGraphExpression is CppAGraphExpression

    def test_switch_back_to_auto(self):
        set_backend("python")
        set_backend("auto")
        cls = agraph_pkg.AGraphExpression
        if CPP_AVAILABLE:
            assert cls is CppAGraphExpression
        else:
            assert cls is PyAGraphExpression


# ------------------------------------------------------------------ #
#  Generator respects backend switch                                   #
# ------------------------------------------------------------------ #


@pytest.fixture
def component_gen():
    gen = ComponentGenerator(input_x_dimension=2)
    gen.add_operator(ADDITION)
    gen.add_operator(SIN)
    return gen


class TestGeneratorBackend:
    def test_generator_uses_python_backend(self, component_gen):
        set_backend("python")
        gen = AGraphGenerator(3, 6, component_gen, random_state=0)
        indv = gen()
        assert isinstance(indv, EvolvableExpression)
        assert type(indv.expression) is PyAGraphExpression

    @pytest.mark.skipif(not CPP_AVAILABLE, reason="cppagraph not built")
    def test_generator_uses_cpp_backend(self, component_gen):
        set_backend("cpp")
        gen = AGraphGenerator(3, 6, component_gen, random_state=0)
        indv = gen()
        assert isinstance(indv, EvolvableExpression)
        assert type(indv.expression) is CppAGraphExpression

    @pytest.mark.skipif(not CPP_AVAILABLE, reason="cppagraph not built")
    def test_generator_follows_switch(self, component_gen):
        """A single generator instance follows backend changes."""
        gen = AGraphGenerator(3, 6, component_gen, random_state=0)

        set_backend("python")
        py_indv = gen()
        assert type(py_indv.expression) is PyAGraphExpression

        set_backend("cpp")
        cpp_indv = gen()
        assert type(cpp_indv.expression) is CppAGraphExpression
