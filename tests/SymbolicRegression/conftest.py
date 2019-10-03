# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest
import numpy as np

from bingo.symbolic_regression.equation import Equation
try:
    from bingocpp.build import bingocpp as bingocpp
    CppEquation = bingocpp.Equation
except ImportError:
    bingocpp = None
    CppEquation = object

class SumEquation(Equation):
    def evaluate_equation_at(self, x):
        return np.sum(x, axis=1).reshape((-1, 1))

    def evaluate_equation_with_x_gradient_at(self, x):
        x_sum = self.evaluate_equation_at(x)
        return x_sum, x

    def evaluate_equation_with_local_opt_gradient_at(self, x):
        pass

    def get_complexity(self):
        pass

    def get_latex_string(self):
        pass

    def get_console_string(self):
        pass

    def __str__(self):
        pass

    def distance(self, _chromosome):
        return 0

class SumEquationCpp(CppEquation):
    def evaluate_equation_at(self, x):
        return np.sum(x, axis=1).reshape((-1, 1))

    def evaluate_equation_with_x_gradient_at(self, x):
        x_sum = self.evaluate_equation_at(x)
        return x_sum, x

    def evaluate_equation_with_local_opt_gradient_at(self, x):
        pass

    def get_complexity(self):
        pass

    def get_latex_string(self):
        pass

    def get_console_string(self):
        pass

    def distance(self, _chromosome):
        return 0

@pytest.fixture()
def dummy_sum_equation():
    return SumEquation()

@pytest.fixture()
def dummy_sum_equation_cpp():
    return SumEquationCpp()
