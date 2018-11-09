import copy
from abc import ABCMeta, abstractmethod


class GeneticIndividual(object, metaclass=ABCMeta):
    def __init__(self):
        self.genetic_age = 0
        self.fitness = None
        self.fit_set = False

    def copy(self):
        return copy.deepcopy(self)

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def needs_local_optimization(self):
        pass

    @abstractmethod
    def get_number_local_optimization_params(self):
        pass

    @abstractmethod
    def set_local_optimization_params(self, params):
        pass


class EquationIndividual(GeneticIndividual, metaclass=ABCMeta):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def evaluate_equation_at(self, x):
        pass

    @abstractmethod
    def evaluate_equation_derivative_at(self, x):
        pass

    @abstractmethod
    def evaluate_equation_with_local_optimization_gradient_at(self, x):
        pass

    @abstractmethod
    def get_latex_string(self):
        pass

    @abstractmethod
    def get_console_string(self):
        pass

    @abstractmethod
    def get_complexity(self):
        pass