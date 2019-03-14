import numpy as np
import scipy.optimize as optimize

from ..Base.Evaluation import Evaluation
from ..Base.FitnessEvaluator import FitnessEvaluator

class FloatLocalOptimizationFunction(FitnessEvaluator):

    def __init__(self, fitness_function, algorithm='Nelder-Mead'):
        self._fitness_function = fitness_function
        self._algorithm = algorithm

    def __call__(self, individual):
        if individual.needs_local_optimization:
            self._optimize_params(individual)
        return self._evaluate_fitness(individual)

    def _optimize_params(self, individual):
        num_params = individual.get_number_local_optimization_params()
        c_0 = np.random.uniform(0, 10000, num_params)
        params = self._run_algorithm_for_optimization(
            self._sub_routine_for_fit_function, individual, c_0)
        print("done optimizing")
        individual.set_local_optimization_params(params)

    def _sub_routine_for_fit_function(self, params, individual):
        individual.set_local_optimization_params(params)
        return self._fitness_function(individual)

    def _run_algorithm_for_optimization(self, sub_routine, individual, params):
        return optimize.minimize(sub_routine, params, args=(individual), method=self._algorithm, tol=1e0)

    def _evaluate_fitness(self, individual):
        return self._fitness_function(individual)
