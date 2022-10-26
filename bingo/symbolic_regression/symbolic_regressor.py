"""
This module contains the implementation of an object used for symbolic
regression via a scikit-learn interface.
"""
import os
import random

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from bingo.evaluation.evaluation import Evaluation
from bingo.evolutionary_algorithms.age_fitness import AgeFitnessEA
from bingo.evolutionary_algorithms.generalized_crowding import \
    GeneralizedCrowdingEA
from bingo.evolutionary_optimizers.fitness_predictor_island import \
    FitnessPredictorIsland
from bingo.evolutionary_optimizers.island import Island
from bingo.local_optimizers.scipy_optimizer import ScipyOptimizer
from bingo.local_optimizers.local_opt_fitness \
    import LocalOptFitnessFunction
from bingo.stats.hall_of_fame import HallOfFame
from bingo.symbolic_regression.agraph.component_generator import \
    ComponentGenerator
from bingo.symbolic_regression.agraph.crossover import AGraphCrossover
from bingo.symbolic_regression.agraph.mutation import AGraphMutation
from bingo.symbolic_regression import AGraphGenerator
from bingo.symbolic_regression.explicit_regression import ExplicitRegression, \
    ExplicitTrainingData  # this forces use of python fit funcs

DEFAULT_OPERATORS = {"+", "-", "*", "/"}
SUPPORTED_EA_STRS = ["AgeFitnessEA", "GeneralizedCrowdingEA"]
INF_REPLACEMENT = 1e100


# pylint: disable=too-many-instance-attributes, too-many-locals
class SymbolicRegressor(RegressorMixin, BaseEstimator):
    """Class for performing symbolic regression using genetic programming.

    Parameters
    ----------
    population_size: int, optional
        The number of individuals in a population.
    stack_size: int, optional
        The max number of commands per individual.
    operators: iterable of str, optional
        Potential operations that can be used.
    use_simplification: bool, optional
        Whether to use simplification to speed up evaluation or not.
    crossover_prob: float, optional
        Probability in [0, 1] of crossover occurring on an individual.
    mutation_prob: float, optional
        Probability in [0, 1] of mutation occurring on an individual.
    metric: str, optional
        Error metric to use for fitness (e.g., "rmse", "mse", "mae").
    clo_alg: str, optional
        Algorithm to use for local optimization (e.g., "lm", "BFGS", etc.).
    generations: int, optional
        Maximum number of generations allowed for evolution.
    fitness_threshold: float, optional
        Error/fitness threshold to stop evolution at.
    max_time: int, optional
        Number of seconds to stop evolution at.
    max_evals: int, optional
        Number of fitness evaluation to stop evolution at.
    evolutionary_algorithm: `EvolutionaryAlgorithm`, optional
        Evolutionary algorithm to use in evolution.
    clo_threshold: float, optional
        Threshold/tolerance for local optimization.
    scale_max_evals: bool, optional
        Whether to scale `max_evals` based on fitness predictors or not.
    random_state: int, optional
        Seed for random processes.
    """
    def __init__(self, *, population_size=500, stack_size=32,
                 operators=None, use_simplification=False,
                 crossover_prob=0.4, mutation_prob=0.4,
                 metric="mse", clo_alg="lm",
                 generations=int(1e19), fitness_threshold=1.0e-16,
                 max_time=1800, max_evals=int(1e19),
                 evolutionary_algorithm=None,
                 clo_threshold=1.0e-8, scale_max_evals=False,
                 random_state=None):
        self.population_size = population_size
        self.stack_size = stack_size

        if operators is None:
            operators = DEFAULT_OPERATORS
        self.operators = operators

        self.use_simplification = use_simplification

        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob

        self.metric = metric

        self.clo_alg = clo_alg

        self.generations = generations
        self.fitness_threshold = fitness_threshold
        self.max_time = max_time
        self.max_evals = max_evals
        self.scale_max_evals = scale_max_evals

        if evolutionary_algorithm is None:
            evolutionary_algorithm = AgeFitnessEA
        self.evolutionary_algorithm = evolutionary_algorithm

        self.clo_threshold = clo_threshold

        # TODO make private attribute,
        #  as well as other attributes not defined in __init__?
        self.best_ind = None

        self.random_state = random_state

    def _get_local_opt(self, X, y, tol):
        training_data = ExplicitTrainingData(X, y)
        fitness = ExplicitRegression(training_data=training_data,
                                     metric=self.metric)
        optimizer = ScipyOptimizer(fitness, method=self.clo_alg, tol=tol)
        local_opt_fitness = LocalOptFitnessFunction(fitness, optimizer)
        return local_opt_fitness

    def _make_island(self, dset_size, evo_alg, hof):
        if dset_size < 1200:
            return Island(evo_alg, self.generator, self.population_size,
                          hall_of_fame=hof)
        return FitnessPredictorIsland(evo_alg, self.generator,
                                      self.population_size, hall_of_fame=hof,
                                      predictor_size_ratio=800 / dset_size)

    def _force_diversity_in_island(self, island):
        diverse_pop = []
        pop_strings = set()

        i = 0
        while len(diverse_pop) < self.population_size:
            i += 1
            ind = self.generator()
            ind_str = str(ind)
            if ind_str not in pop_strings or i > 15 * self.population_size:
                pop_strings.add(ind_str)
                diverse_pop.append(ind)
        island.population = diverse_pop

    # pylint: disable=attribute-defined-outside-init
    def _get_archipelago(self, X, y, n_processes):
        self.component_generator = ComponentGenerator(X.shape[1])
        for operator in self.operators:
            self.component_generator.add_operator(operator)

        self.crossover = AGraphCrossover()
        self.mutation = AGraphMutation(self.component_generator)

        self.generator = \
            AGraphGenerator(self.stack_size, self.component_generator,
                            use_simplification=self.use_simplification,
                            use_python=True)

        local_opt_fitness = self._get_local_opt(X, y, self.clo_threshold)
        evaluator = Evaluation(local_opt_fitness, multiprocess=n_processes)

        if self.evolutionary_algorithm == AgeFitnessEA:
            evo_alg = self.evolutionary_algorithm(evaluator, self.generator,
                                                  self.crossover,
                                                  self.mutation,
                                                  self.crossover_prob,
                                                  self.mutation_prob,
                                                  self.population_size)
        elif self.evolutionary_algorithm == GeneralizedCrowdingEA:
            evo_alg = self.evolutionary_algorithm(evaluator, self.crossover,
                                                  self.mutation,
                                                  self.crossover_prob,
                                                  self.mutation_prob)
        else:
            raise TypeError(f"{self.evolutionary_algorithm} is an unsupported "
                            "evolutionary algorithm")

        hof = HallOfFame(10)

        island = self._make_island(len(X), evo_alg, hof)
        self._force_diversity_in_island(island)

        return island

    def _refit_best_individual(self, X, y, tol):
        fit_func = self._get_local_opt(X, y, tol)
        best_fitness = fit_func(self.best_ind)
        best_constants = tuple(self.best_ind.constants)
        for _ in range(5):
            self.best_ind._needs_opt = True
            fitness = fit_func(self.best_ind)
            if fitness < best_fitness:
                best_fitness = fitness
                best_constants = tuple(self.best_ind.constants)
        self.best_ind.fitness = best_fitness
        self.best_ind.set_local_optimization_params(best_constants)

    def fit(self, X, y, sample_weight=None):
        """Fit this model to the given data.

        Parameters
        ----------
        X: MxD numpy array of numeric
            Input values. D is the number of dimensions and
            M is the number of data points.
        y: Mx1 numpy array of numeric
            Target/output values. M is the number of data points.
        sample_weight: Mx1 numpy array of numeric, optional
            Weights per sample/data point. M is the number of data points.

        Returns
        -------
        self: `SymbolicRegressor`
            The fitted version of this object.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            random.seed(self.random_state)

        if sample_weight is not None:
            print("sample weight not None, TODO")
            raise NotImplementedError

        n_cpus = int(os.environ.get("OMP_NUM_THREADS", "0"))

        self.archipelago = self._get_archipelago(X, y, n_cpus)

        max_eval_scaling = 1
        if isinstance(self.archipelago, FitnessPredictorIsland):
            if self.scale_max_evals:
                max_eval_scaling = \
                    len(X) / self.archipelago._predictor_size / 1.1

        _ = self.archipelago.evolve_until_convergence(
            max_generations=self.generations,
            fitness_threshold=self.fitness_threshold,
            max_time=self.max_time,
            max_fitness_evaluations=self.max_evals * max_eval_scaling,
            convergence_check_frequency=10
        )

        # most likely found sol in 0 gens
        if len(self.archipelago.hall_of_fame) == 0:
            self.best_ind = self.archipelago.get_best_individual()
        else:
            self.best_ind = self.archipelago.hall_of_fame[0]

        self._refit_best_individual(X, y, tol=1e-6)

        return self

    def get_best_individual(self):
        """Gets the best model found from fit().

        Returns
        -------
        best_ind: `AGraph`
            Model with the best fitness from fit().
        """
        if self.best_ind is None:
            raise ValueError("Best individual not set")
        return self.best_ind

    def predict(self, X):
        """Use the best individual to predict the outputs of `X`.

        Parameters
        ----------
        X: MxD numpy array of numeric
            Input values. D is the number of dimensions and
            M is the number of data points.

        Returns
        -------
        pred_y: Mx1 numpy array of numeric
            Predicted target/output values. M is the number of data points.
        """
        best_ind = self.get_best_individual()
        output = best_ind.evaluate_equation_at(X)

        # convert nan to 0, inf to large number, and -inf to small number
        return np.nan_to_num(output, posinf=INF_REPLACEMENT,
                             neginf=-INF_REPLACEMENT)
