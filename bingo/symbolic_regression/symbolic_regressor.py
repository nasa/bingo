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
from bingo.evolutionary_algorithms.generalized_crowding import (
    GeneralizedCrowdingEA,
)
from bingo.evolutionary_optimizers.fitness_predictor_island import (
    FitnessPredictorIsland,
)
from bingo.evolutionary_optimizers.island import Island

# from bingo.evolutionary_optimizers.parallel_archipelago import ParallelArchipelago
# from bingo.evolutionary_optimizers.serial_archipelago import SerialArchipelago
from bingo.local_optimizers.scipy_optimizer import ScipyOptimizer
from bingo.local_optimizers.local_opt_fitness import LocalOptFitnessFunction

# from bingo.stats.hall_of_fame import HallOfFame
from bingo.stats.pareto_front import ParetoFront
from .agraph.component_generator import ComponentGenerator
from .agraph.crossover import AGraphCrossover
from .agraph.mutation import AGraphMutation
from bingo.symbolic_regression import AGraphGenerator
from .explicit_regression import (
    ExplicitRegression,
    ExplicitTrainingData,
)  # this forces use of python fit funcs
from .equation_regressor import EquationRegressor

DEFAULT_OPERATORS = {"+", "-", "*", "/"}
SUPPORTED_EA_STRS = {
    "AgeFitnessEA": AgeFitnessEA,
    "GeneralizedCrowdingEA": GeneralizedCrowdingEA,
}
BEST_POP_MAX = 100
TIME_REDUCTION_FACTOR = 0.97


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

    def __init__(
        self,
        *,
        population_size=500,
        stack_size=32,
        operators=None,
        use_simplification=False,
        crossover_prob=0.4,
        mutation_prob=0.4,
        metric="mse",
        clo_alg="lm",
        generations=int(1e19),
        fitness_threshold=1.0e-16,
        max_time=1800,
        max_evals=int(1e19),
        evolutionary_algorithm=None,
        clo_threshold=1.0e-5,
        scale_max_evals=False,
        random_state=None,
    ):
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
        self.max_time = max_time * TIME_REDUCTION_FACTOR
        self.max_evals = max_evals
        self.scale_max_evals = scale_max_evals

        if evolutionary_algorithm is None:
            evolutionary_algorithm = AgeFitnessEA
        elif evolutionary_algorithm in SUPPORTED_EA_STRS:
            evolutionary_algorithm = SUPPORTED_EA_STRS[evolutionary_algorithm]
        self.evolutionary_algorithm = evolutionary_algorithm

        self.clo_threshold = clo_threshold

        # TODO make private attribute,
        #  as well as other attributes not defined in __init__?
        self.generator = None
        self.component_generator = None
        self.mutation = None
        self.crossover = None
        self.archipelago = None
        self.best_ind = None
        self.best_pop = None

        self.random_state = random_state

    def set_params(self, **params):
        # TODO not clean
        new_params = self.get_params()
        new_params.update(params)
        super().set_params(**new_params)
        self.__init__(**new_params)
        return self

    def _get_local_opt(self, X, y, tol):
        training_data = ExplicitTrainingData(X, y)
        fitness = ExplicitRegression(training_data=training_data, metric=self.metric)
        optimizer = ScipyOptimizer(fitness, method=self.clo_alg, tol=tol)
        local_opt_fitness = LocalOptFitnessFunction(fitness, optimizer)
        return local_opt_fitness

    def _make_island(self, dset_size, evo_alg, hof):
        if dset_size < 1200:
            return Island(
                evo_alg, self.generator, self.population_size, hall_of_fame=hof
            )
        return FitnessPredictorIsland(
            evo_alg,
            self.generator,
            self.population_size,
            hall_of_fame=hof,
            predictor_size_ratio=800 / dset_size,
        )

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

        self.generator = AGraphGenerator(
            self.stack_size,
            self.component_generator,
            use_simplification=self.use_simplification,
            use_python=True,
        )

        local_opt_fitness = self._get_local_opt(X, y, self.clo_threshold)
        evaluator = Evaluation(local_opt_fitness, multiprocess=n_processes)

        if self.evolutionary_algorithm == AgeFitnessEA:
            evo_alg = self.evolutionary_algorithm(
                evaluator,
                self.generator,
                self.crossover,
                self.mutation,
                self.crossover_prob,
                self.mutation_prob,
                self.population_size,
            )
        elif self.evolutionary_algorithm == GeneralizedCrowdingEA:
            evo_alg = self.evolutionary_algorithm(
                evaluator,
                self.crossover,
                self.mutation,
                self.crossover_prob,
                self.mutation_prob,
            )
        else:
            raise TypeError(
                f"{self.evolutionary_algorithm} is an unsupported "
                "evolutionary algorithm"
            )

        hof = ParetoFront(
            secondary_key=lambda ag: ag.get_complexity(),
            similarity_function=agraph_similarity,
        )

        island = self._make_island(len(X), evo_alg, hof)
        self._force_diversity_in_island(island)

        # if self.parallel:
        #     return ParallelArchipelago(island, hall_of_fame=hof)
        # else:

        return island

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
            Not currently supported

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
                max_eval_scaling = len(X) / self.archipelago._predictor_size / 1.1

        _ = self.archipelago.evolve_until_convergence(
            max_generations=self.generations,
            fitness_threshold=self.fitness_threshold,
            max_time=self.max_time,
            max_fitness_evaluations=self.max_evals * max_eval_scaling,
            convergence_check_frequency=10,
        )

        self.best_pop = self._find_best_population(X, y)
        self.best_ind = min(self.best_pop, key=lambda x: x.fitness)

        return self

    def _find_best_population(self, X, y, max_pop=BEST_POP_MAX):
        if len(self.archipelago.hall_of_fame) == 0:
            self.archipelago.update_hall_of_fame()

        best_equs = []
        if len(self.archipelago.hall_of_fame) <= max_pop:
            for equ in self.archipelago.hall_of_fame:
                best_equs.append(equ)

            # TODO: this could be improved by only taking unique equations
            additional_equs = self.archipelago.population
            if len(self.archipelago.population) > max_pop - len(best_equs):
                additional_equs = np.random.choice(
                    self.archipelago.population,
                    # TODO: this^ will have to change if/when archipelago changes from an island/fpi
                    max_pop - len(best_equs),
                    replace=False,
                )

            for equ in additional_equs:
                best_equs.append(equ)
        else:
            best_equs.append(self.archipelago.hall_of_fame[0])
            best_equs.append(self.archipelago.hall_of_fame[1])
            for equ in np.random.choice(
                self.archipelago.hall_of_fame[1:-1], max_pop - 2, replace=False
            ):
                best_equs.append(equ)

        best_regressors = []
        for equ in best_equs:
            reg = EquationRegressor(equ, metric=self.metric, algo=self.clo_alg)
            reg.fit(X, y)
            best_regressors.append(reg)

        return best_regressors

    def get_best_individual(self):
        """Gets the best model found from fit().

        Returns
        -------
        best_individual: `RegressorMixin`
            Model with the best fitness from fit().

        Raises
        ------
        ValueError
            If fit() has not been called yet
        """
        if self.best_ind is None:
            raise ValueError("Best individual not set. Make sure fit() was called.")
        return self.best_ind

    def get_best_population(self):
        """Gets best group of models from fit()

        Returns
        -------
        list of `RegressorMixin`
            Models from pareto front and final population from fit().

        Raises
        ------
        ValueError
            If fit() has not been called yet
        """
        if self.best_pop is None:
            raise ValueError("Best population not set. Make sure fit() was called.")
        return self.best_pop

    def get_pareto_front(self):
        """Gets best group of models from fit()

        Returns
        -------
        list of `RegressorMixin`
            Models with the best fitnesses and complexities from fit().

        Raises
        ------
        ValueError
            If fit() has not been called yet
        """
        if self.best_pop is None:
            raise ValueError("Pareto front not set. Make sure fit() was called.")
        hof = ParetoFront(
            secondary_key=lambda equ: equ.complexity,
            similarity_function=lambda x, y: x.fitness == y.fitness
            and x.complexity == y.complexity,
        )
        hof.update(self.best_pop)
        return list(hof)

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
        return best_ind.predict(X)


def agraph_similarity(ag_1, ag_2):
    """a similarity metric between agraphs"""
    return (
        ag_1.fitness == ag_2.fitness and ag_1.get_complexity() == ag_2.get_complexity()
    )
