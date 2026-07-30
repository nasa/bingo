"""Scikit-learn interface for expression-backed symbolic regression."""

import os
import random

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_X_y, check_array

from bingo.evaluation.evaluation import Evaluation
from bingo.evolutionary_algorithms.age_fitness import AgeFitnessEA
from bingo.evolutionary_algorithms.generalized_crowding import GeneralizedCrowdingEA
from bingo.evolutionary_optimizers.island import Island
from bingo.expressions import (
    AGraphCrossover,
    AGraphGenerator,
    AGraphMutation,
    ComponentGenerator,
)
from bingo.stats.pareto_front import ParetoFront

from .explicit_regression import ExplicitRegression


DEFAULT_OPERATORS = ("+", "-", "*", "/")
SUPPORTED_EA_STRS = {
    "AgeFitnessEA": AgeFitnessEA,
    "GeneralizedCrowdingEA": GeneralizedCrowdingEA,
}


class SymbolicRegressor(RegressorMixin, BaseEstimator):
    """Evolve an Expression for explicit symbolic regression.

    Expressions own numerical fitting and loss evaluation. This estimator owns
    scikit-learn validation and configures their evolutionary search.
    """

    def __init__(
        self,
        *,
        population_size=500,
        min_stack_size=16,
        max_stack_size=32,
        operators=None,
        simplification="cas",
        crossover_prob=0.4,
        mutation_prob=0.4,
        loss="mse",
        generations=int(1e19),
        fitness_threshold=1.0e-16,
        max_time=1800,
        max_evals=int(1e19),
        evolutionary_algorithm=None,
        fit_tolerance=1.0e-5,
        random_state=None,
    ):
        self.population_size = population_size
        self.min_stack_size = min_stack_size
        self.max_stack_size = max_stack_size
        self.operators = DEFAULT_OPERATORS if operators is None else operators
        self.simplification = simplification
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.loss = loss
        self.generations = generations
        self.fitness_threshold = fitness_threshold
        self.max_time = max_time
        self.max_evals = max_evals
        self.evolutionary_algorithm = evolutionary_algorithm
        self.fit_tolerance = fit_tolerance
        self.random_state = random_state

    def _make_island(self, X, y, n_processes):
        component_generator = ComponentGenerator(
            X.shape[1], random_state=self.random_state
        )
        for operator in self.operators:
            component_generator.add_operator(operator)

        generator = AGraphGenerator(
            self.min_stack_size,
            self.max_stack_size,
            component_generator,
            simplification=self.simplification,
            random_state=self.random_state,
        )
        crossover = AGraphCrossover(
            self.min_stack_size, self.max_stack_size, random_state=self.random_state
        )
        mutation = AGraphMutation(component_generator, random_state=self.random_state)
        objective = ExplicitRegression(X, y, self.loss, self.fit_tolerance)
        evaluator = Evaluation(objective, multiprocess=n_processes)

        evolutionary_algorithm = self.evolutionary_algorithm
        if evolutionary_algorithm is None:
            evolutionary_algorithm = AgeFitnessEA
        elif evolutionary_algorithm in SUPPORTED_EA_STRS:
            evolutionary_algorithm = SUPPORTED_EA_STRS[evolutionary_algorithm]

        if evolutionary_algorithm is AgeFitnessEA:
            ea = evolutionary_algorithm(
                evaluator,
                generator,
                crossover,
                mutation,
                self.crossover_prob,
                self.mutation_prob,
                self.population_size,
            )
        elif evolutionary_algorithm is GeneralizedCrowdingEA:
            ea = evolutionary_algorithm(
                evaluator, crossover, mutation, self.crossover_prob, self.mutation_prob
            )
        else:
            raise TypeError(
                f"{evolutionary_algorithm} is an unsupported evolutionary algorithm"
            )

        hall_of_fame = ParetoFront(
            secondary_key=lambda individual: individual.complexity,
            similarity_function=lambda first, second: first == second,
        )
        return Island(ea, generator, self.population_size, hall_of_fame=hall_of_fame)

    def fit(self, X, y, sample_weight=None):
        """Fit an expression to numeric feature and target arrays."""
        if sample_weight is not None:
            raise NotImplementedError("sample_weight is not supported")

        X, y = check_X_y(X, y, ensure_2d=True, dtype=float, y_numeric=True)
        self.n_features_in_ = X.shape[1]
        if self.random_state is not None:
            np.random.seed(self.random_state)
            random.seed(self.random_state)

        n_processes = int(os.environ.get("OMP_NUM_THREADS", "0"))
        self.archipelago_ = self._make_island(X, y, n_processes)
        self.archipelago_.evolve_until_convergence(
            max_generations=self.generations,
            fitness_threshold=self.fitness_threshold,
            max_time=self.max_time,
            max_fitness_evaluations=self.max_evals,
            convergence_check_frequency=10,
        )
        self.archipelago_.update_hall_of_fame()
        self.best_population_ = list(self.archipelago_.hall_of_fame)
        if not self.best_population_:
            self.best_population_ = list(self.archipelago_.population)
        self.best_ind_ = min(
            self.best_population_, key=lambda individual: individual.fitness
        )
        return self

    def get_best_individual(self):
        """Return the fitted evolvable Expression with the lowest loss."""
        if not hasattr(self, "best_ind_"):
            raise ValueError("Best individual not set. Make sure fit() was called.")
        return self.best_ind_

    def get_best_population(self):
        """Return the fitted expression Pareto front."""
        if not hasattr(self, "best_population_"):
            raise ValueError("Best population not set. Make sure fit() was called.")
        return self.best_population_

    def get_pareto_front(self):
        """Return the fitted expression Pareto front."""
        return self.get_best_population()

    def predict(self, X):
        """Predict targets with the best fitted Expression."""
        if not hasattr(self, "best_ind_"):
            raise NotFittedError("This SymbolicRegressor instance is not fitted yet.")
        X = check_array(X, ensure_2d=True, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but SymbolicRegressor is expecting "
                f"{self.n_features_in_} features as input."
            )
        return self.best_ind_.predict(X)
