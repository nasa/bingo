"""
This module contains the basic structure for evolutionary optimization in
bingo.  The general framework allows access to an evolve_until_convergence
function.
"""
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from ..Util.ArgumentValidation import argument_validation


OptimizeResult = namedtuple('OptimizeResult', ['success', 'status', 'message',
                                               'ngen', 'fitness'])


class EvolutionaryOptimizer(metaclass=ABCMeta):
    """ Fundamental bingo object that coordinates evolutionary optimization

    Abstract base class for evolutionary optimization.  The primary role of
    this class is to house the evolve_until_convergence function. Classes which
    extend this one will have access to this function's capability.

    Attributes
    ----------
    generational_age: int
        The number of generations the optimizer has been evolved
    """
    def __init__(self):
        self.generational_age = 0
        self._starting_age = 0
        self._fitness_improvement_age = 0
        self._best_fitness = self.get_best_fitness()

    @argument_validation(max_generations={">=": 1},
                         min_generations={">=": 0},
                         convergence_check_frequency={">": 0})
    def evolve_until_convergence(self, max_generations,
                                 absolute_error_threshold,
                                 convergence_check_frequency=1,
                                 min_generations=0,
                                 stagnation_generations=None):
        """Evolution occurs until one of three convergence criteria is met

        Convergence criteria:
          * a maximum number of generations have been evolved
          * a fitness below an absolute threshold has been achieved
          * improvement upon best fitness has not happened for a set number of
            generations

        Parameters
        ----------
        max_generations: int
            The maximum number of generations the optimization will run.
        absolute_error_threshold: float
            The minimum fitness that must be achieved in order for the
            algorithm to converge.
        convergence_check_frequency: int, default 1
            The number of generations that will run between checking for
            convergence.
        min_generations: int, default 0
            The minimum number of generations the algorithm will run.
        stagnation_generations: int (optional)
            The number of generations after which evolution will stop if no
            improvement is seen.

        Returns
        --------
        OptimizeResult :
            Object containing information about the result of the run
        """
        self._starting_age = self.generational_age

        while self.generational_age - self._starting_age < min_generations:
            self.evolve(convergence_check_frequency)
            self._update_best_fitness()

        if self._convergence(absolute_error_threshold):
            return self._make_optim_result(0, absolute_error_threshold)
        if self._stagnation(stagnation_generations):
            return self._make_optim_result(1, stagnation_generations)

        while self.generational_age - self._starting_age < max_generations:
            self.evolve(convergence_check_frequency)
            self._update_best_fitness()

            if self._convergence(absolute_error_threshold):
                return self._make_optim_result(0, absolute_error_threshold)
            if self._stagnation(stagnation_generations):
                return self._make_optim_result(1, stagnation_generations)

        return self._make_optim_result(2, max_generations)

    def _update_best_fitness(self):
        last_best_fitness = self._best_fitness
        self._best_fitness = self.get_best_fitness()
        if self._best_fitness < last_best_fitness:
            self._fitness_improvement_age = self.generational_age

    def _convergence(self, threshold):
        return self._best_fitness <= threshold

    def _stagnation(self, threshold):
        if threshold is None:
            return False
        stagnation_time = self.generational_age - self._fitness_improvement_age
        return stagnation_time >= threshold

    def _make_optim_result(self, status, aux_info):
        ngen = self.generational_age - self._starting_age
        if status == 0:
            message = "Absolte convergence occurred with best fitness < " + \
                      "{}".format(aux_info)
            success = True
        elif status == 1:
            message = "Stagnation occurred with no improvement for more " + \
                      "than {} generations".format(aux_info)
            success = False
        else:  # status == 2:
            message = "The maximum number of generational steps " + \
                      "({}) occurred".format(aux_info)
            success = False

        return OptimizeResult(success, status, message, ngen,
                              self._best_fitness)

    @abstractmethod
    def evolve(self, num_generations):
        """The function responsible for generational evolution.

        Notes
        -----
        Implementation of this function should include increasing the
        generational_age

        Parameters
        ----------
        num_generations : int
            The number of generations to evolve
        """
        raise NotImplementedError

    @abstractmethod
    def get_best_fitness(self):
        """ Gets the value of the most fit individual

        Returns
        -------
         :
            Fitness of best individual
        """
        raise NotImplementedError

