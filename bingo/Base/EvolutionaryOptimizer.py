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
    def __init__(self, hall_of_fame=None):
        self.generational_age = 0
        self._starting_age = 0
        self._fitness_improvement_age = 0
        self._best_fitness = None
        self._hall_of_fame = hall_of_fame

    @argument_validation(max_generations={">=": 1},
                         min_generations={">=": 0},
                         convergence_check_frequency={">": 0})
    def evolve_until_convergence(self, max_generations,
                                 fitness_threshold,
                                 convergence_check_frequency=1,
                                 min_generations=0,
                                 stagnation_generations=None,
                                 max_fitness_evaluations=None):
        """Evolution occurs until one of four convergence criteria is met

        Convergence criteria:
          * a maximum number of generations have been evolved
          * a fitness below an absolute threshold has been achieved
          * improvement upon best fitness has not happened for a set number of
            generations
          * the maximum number of fitness function evaluations has been reached

        Parameters
        ----------
        max_generations: int
            The maximum number of generations the optimization will run.
        fitness_threshold: float
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
        max_fitness_evaluations: int (optional)
            The maximum number of fitness function evaluations (approx) the
            optimizer will run.

        Returns
        --------
        OptimizeResult :
            Object containing information about the result of the run
        """
        self._starting_age = self.generational_age
        self._update_best_fitness()

        while self.generational_age - self._starting_age < min_generations:
            self.evolve(convergence_check_frequency)
            self._update_best_fitness()

        if self._convergence(fitness_threshold):
            return self._make_optim_result(0, fitness_threshold)
        if self._stagnation(stagnation_generations):
            return self._make_optim_result(1, stagnation_generations)
        if self._hit_max_evals(max_fitness_evaluations):
            return self._make_optim_result(3, max_fitness_evaluations)

        while self.generational_age - self._starting_age < max_generations:
            self.evolve(convergence_check_frequency)
            self._update_best_fitness()

            if self._convergence(fitness_threshold):
                return self._make_optim_result(0, fitness_threshold)
            if self._stagnation(stagnation_generations):
                return self._make_optim_result(1, stagnation_generations)
            if self._hit_max_evals(max_fitness_evaluations):
                return self._make_optim_result(3, max_fitness_evaluations)

        return self._make_optim_result(2, max_generations)

    def _update_best_fitness(self):
        last_best_fitness = self._best_fitness
        self._best_fitness = self.get_best_fitness()
        if last_best_fitness is None or self._best_fitness < last_best_fitness:
            self._fitness_improvement_age = self.generational_age

    def _convergence(self, threshold):
        return self._best_fitness <= threshold

    def _stagnation(self, threshold):
        if threshold is None:
            return False
        stagnation_time = self.generational_age - self._fitness_improvement_age
        return stagnation_time >= threshold

    def _hit_max_evals(self, threshold):
        if threshold is None:
            return False
        return self.get_fitness_evaluation_count() >= threshold

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
        elif status == 2:
            message = "The maximum number of generational steps " + \
                      "({}) occurred".format(aux_info)
            success = False
        else:  # status == 3:
            message = "The maximum number of fitness evaluations " + \
                      "({}) was exceeded. Total fitness ".format(aux_info) + \
                      "evaluations:".format(self.get_fitness_evaluation_count())
            success = False

        return OptimizeResult(success, status, message, ngen,
                              self._best_fitness)

    def evolve(self, num_generations):
        """The function responsible for generational evolution.

        Parameters
        ----------
        num_generations : int
            The number of generations to evolve
        """
        self._do_evolution(num_generations)
        if self._hall_of_fame is not None:
            self._hall_of_fame.update(self._get_potential_hof_members())

    @abstractmethod
    def _do_evolution(self, num_generations):
        """Definition of this function should do the heavy lifting of
        performing evolutionary development.

        Parameters
        ----------
        num_generations : int
            The number of generations to evolve

        Notes
        -----
        This function is responsible for incrementing generational age
        """
        raise NotImplementedError

    @abstractmethod
    def _get_potential_hof_members(self):
        """Definition of this function should return the individuals which
        should be considered for induction into the hall of fame.

        Returns
        ----------
        list of Chromosomes :
            Potential hall of fame members
        """
        raise NotImplementedError

    @abstractmethod
    def get_best_individual(self):
        """ Gets the most fit individual

        Returns
        -------
        Chromosome :
            Best individual
        """
        raise NotImplementedError

    @abstractmethod
    def get_best_fitness(self):
        """ Gets the fitness value of the most fit individual

        Returns
        -------
         :
            Fitness of best individual
        """
        raise NotImplementedError

    @abstractmethod
    def get_fitness_evaluation_count(self):
        """ Gets the number of fitness evaluations performed

        Returns
        -------
        int :
            number of fitness evaluations
        """
        raise NotImplementedError

