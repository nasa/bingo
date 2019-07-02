"""
This module contains the basic structure for evolutionary optimization in
bingo.  The general framework allows access to an evolve_until_convergence
function.
"""
from abc import ABCMeta, abstractmethod
from collections import namedtuple
import dill
import os
from ..Util.ArgumentValidation import argument_validation


OptimizeResult = namedtuple('OptimizeResult', ['success', 'status', 'message',
                                               'ngen', 'fitness'])

# TODO hof in attributes doc
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
        self.hall_of_fame = hall_of_fame
        self._previous_checkpoints = []

    @argument_validation(max_generations={">=": 1},
                         min_generations={">=": 0},
                         convergence_check_frequency={">": 0})
    def evolve_until_convergence(self, max_generations,
                                 fitness_threshold,
                                 convergence_check_frequency=1,
                                 min_generations=0,
                                 stagnation_generations=None,
                                 max_fitness_evaluations=None,
                                 checkpoint_base_name=None,
                                 num_checkpoints=None):
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
        checkpoint_base_name: str
            base file name for checkpoint files
        num_checkpoints: int (optional)
            number of recent checkpoints to keep, previous ones are removed

        Returns
        --------
        OptimizeResult :
            Object containing information about the result of the run
        """
        self._starting_age = self.generational_age
        self._update_best_fitness()
        self._update_checkpoints(checkpoint_base_name, num_checkpoints,
                                 reset=True)

        while self.generational_age - self._starting_age < min_generations:
            self.evolve(convergence_check_frequency)
            self._update_best_fitness()
            self._update_checkpoints(checkpoint_base_name, num_checkpoints)

        if self._convergence(fitness_threshold):
            return self._make_optim_result(0, fitness_threshold)
        if self._stagnation(stagnation_generations):
            return self._make_optim_result(1, stagnation_generations)
        if self._hit_max_evals(max_fitness_evaluations):
            return self._make_optim_result(3, max_fitness_evaluations)

        while self.generational_age - self._starting_age < max_generations:
            self.evolve(convergence_check_frequency)
            self._update_best_fitness()
            self._update_checkpoints(checkpoint_base_name, num_checkpoints)

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

    def _update_checkpoints(self, checkpoint_base_name, num_checkpoints,
                            reset=False):
        if reset:
            self._previous_checkpoints = []

        if checkpoint_base_name is not None:
            checkpoint_file_name = "{}_{}.pkl".format(checkpoint_base_name,
                                                      self.generational_age)
            self.dump_to_file(checkpoint_file_name)
            if num_checkpoints is not None:
                self._previous_checkpoints.append(checkpoint_file_name)
                if len(self._previous_checkpoints) > num_checkpoints:
                    os.remove(self._previous_checkpoints.pop(0))

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
        if self.hall_of_fame is not None:
            self.hall_of_fame.update(self._get_potential_hof_members())

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

    def dump_to_file(self, filename):
        """ Dump the EO object to a pickle file

        Parameters
        ----------
        filename : str
            the name of the pickle file to dump
        """
        with open(filename, "wb") as dump_file:
            dill.dump(self, dump_file, protocol=dill.HIGHEST_PROTOCOL)


def load_evolutionary_optimizer_from_file(filename):
    """ Load an EO object from a pickle file

    Parameters
    ----------
    filename : str
        the name of the pickle file to load

    Returns
    -------
    str :
        an evolutionary optimizer
    """
    with open(filename, "rb") as load_file:
        ev_opt = dill.load(load_file)
    return ev_opt
