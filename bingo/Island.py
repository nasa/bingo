"""
This module contains the code for an island in an island-based GA optimization
it is general enough to work on any representation/fitness
"""
import logging
import random
import numpy as np

LOGGER = logging.getLogger(__name__)


class Island(object):
    """
    Island: code for island of genetic algorithm
    """

    def __init__(self, gene_manipulator, fitness_function,
                 target_pop_size=64, cx_prob=0.7, mut_prob=0.01,
                 age_fitness=False):
        """
        Initialization of island

        :param gene_manipulator: the object which is responsible for
                                 generation, crossover, mutation and distance
                                 operations of individuals in the island
        :param fitness_function: the function which describes fitnesses of
                                 individuals in the island
        :param target_pop_size: targeted number of individuals in the island
        :param cx_prob: crossover probability
        :param mut_prob: mutation probability
        """
        self.gene_manipulator = gene_manipulator
        self.fitness_function = fitness_function
        self.target_pop_size = target_pop_size
        self.mut_prob = mut_prob
        self.cx_prob = cx_prob
        self.pop = []
        self.generate_population()
        self.age = 0
        self.fitness_evals = 0
        self.pareto_front = []
        if age_fitness:
            self.generational_step = self.age_fitness_pareto_step
        else:
            self.generational_step = self.deterministic_crowding_step

    def generate_population(self):
        """
        Generates a new random population using the gene manipulator to fill
        the island
        """
        self.pop = [self.gene_manipulator.generate()
                    for _ in range(self.target_pop_size)]

    def deterministic_crowding_step(self):
        """
        Performs a deterministic crowding generational step
        """
        self.age += 1
        for indv in self.pop:
            indv.genetic_age += 1
        # randomly pair by shuffling
        random.shuffle(self.pop)
        for i in range(self.target_pop_size//2):
            p_1 = self.pop[i*2]
            p_2 = self.pop[i*2+1]
            # see if any events occur
            do_cx = random.random() <= self.cx_prob
            do_mut1 = random.random() <= self.mut_prob
            do_mut2 = random.random() <= self.mut_prob
            if do_cx or do_mut1 or do_mut2:
                # do crossover
                if do_cx:
                    c_1, c_2 = self.gene_manipulator.crossover(p_1, p_2)
                else:
                    c_1 = p_1.copy()
                    c_2 = p_2.copy()
                # do mutations
                if do_mut1:
                    c_1 = self.gene_manipulator.mutation(c_1)
                if do_mut2:
                    c_2 = self.gene_manipulator.mutation(c_2)
                # calculate fitnesses
                if p_1.fit_set is False:
                    p_1.fitness = self.fitness_function(p_1)
                    p_1.fit_set = True
                    self.fitness_evals += 1
                if p_2.fit_set is False:
                    p_2.fitness = self.fitness_function(p_2)
                    p_2.fit_set = True
                    self.fitness_evals += 1
                if c_1.fit_set is False:
                    c_1.fitness = self.fitness_function(c_1)
                    c_1.fit_set = True
                    self.fitness_evals += 1
                if c_2.fit_set is False:
                    c_2.fitness = self.fitness_function(c_2)
                    c_2.fit_set = True
                    self.fitness_evals += 1
                # do selection
                dist_a = self.gene_manipulator.distance(p_1, c_1) + \
                         self.gene_manipulator.distance(p_2, c_2)
                dist_b = self.gene_manipulator.distance(p_1, c_2) + \
                         self.gene_manipulator.distance(p_2, c_1)
                if dist_a <= dist_b:
                    if c_1.fitness < p_1.fitness or \
                            np.any(np.isnan(p_1.fitness)):
                        self.pop[i*2] = c_1
                    if c_2.fitness < p_2.fitness or \
                            np.any(np.isnan(p_2.fitness)):
                        self.pop[i*2+1] = c_2
                else:
                    if c_2.fitness < p_1.fitness or \
                            np.any(np.isnan(p_1.fitness)):
                        self.pop[i*2] = c_2
                    if c_1.fitness < p_2.fitness or \
                            np.any(np.isnan(p_2.fitness)):
                        self.pop[i*2+1] = c_1

    def age_fitness_pareto_step(self):
        """
        Performs a age-fitness pareto generational step
        """
        # increment age
        self.age += 1
        for indv in self.pop:
            indv.genetic_age += 1

        # random mating
        # randomly pair by shuffling
        random.shuffle(self.pop)
        start_pop_size = len(self.pop)
        for i in range(start_pop_size//2):
            p_1 = self.pop[i*2]
            p_2 = self.pop[i*2+1]
            # see if any events occur
            do_cx = random.random() <= self.cx_prob
            do_mut1 = random.random() <= self.mut_prob
            do_mut2 = random.random() <= self.mut_prob
            # do crossover
            if do_cx:
                c_1, c_2 = self.gene_manipulator.crossover(p_1, p_2)
                # with possible mutations
                if do_mut1:
                    c_1 = self.gene_manipulator.mutation(c_1)
                if do_mut2:
                    c_2 = self.gene_manipulator.mutation(c_2)
                self.pop.append(c_1)
                self.pop.append(c_2)
            else:
                # just do mutations
                if do_mut1:
                    c_1 = p_1.copy()
                    c_1 = self.gene_manipulator.mutation(c_1)
                    self.pop.append(c_1)
                if do_mut2:
                    c_2 = p_2.copy()
                    c_2 = self.gene_manipulator.mutation(c_2)
                    self.pop.append(c_2)

        # add in one newly generated
        self.pop.append(self.gene_manipulator.generate())

        # selection via age-fitness domination
        selection_attempts = 0
        while len(self.pop) > self.target_pop_size and \
              selection_attempts < start_pop_size * 50:
            # select random pairs
            indv_a = np.random.randint(len(self.pop))
            indv_b = np.random.randint(len(self.pop))
            while indv_b == indv_a:
                indv_b = np.random.randint(len(self.pop))
            # get fitnesses
            if self.pop[indv_a].fit_set is False:
                self.pop[indv_a].fitness = self.fitness_function(
                    self.pop[indv_a])
                self.pop[indv_a].fit_set = True
                self.fitness_evals += 1
            if self.pop[indv_b].fit_set is False:
                self.pop[indv_b].fitness = self.fitness_function(
                    self.pop[indv_b])
                self.pop[indv_b].fit_set = True
                self.fitness_evals += 1
            # check for domination in age-fitness
            if np.any(np.isnan(self.pop[indv_a].fitness)):
                del self.pop[indv_a]
            elif np.any(np.isnan(self.pop[indv_b].fitness)):
                del self.pop[indv_b]
            elif self.pop[indv_a].genetic_age < self.pop[indv_b].genetic_age:
                if self.pop[indv_a].fitness <= self.pop[indv_b].fitness:
                    del self.pop[indv_b]
            elif self.pop[indv_a].genetic_age > self.pop[indv_b].genetic_age:
                if self.pop[indv_b].fitness <= self.pop[indv_a].fitness:
                    del self.pop[indv_a]
            else:  # equal ages
                if self.pop[indv_a].fitness <= self.pop[indv_b].fitness:
                    del self.pop[indv_b]
                else:
                    del self.pop[indv_a]
            selection_attempts += 1

        # LOGGER.debug("population zise: " + str(len(self.pop)) +
        #              "    target size: " + str(self.target_pop_size) +
        #              "    attempts: " + str(selection_attempts))

    def best_indv(self):
        """
        Finds individual with best (lowest) fitness

        :return: fitness of best individual
        """
        best = self.pop[0]
        if best.fit_set is False:
            best.fitness = self.fitness_function(best)
            best.fit_set = True
            self.fitness_evals += 1
        for indv in self.pop[1:]:
            if indv.fit_set is False:
                indv.fitness = self.fitness_function(indv)
                indv.fit_set = True
                self.fitness_evals += 1
            if indv.fitness < best.fitness or np.isnan(best.fitness).any():
                best = indv
        return best

    def dominate(self, indv1, indv2):
        """
        Returns whether or not individual1 dominates individual 2 in all their
        fitness measures: i.e. True if all values in indv1.fitness tuple are
        less than or equal to the counterpart in individual 2 else False

        :param indv1: first individual with fitness member
        :param indv2: second individual with fitness member
        :return: Does indv1 dominate indv2 (boolean)
        """
        if indv1.fit_set is False:
            indv1.fitness = self.fitness_function(indv1)
            indv1.fit_set = True
            self.fitness_evals += 1
        if indv2.fit_set is False:
            indv2.fitness = self.fitness_function(indv2)
            indv2.fit_set = True
            self.fitness_evals += 1
        dominate = True
        for f_1, f_2 in zip(indv1.fitness, indv2.fitness):
            if f_2 < f_1:
                dominate = False
        return dominate

    def similar(self, indv1, indv2):
        """
        Returns whether the fitness measures are equal for two individuals

        :param indv1: first individual with fitness member
        :param indv2: second individual with fitness member
        :return: is indv1.fitness == indv2.fitness (boolean)
        """
        if indv1.fit_set is False:
            indv1.fitness = self.fitness_function(indv1)
            indv1.fit_set = True
            self.fitness_evals += 1
        if indv2.fit_set is False:
            indv2.fitness = self.fitness_function(indv2)
            indv2.fit_set = True
            self.fitness_evals += 1
        return indv1.fitness == indv2.fitness

    def update_pareto_front(self):
        """
        Updates the pareto front based on the current population
        """
        # see if fitness is a tuple or list
        if self.pop[0].fit_set is False:
            self.pop[0].fitness = self.fitness_function(self.pop[0])
            self.pop[0].fit_set = True
            self.fitness_evals += 1
        single_metric = not isinstance(self.pop[0].fitness, tuple) and \
                        not isinstance(self.pop[0].fitness, list)

        # single metric
        if single_metric:
            self.pareto_front = [self.best_indv().copy()]

        # multiple metrics
        else:
            # remove current pareto indv who are dominated by others
            to_remove = []
            for i, p_1 in enumerate(self.pareto_front):
                for j, p_2 in enumerate(self.pareto_front):
                    if i != j:
                        if self.dominate(p_1, p_2):
                            if not self.similar(p_1, p_2) or i < j:
                                to_remove.append(p_2)
            to_remove = list(set(to_remove))
            while len(to_remove) > 0:
                self.pareto_front.remove(to_remove.pop())

            for indv in self.pop:
                if indv.fit_set is False:
                    indv.fitness = self.fitness_function(indv)
                    indv.fit_set = True
                    self.fitness_evals += 1
                # see if indv is dominated by any of the current pareto front
                # also see if it is similar to any of them
                dominated = False
                similar = False
                not_a_number = np.isnan(indv.fitness[0])
                for pareto_indv in self.pareto_front:
                    if self.dominate(pareto_indv, indv):
                        dominated = True
                    if self.similar(pareto_indv, indv):
                        similar = True

                if not dominated and not similar and not not_a_number:
                    # remove any pareto indv who are dominated by inv
                    to_remove = []
                    for pareto_indv in self.pareto_front:
                        if self.dominate(indv, pareto_indv):
                            to_remove.append(pareto_indv)
                    while len(to_remove) > 0:
                        self.pareto_front.remove(to_remove.pop())

                    # then add to pareto front
                    self.pareto_front.append(indv)

            # sort the updated front
            self.pareto_front.sort(key=lambda x: x.fitness)

    def dump_population(self, subset=None, with_removal=False):
        """
        Dumps the population to a pickleable object

        :param subset: list of indices for the subset of the population which
                       is dumped. A None value results in all of the population
                       being dumped.
        :param with_removal: boolean describing whether the elements should be
                             removed from population after dumping
        :return: population in list form
        """
        if subset is None:
            subset = list(range(self.target_pop_size))
        pop_list = []
        for i, indv in enumerate(self.pop):
            if i in subset:
                pop_list.append(self.gene_manipulator.dump(indv))
        if with_removal:
            self.pop[:] = [indv for i, indv in enumerate(self.pop)
                           if i not in subset]
        return pop_list

    def dump_pareto(self):
        """
        Dumps the pareto population to a pickleable object

        :return: pareto front in list form
        """
        pop_list = []
        for indv in self.pareto_front:
            pop_list.append(self.gene_manipulator.dump(indv))
        return pop_list

    def load_population(self, pop_list, replace=True):
        """
        loads population from a pickleable object

        :param pop_list: list of population which is loaded
        :param replace: default (True) value results in all of the population
                        being loaded/replaced. False value means that the
                        population in pop_list is appended to the current
                        population
        """
        if replace:
            self.pop = []
        for indv_list in pop_list:
            self.pop.append(self.gene_manipulator.load(indv_list))
        self.target_pop_size = len(self.pop)
