"""
Defines an island that has 3 populations and performs the simultaneous
evolution of a main population, and a fitness predictor population.  The third
population is a set of training individuals on which the quality of the fitness
predictors is judged.
This is loosely based on the work of Schmidt and Lipson 2008?
"""
import numpy as np

from Island import Island


class CoevolutionIsland(object):
    """
    Coevolution island with 3 populations
    solution_pop: the solutions to symbolic regression
    predictor_pop: sub sampling
    """

    def __init__(self, data_x, data_y, solution_manipulator,
                 predictor_manipulator,
                 solution_pop_size=64, solution_cx=0.7, solution_mut=0.01,
                 predictor_pop_size=16, predictor_cx=0.5, predictor_mut=0.1,
                 predictor_ratio=0.1, predictor_update_freq=50,
                 trainer_pop_size=16, trainer_update_freq=50,
                 verbose=False):
        self.data_x = data_x
        self.data_y = data_y
        self.verbose = verbose

        # initialize solution island
        self.solution_island = Island(solution_manipulator,
                                      self.solution_fitness_est,
                                      pop_size=solution_pop_size,
                                      cx_prob=solution_cx,
                                      mut_prob=solution_mut)
        # initialize fitness predictor island
        self.predictor_island = Island(predictor_manipulator,
                                       self.predictor_fitness,
                                       pop_size=predictor_pop_size,
                                       cx_prob=predictor_cx,
                                       mut_prob=predictor_mut)
        self.predictor_update_freq = predictor_update_freq

        # initialize trainers
        self.trainers = []
        self.trainers_true_fitness = []
        for _ in range(trainer_pop_size):
            legal_trainer_found = False
            while not legal_trainer_found:
                ind = np.random.randint(0, solution_pop_size)
                sol = self.solution_island.pop[ind]
                true_fitness = self.solution_fitness_true(sol)
                legal_trainer_found = not np.isnan(true_fitness)
                for pred in self.predictor_island.pop:
                    if np.isnan(pred.fit_func(sol, self.data_x, self.data_y)):
                        legal_trainer_found = False
            self.trainers.append(self.solution_island.pop[ind].copy())
            self.trainers_true_fitness.append(true_fitness)
        self.trainer_update_freq = trainer_update_freq

        # computational balance
        self.predictor_ratio = predictor_ratio
        self.predictor_to_solution_eval_cost = len(self.trainers)

        # find best predictor for use as starting fitness
        # function in solution island
        self.best_predictor = self.predictor_island.best_indv().copy()

        # initial output
        if self.verbose:
            best_pred = self.best_predictor
            print "P>", self.predictor_island.age, best_pred.fitness, best_pred
            self.solution_island.update_pareto_front()
            best_sol = self.solution_island.pareto_front[0]
            print "S>", self.solution_island.age, best_sol.fitness, \
                best_sol.latexstring()

    def solution_fitness_est(self, solution):
        """estimated fitness for solution pop based on the best predictor"""
        fit = self.best_predictor.fit_func(solution, self.data_x, self.data_y)
        return fit, solution.complexity()

    def predictor_fitness(self, predictor):
        """fitness function for predictor population"""
        err = 0.0
        for train, true_fit in zip(self.trainers, self.trainers_true_fitness):
            predicted_fit = predictor.fit_func(train, self.data_x, self.data_y)
            err += abs(true_fit - predicted_fit)
        return err/len(self.trainers)

    def solution_fitness_true(self, solution):
        """full calculation of fitness for solution population"""
        err = 0.0
        nan_count = 0
        tot_n = self.data_x.shape[0]
        for x, y in zip(self.data_x, self.data_y):
            diff = abs(solution.evaluate(x) - y)
            if np.isnan(diff):
                nan_count += 1
            else:
                err += diff/tot_n
        if nan_count < 0.1*tot_n:
            return err*(tot_n/(tot_n-nan_count))
        else:
            return np.nan

    def add_new_trainer(self):
        """add/replace trainer in current trainer population"""
        s_best = self.solution_island.pop[0]
        max_variance = 0
        for sol in self.solution_island.pop:
            pfit_list = []
            for pred in self.predictor_island.pop:
                pfit_list.append(pred.fit_func(sol, self.data_x, self.data_y))
            try:
                variance = np.var(pfit_list)
            except (ArithmeticError, OverflowError, FloatingPointError,
                    ValueError):
                variance = np.nan
            if variance > max_variance:
                max_variance = variance
                s_best = sol.copy()
        location = (self.solution_island.age / self.trainer_update_freq)\
                   % len(self.trainers)
        if self.verbose:
            print "updating trainer at location", location
        self.trainers[location] = s_best

    def deterministic_crowding_step(self):
        """
        deterministic crowding step for solution pop, and take necessary steps
        for other pops
        """
        # do some step(s) on predictor island if the ratio is low
        current_ratio = (float(self.predictor_island.fitness_evals) /
                         (self.predictor_island.fitness_evals +
                          float(self.solution_island.fitness_evals) /
                          self.predictor_to_solution_eval_cost))

        # evolving predictors
        while current_ratio < self.predictor_ratio:
            # update trainers if it is time to
            if (self.predictor_island.age+1) % self.trainer_update_freq == 0:
                self.add_new_trainer()
                for indv in self.predictor_island.pop:
                    indv.fitness = None
            # do predictor step
            self.predictor_island.deterministic_crowding_step()
            if self.verbose:
                best_pred = self.predictor_island.best_indv()
                print "P>", self.predictor_island.age,
                print best_pred.fitness, best_pred
            current_ratio = (float(self.predictor_island.fitness_evals) /
                             (self.predictor_island.fitness_evals +
                              float(self.solution_island.fitness_evals) /
                              self.predictor_to_solution_eval_cost))

        # update fitness predictor if it is time to
        if (self.solution_island.age+1) % self.predictor_update_freq == 0:
            if self.verbose:
                print "updating predictor"
            self.best_predictor = self.predictor_island.best_indv().copy()
            for indv in self.solution_island.pop:
                indv.fitness = None

        # do step on solution island
        self.solution_island.deterministic_crowding_step()
        self.solution_island.update_pareto_front()
        if self.verbose:
            best_sol = self.solution_island.pareto_front[0]
            print "S>", self.solution_island.age, best_sol.fitness, \
                best_sol.latexstring()

    def dump_populations(self, s_subset=None, p_subset=None, t_subset=None):
        """
        dump 3 populations to pickleable object
        """
        # dump solutions
        solution_list = self.solution_island.dump_population(s_subset)

        # dump predictors
        predictor_list = self.predictor_island.dump_population(p_subset)

        # dump trainers
        trainer_list = []
        if t_subset is None:
            t_subset = range(len(self.trainers))
        for i, (indv, tfit) in enumerate(zip(self.trainers,
                                             self.trainers_true_fitness)):
            if i in t_subset:
                trainer_list.append(
                    (self.solution_island.gene_manipulator.dump(indv), tfit))

        return solution_list, predictor_list, trainer_list

    def load_populations(self, pop_lists, s_subset=None, p_subset=None,
                         t_subset=None):
        """
        load 3 populations from pickleable object
        """
        # load solutions
        self.solution_island.load_population(pop_lists[0], s_subset)

        # load predictors
        self.predictor_island.load_population(pop_lists[1], p_subset)

        # load trainers
        if t_subset is None:
            self.trainers = [None]*len(pop_lists[2])
            self.trainers_true_fitness = [None]*len(pop_lists[2])
            t_subset = range(len(pop_lists[2]))
        for ind, (indv_list, t_fit) in zip(t_subset, pop_lists[2]):
            self.trainers[ind] = \
                self.solution_island.gene_manipulator.load(indv_list)
            self.trainers_true_fitness[ind] = t_fit

        self.best_predictor = self.predictor_island.best_indv().copy()

    def print_trainers(self):
        """for debugging: print trainers to screen"""
        for i, train, tfit in zip(range(len(self.trainers)), self.trainers,
                                  self.trainers_true_fitness):
            print "T>", i, tfit, train.latexstring()

    def use_true_fitness(self):
        """
        sets the fitness function for the solution population to the true
        (full) fitness
        """
        self.solution_island.fitness_function = \
            self.true_fitness_plus_complexity
        for indv in self.solution_island.pop:
            indv.fitness = None


    def true_fitness_plus_complexity(self, solution):
        """
        gets the true fitness and complexity of a solution individual
        """
        return self.solution_fitness_true(solution), solution.complexity()
