"""
Defines an island that has 3 populations and performs the simultaneous
evolution of a main population, and a fitness predictor population.  The third
population is a set of training individuals on which the quality of the fitness
predictors is judged.
This is loosely based on the work of Schmidt and Lipson 2008?
"""
import numpy as np

from .Island import Island
from .Utils import calculate_partials


class CoevolutionIsland(object):
    """
    Coevolution island with 3 populations
    solution_pop: the solutions to symbolic regression
    predictor_pop: sub sampling
    trainers: population of solutions which are used to train the predictors


    :param data_x: 2d numpy array of independent data, 1st dimension
                   corresponds to multiple datapoints and 2nd dimension
                   corresponds to multiple x variables
    :param data_y: 1d numpy array for the dependent variable.  a None value
                   here indicates that implicit (constant) symbolic
                   regression should be used on the x data
    :param solution_manipulator: a gene manipulator for the symbolic
                                 regression solution population
    :param predictor_manipulator: a gene manipulator for the fitness
                                  predictor population
    :param solution_pop_size: size of the solution population
    :param solution_cx: crossover probability for the solution population
    :param solution_mut: mutation probability for the solution population
    :param predictor_pop_size: size of the fitness predictor population
    :param predictor_cx: crossover probability for the fitness predictor
                         population
    :param predictor_mut: mutation probability for the fitness predictor
                          population
    :param predictor_ratio: approximate ratio of time spent on fitness
                            predictor calculations and the total
                            computation time
    :param predictor_update_freq: number of generations of the solution
                                  population after which the fitness
                                  predictor is updated
    :param trainer_pop_size: size of the trainer population
    :param trainer_update_freq: number of generations of the solution
                                population after which a new trainer is
                                added to the trainer population
    :param required_params: number of unique parameters that are required
                            in implicit (constant) symbolic regression
    :param verbose: True for extra output printed to screen
    """

    def __init__(self, data_x, data_y, solution_manipulator,
                 predictor_manipulator, fitness_metric,
                 solution_pop_size=64, solution_cx=0.7, solution_mut=0.01,
                 predictor_pop_size=16, predictor_cx=0.5, predictor_mut=0.1,
                 predictor_ratio=0.1, predictor_update_freq=50,
                 trainer_pop_size=16, trainer_update_freq=50,
                 verbose=False,
                 **fitness_metric_args):
        """
        Initializes coevolution island
        """
        self.verbose = verbose
        self.fitness_metric = fitness_metric
        self.fitness_metric_args = fitness_metric_args

        # fill fitness metric args
        if self.fitness_metric.need_dx_dt:
            data_x, fitness_metric_args['dx_dt'], inds = \
                calculate_partials(data_x)
            if data_y is not None:
                data_y = data_y[inds, ...]
        if self.fitness_metric.need_x:
            fitness_metric_args['x'] = data_x
        if self.fitness_metric.need_y:
            fitness_metric_args['y'] = data_y

        # check if fitness predictors are valid range
        if data_x.shape[0] < predictor_manipulator.max_index:
            predictor_manipulator.max_index = data_x.shape[0]

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
                    if np.isnan(pred.fit_func(sol, self.fitness_metric,
                                              **self.fitness_metric_args)):
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
            print("P>", self.predictor_island.age, best_pred.fitness, best_pred)
            self.solution_island.update_pareto_front()
            best_sol = self.solution_island.pareto_front[0]
            print("S>", self.solution_island.age, best_sol.fitness, \
                best_sol.latexstring())

    def solution_fitness_est(self, solution):
        """
        Estimated fitness for solution pop based on the best predictor

        :param solution: individual of the solution population for which the
                         fitness will be calculated
        :return: fitness, complexity
        """
        fit = self.best_predictor.fit_func(solution, self.fitness_metric,
                                           **self.fitness_metric_args)
        return fit, solution.complexity()

    def predictor_fitness(self, predictor):
        """
        Fitness function for predictor population, based on the ability to
        accurately describe the true fitness of the trainer population

        :param predictor: predictor for which the fitness is assessed
        :return: fitness
        """
        err = 0.0
        for train, true_fit in zip(self.trainers, self.trainers_true_fitness):
            predicted_fit = predictor.fit_func(train, self.fitness_metric,
                                               **self.fitness_metric_args)
            err += abs(true_fit - predicted_fit)
        return err/len(self.trainers)

    def solution_fitness_true(self, solution):
        """
        full calculation of fitness for solution population

        :param solution: individual of the solution population for which the
                         fitness will be calculated
        :return: fitness
        """

        # calculate what is needed from individual (f and/or df_dx)
        metric_args = dict(self.fitness_metric_args)
        if self.fitness_metric.need_df_dx:
            f_of_x, df_dx = solution.evaluate_deriv(self.fitness_metric,
                                               **self.fitness_metric_args)
            metric_args['df_dx'] = df_dx
            if self.fitness_metric.need_f:
                metric_args['f'] = f_of_x
        elif self.fitness_metric.need_f:
            metric_args['f'] = solution.evaluate(self.fitness_metric,
                                                 **self.fitness_metric_args)

        # calculate fitness metric
        err = self.fitness_metric.evaluate_metric(**metric_args)

        return err

    def add_new_trainer(self):
        """
        Add/replace trainer to current trainer population.  The trainer which
        maximizes discrepancy between fitness predictors is chosen
        """
        s_best = self.solution_island.pop[0]
        max_variance = 0
        for sol in self.solution_island.pop:
            pfit_list = []
            for pred in self.predictor_island.pop:
                pfit_list.append(pred.fit_func(sol, self.fitness_metric,
                                               **self.fitness_metric_args))
            try:
                variance = np.var(pfit_list)
            except (ArithmeticError, OverflowError, FloatingPointError,
                    ValueError):
                variance = np.nan
            if variance > max_variance:
                max_variance = variance
                s_best = sol.copy()
        location = (self.solution_island.age // self.trainer_update_freq)\
                   % len(self.trainers)
        if self.verbose:
            print("updating trainer at location", location)
        self.trainers[location] = s_best

    def deterministic_crowding_step(self):
        """
        Deterministic crowding step for solution population, This function
        takes the necessary steps for the other populations to maintain desired
        predictor/solution computation ratio
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
                print("P>", self.predictor_island.age, end=' ')
                print(best_pred.fitness, best_pred)
            current_ratio = (float(self.predictor_island.fitness_evals) /
                             (self.predictor_island.fitness_evals +
                              float(self.solution_island.fitness_evals) /
                              self.predictor_to_solution_eval_cost))

        # update fitness predictor if it is time to
        if (self.solution_island.age+1) % self.predictor_update_freq == 0:
            if self.verbose:
                print("updating predictor")
            self.best_predictor = self.predictor_island.best_indv().copy()
            for indv in self.solution_island.pop:
                indv.fitness = None

        # do step on solution island
        self.solution_island.deterministic_crowding_step()
        self.solution_island.update_pareto_front()
        if self.verbose:
            best_sol = self.solution_island.pareto_front[0]
            print("S>", self.solution_island.age, best_sol.fitness, \
                best_sol.latexstring())

    def dump_populations(self, s_subset=None, p_subset=None, t_subset=None):
        """
        Dump the 3 populations to a pickleable object (tuple of lists)

        :param s_subset: list of indices for the subset of the solution
                         population which is dumped. A None value results in
                         all of the population being dumped.
        :param p_subset: list of indices for the subset of the fitness
                         predictor population which is dumped. A None value
                         results in all of the population being dumped.
        :param t_subset: list of indices for the subset of the trainer
                         population which is dumped. A None value results in
                         all of the population being dumped.
        :return: tuple of lists of populations
        """
        # dump solutions
        solution_list = self.solution_island.dump_population(s_subset)

        # dump predictors
        predictor_list = self.predictor_island.dump_population(p_subset)

        # dump trainers
        trainer_list = []
        if t_subset is None:
            t_subset = list(range(len(self.trainers)))
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

        :param pop_lists: tuple of lists of the 3 populations
        :param s_subset: list of indices for the subset of the solution
                         population which is loaded and replaced. A None value
                         results in all of the population being
                         loaded/replaced.
        :param p_subset: list of indices for the subset of the fitness
                         predictor population which is loaded and replaced. A
                         None value  results in all of the population being
                         loaded/replaced.
        :param t_subset: list of indices for the subset of the trainer
                         population which is loaded and replaced. A None value
                         results in  all of the population being
                         loaded/replaced.
        """
        # load solutions
        self.solution_island.load_population(pop_lists[0], s_subset)

        # load predictors
        self.predictor_island.load_population(pop_lists[1], p_subset)

        # load trainers
        if t_subset is None:
            self.trainers = [None]*len(pop_lists[2])
            self.trainers_true_fitness = [None]*len(pop_lists[2])
            t_subset = list(range(len(pop_lists[2])))
        for ind, (indv_list, t_fit) in zip(t_subset, pop_lists[2]):
            self.trainers[ind] = \
                self.solution_island.gene_manipulator.load(indv_list)
            self.trainers_true_fitness[ind] = t_fit

        self.best_predictor = self.predictor_island.best_indv().copy()

    def print_trainers(self):
        """
        For debugging: print trainers to screen
        """
        for i, train, tfit in zip(list(range(len(self.trainers))),
                                  self.trainers,
                                  self.trainers_true_fitness):
            print("T>", i, tfit, train.latexstring())

    def use_true_fitness(self):
        """
        Sets the fitness function for the solution population to the true
        (full) fitness rather than using a fitness predictor.
        """
        self.solution_island.fitness_function = \
            self.true_fitness_plus_complexity
        for indv in self.solution_island.pop:
            indv.fitness = None

    def true_fitness_plus_complexity(self, solution):
        """
        Gets the true (full) fitness and complexity of a solution individual

        :param solution: individual of the solution population for which the
                         fitness will be calculated
        :return: fitness, complexity
        """
        return self.solution_fitness_true(solution), solution.complexity()
