"""
This module encapsulates different fitness metrics that can be used for
symbolic regression in bingo
"""

import abc
import warnings
import numpy as np
from scipy import optimize


class FitnessMetric(object, metaclass=abc.ABCMeta):
    """fitness metric superclass"""

    def __init__(self):
        """empty init"""
        pass

    def evaluate_fitness(self, individual, training_data):
        """
        Evaluate the fitness of an individual and optimize it if necessary
        :param individual: an AGraph-like individual to be evaluated
        :param training_data: ExplicitTrainingData
        :return fitness of the individual
        """
        # do optimization if necessary
        if individual.needs_optimization():
            self.optimize_constants(individual, training_data)

        fvec = self.evaluate_fitness_vector(individual, training_data)
        return np.mean(np.abs(fvec))

    @abc.abstractmethod
    def evaluate_fitness_vector(self, individual, training_data):
        """
        returns the fitness vector of an individual using a given set of
        training data
        :param individual: a gene to be evaluated
        :param training_data: the data used by the fitness metric
        :return: fitness vector
        """
        pass

    def optimize_constants(self, individual, training_data):
        """
        perform levenberg-marquardt optimization on embedded constants
        :param individual: a gene to be evaluated
        :param training_data: the data used by the fitness metric
        """
        num_constants = individual.count_constants()
        c_0 = np.random.uniform(-100, 100, num_constants)

        # define fitness function for optimization
        def const_opt_fitness(consts):
            """ fitness function for constant optimization"""
            individual.set_constants(consts)
            fvec = self.evaluate_fitness_vector(individual, training_data)
            return fvec

        # do optimization
        sol = optimize.root(const_opt_fitness, c_0, method='lm')

        # put optimal values in command list
        individual.set_constants(sol.x)


class StandardRegression(FitnessMetric):
    """ Traditional fitness evaluation """

    def __init__(self, const_deriv=False):
        """
        Initialization
        :param const_deriv: boolean for whether optimization of constants will
                            use calculated derivative (true) or numerical
                            derivatives (false)
        """
        super().__init__()
        self.const_deriv = const_deriv

    def evaluate_fitness_vector(self, individual, training_data):
        """
        fitness vector = f(x) - y
        where f is defined by the individual and x, y are in the training data
        :param individual: an AGraph-like individual to be evaluated
        :param training_data: ExplicitTrainingData
        :return fitness vector
        """
        f_of_x = individual.evaluate(training_data.x)

        return (f_of_x - training_data.y).flatten()

    def evaluate_fit_vec_w_const_deriv(self, individual,
                                       training_data):
        """
        returns the fitness vector and its derivatove with respect any included
        constantsof an individual using a given set of training data
        :param individual: a gene to be evaluated
        :param training_data: the data used by the fitness metric
        :return: fitness vector, dfitness/dconstants array
        """

        f_of_x, df_dc = individual.evaluate_with_const_deriv(training_data.x)

        return (f_of_x - training_data.y).flatten(), df_dc

    def optimize_constants(self, individual, training_data):
        """
        perform levenberg-marquardt optimization on embedded constants
        :param individual: a gene to be evaluated
        :param training_data: the data used by the fitness metric
        """
        num_constants = individual.count_constants()
        c_0 = np.random.uniform(-100, 100, num_constants)

        if self.const_deriv:
            # define fitness function for optimization
            def const_opt_fitness(consts):
                """ fitness function for constant optimization"""
                individual.set_constants(consts)
                fvec, dfvec_dc = self.evaluate_fit_vec_w_const_deriv(
                    individual, training_data)
                return fvec, dfvec_dc

            # do optimization
            sol = optimize.root(const_opt_fitness, c_0, jac=True, method='lm')

        else:
            # define fitness function for optimization
            def const_opt_fitness(consts):
                """ fitness function for constant optimization"""
                individual.set_constants(consts)
                fvec = self.evaluate_fitness_vector(individual, training_data)
                return fvec

            # do optimization
            sol = optimize.root(const_opt_fitness, c_0, method='lm')

        # put optimal values in command list
        individual.set_constants(sol.x)


class ImplicitRegression(FitnessMetric):
    """ Implicit Regression, version 2"""

    def __init__(self, required_params=None, normalize_dot=False,
                 acceptable_nans=0.1):
        """
        Initialization
        Fitness of this metric is related cos of angle between between df_dx
        and dx_dt. Different normalization and error checking are available.

        :param required_params: minimum number of nonzero components of dot
        :param normalize_dot: normalize the terms in the dot product
        """
        super().__init__()
        self.required_params = required_params
        self.normalize_dot = normalize_dot
        self.acceptable_finite_fraction = 1 - acceptable_nans

    def evaluate_fitness_vector(self, individual, training_data):
        """
        Fitness of this metric is related cos of angle between between df_dx
        and dx_dt. Different normalization and erorr checking are available.

        :param individual: an AGraph-like individual to be evaluated
        :param training_data: ImplicitTrainingData
        :return fitness vector
        """
        _, df_dx = individual.evaluate_deriv(x=training_data.x)

        if self.normalize_dot:
            dot = (df_dx / np.linalg.norm(df_dx, axis=1).reshape((-1, 1))) * \
                  (training_data.dx_dt /
                   np.linalg.norm(training_data.dx_dt, axis=1).reshape((-1, 1)))
        else:
            dot = df_dx * training_data.dx_dt

        if self.required_params is not None:
            n_params_used = (abs(dot) > 1e-16).sum(1)
            enough_params_used = np.any(n_params_used >= self.required_params)
            if not enough_params_used:  # not enough parameters
                return np.full((training_data.x.shape[0],), np.inf)

        denominator = np.sum(np.abs(dot), axis=1)
        new = np.sum(dot, axis=1) / denominator
        new[~np.isfinite(denominator)] = np.inf
        return new

    def evaluate_fitness(self, individual, training_data):
        """
        Fitness of this metric is related cos of angle between between df_dx
        and dx_dt. Different normalization and erorr checking are available.

        :param individual: an AGraph-like individual to be evaluated
        :param training_data: ImplicitTrainingData
        :return: the mean of the fitness vector, ignoring nans
        """
        # do optimization if necessary
        if individual.needs_optimization():
            self.optimize_constants(individual, training_data)

        fvec = self.evaluate_fitness_vector(individual, training_data)
        finite_fraction = np.count_nonzero(np.isfinite(fvec))/fvec.shape[0]
        if finite_fraction < self.acceptable_finite_fraction:
            err = np.inf
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                err = np.nanmean(np.abs(fvec))
        return err


# I DONT THINK THIS ONE WORKS BECAUSE IT FAILS TO CONSIDER ELASTIC STRAIN
# CHANGES DURING THE STEPS.  THIS COULD BE MADE TO WORK IF WE WERE TO HAVE
# PLASTIC STRAIN AS AN INPUT RATHER THAN TOTAL STRAIN
# class PlasticityRegression(FitnessMetric):
#     """ Implicit Regression, version """
#
#     need_df_dx = True
#     need_dx_dt = True
#
#     @staticmethod
#     def evaluate_vector(x, df_dx, dx_dt, required_params=2):
#         """ error = cos of angle between between df_dx and dx_dt """
#
#         df_ds = np.copy(df_dx[:, :3])
#         df_dp = np.copy(df_dx[:, :-3])
#         dp_dt = np.copy(dx_dt[:, :-3])
#         de_dt = np.copy(dx_dt[:, -3:])
#
#         C_f = (df_dp/np.linalg.norm(df_dp, axis=1).reshape((-1, 1))) * \
#               (dp_dt/np.linalg.norm(dp_dt, axis=1).reshape((-1, 1)))
#         C_g = (df_ds/np.linalg.norm(df_ds, axis=1).reshape((-1, 1))) * \
#               (de_dt/np.linalg.norm(de_dt, axis=1).reshape((-1, 1)))
#         n_params_used = np.count_nonzero(abs(C_f) > 1e-16, axis=1)
#         if np.any(n_params_used >= required_params):
#             diff = (1 + np.abs(np.sum(C_f, axis=1))) * \
#                    (2 - np.sum(C_g, axis=1)) - 1
#
#         else:  # not enough parameters in const regression
#             diff = np.full((x.shape[0], ), np.inf)
#         return diff
#
#     @classmethod
#     def evaluate_metric(cls, **kwargs):
#         """ evaluate metric but ignore some nans in vector"""
#         vec = cls.evaluate_vector(**kwargs)
#         nan_count = np.count_nonzero(np.isnan(vec))
#         # ok
#         if nan_count < 0.1 * len(vec):
#             return np.nanmean(np.abs(vec))
#         # too many nans
#         else:
#             return np.nan


class ImplicitRegressionSchmidt(FitnessMetric):
    """ Implicit Regression, version from schmidt and lipson """

    def evaluate_fitness_vector(self, individual, training_data):
        """
        from schmidt and lipson's papers

        :param individual: an AGraph-like individual to be evaluated
        :param training_data: ImplicitTrainingData

        :return: the fitness for each row
        """

        # NOTE: this doesnt work well right now
        #       importantly, it couldn't reproduce the papers

        _, df_dx = individual.evaluate_deriv(x=training_data.x)

        n_params = training_data.x.shape[1]
        # print("----------------------------------")
        worst_fit = 0
        diff_worst = np.full((n_params, ), np.inf)
        for i in range(n_params):
            for j in range(n_params):
                if i != j:
                    df_dxi = np.copy(df_dx[:, i])
                    df_dxj = np.copy(df_dx[:, j])
                    dxi_dxj_2 = (training_data.dx_dt[:, i]/
                                 training_data.dx_dt[:, j])
                    # print("independent:", i, "-", j)
                    for k in range(n_params):
                        if k != i and k != j:
                            # print("  dependent:", i, "-", k)
                            # df_dxi += df_dx[:, k]*dx_dt[:, k]/dx_dt[:, i]
                            df_dxj += df_dx[:, k] * \
                                      training_data.dx_dt[:, k] / \
                                      training_data.dx_dt[:, j]

                    dxi_dxj_1 = df_dxj / df_dxi
                    diff = np.log(1. + np.abs(dxi_dxj_1 + dxi_dxj_2))
                    fit = np.mean(diff)
                    # print("        fit:", fit)
                    if np.isfinite(fit) and fit > worst_fit:
                        # print(i, j)
                        diff_worst = np.copy(diff)
                        worst_fit = fit
        # print(diff_worst)
        return diff_worst


class PairwiseAtomicPotential(FitnessMetric):
    """
    Pairwise atomic potential which is fit with total potential energy for a
    set of configurations
    """

    def evaluate_fitness_vector(self, individual, training_data):
        """
        Fitness is calculated as how well total potential energies are matched
        by the summation of pairwise energies which are calculated by the
        individual
        fitness = sum( f(r_i) ) - U_true      for i in config

        :param individual: an AGraph-like individual to be evaluated
        :param training_data: ImplicitTrainingData

        :return: the fitness for each row
        """
        pair_energies = individual.evaluate(training_data.r).flatten()

        err_vec = []
        for i, energy_true in enumerate(training_data.potential_energy):
            energy = np.sum(pair_energies[training_data.config_lims_r[i]:
                                          training_data.config_lims_r[i+1]])
            err_vec.append(energy - energy_true)

        return np.array(err_vec).flatten()
