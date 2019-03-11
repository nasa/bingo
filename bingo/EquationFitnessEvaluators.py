"""Fitness evaluators for Equations

A collection of fitness evaluators that are intended to be used with bingo
Equations for performing symbolic regression.  All fitness evaluators in this
collection require initialization with training data of some sort.
"""
import numpy as np

from bingo.Base.FitnessEvaluator import VectorBasedEvaluator


class ExplicitRegression(VectorBasedEvaluator):
    """ Traditional fitness evaluation for symbolic regression

    fitness = y - f(x) where x and y are in the training_data (i.e.
    training_data.x and training_data.y) and the function f is defined by the
    input Equation individual.

    Parameters
    ----------
    training_data :
                   data that is used in fitness evaluation.  Must have
                   attributes x and y.
    """
    def _evaluate_fitness_vector(self, individual):
        f_of_x = individual.evaluate_equation_at(self.training_data.x)
        return (f_of_x - self.training_data.y).flatten()


class ImplicitRegression(VectorBasedEvaluator):
    """ Implicit Regression, version 2

    Fitness of this metric is related to the cos of angle between between
    df_dx(x) and dx_dt. df_dx(x) is calculated through derivatives of the input
    Equation individual at training_data.x. dx_dt is from training_data.dx_dt.

    Different normalization and error checking are available.

    Parameters
    ----------
    training_data :
                   data that is used in fitness evaluation.  Must have
                   attributes x and dx_dt.
    required_params : int
                      (optional) minimum number of nonzero components of dot
    normalize_dot : bool
                    normalize the terms in the dot product (default = False)
    """
    def __init__(self, training_data, required_params=None,
                 normalize_dot=False):
        super().__init__(training_data)
        self._required_params = required_params
        self._normalize_dot = normalize_dot

    def _evaluate_fitness_vector(self, individual):
        _, df_dx = individual.evaluate_equation_with_x_gradient_at(
            x=self.training_data.x)

        dot_product = self._do_dfdx_dot_dxdt(df_dx)

        if self._required_params is not None:
            if not self._enough_parameters_used(dot_product):
                return np.full((self.training_data.x.shape[0],), np.inf)

        denominator = np.sum(np.abs(dot_product), axis=1)
        normalized_fitness = np.sum(dot_product, axis=1) / denominator
        normalized_fitness[~np.isfinite(denominator)] = np.inf
        return normalized_fitness

    def _enough_parameters_used(self, dot_product):
        n_params_used = (abs(dot_product) > 1e-16).sum(1)
        enough_params_used = np.any(n_params_used >= self._required_params)
        return enough_params_used

    def _do_dfdx_dot_dxdt(self, df_dx):
        left_dot = df_dx
        right_dot = self.training_data.dx_dt
        if self._normalize_dot:
            left_dot = self._normalize_by_row(left_dot)
            right_dot = self._normalize_by_row(right_dot)
        return left_dot * right_dot

    @staticmethod
    def _normalize_by_row(array):
        return array / np.linalg.norm(array, axis=1).reshape((-1, 1))


class ImplicitRegressionSchmidt(VectorBasedEvaluator):
    """ Implicit Regression, Adapted from Schmidt and Lipson papers

    Fitness in this method is the difference of partial derivatives pairs
    calculated with the data and the input Equation individual.

    Parameters
    ----------
    training_data :
                   data that is used in fitness evaluation.  Must have
                   attributes x and dx_dt.

    Notes
    -----
    This may not be a correct implementation of this algorithm.  Importantly,
    it couldn't reproduce the  results in the papers.
    """
    def _evaluate_fitness_vector(self, individual):
        _, df_dx = individual.evaluate_equation_with_x_gradient_at(
            x=self.training_data.x)

        num_parameters = self.training_data.x.shape[1]
        worst_fitness = 0
        diff_worst = np.full((num_parameters, ), np.inf)
        for i in range(num_parameters):
            for j in range(num_parameters):
                if i != j:
                    df_dxi = np.copy(df_dx[:, i])
                    df_dxj = np.copy(df_dx[:, j])
                    dxi_dxj_2 = (self.training_data.dx_dt[:, i] /
                                 self.training_data.dx_dt[:, j])
                    for k in range(num_parameters):
                        if k != i and k != j:
                            df_dxj += df_dx[:, k] * \
                                      self.training_data.dx_dt[:, k] / \
                                      self.training_data.dx_dt[:, j]

                    dxi_dxj_1 = df_dxj / df_dxi
                    diff = np.log(1. + np.abs(dxi_dxj_1 + dxi_dxj_2))
                    fit = np.mean(diff)
                    if np.isfinite(fit) and fit > worst_fitness:
                        diff_worst = np.copy(diff)
                        worst_fitness = fit
        return diff_worst


class PairwiseAtomicPotential(VectorBasedEvaluator):
    """Fitness based on total potential energy of a set of configurations.

    Pairwise atomic potential which is fit with total potential energy for a
    set of configurations. Fitness is calculated as how well total potential
    energies are matched by the summation of pairwise energies which are
    calculated by the Equation individual

    fitness = sum(abs(  sum( f(r_i) ) - U_true_i  ))    for i in config

    Parameters
    ----------
    training_data :
                   data that is used in fitness evaluation.  Must have
                   attributes r, potential_energy and config_lims_r.
    """

    def _evaluate_fitness_vector(self, individual):
        pair_energies = individual.evaluate_equation_at(
            self.training_data.r).flatten()

        err_vec = []
        for i, energy_true in enumerate(self.training_data.potential_energy):
            energy = np.sum(pair_energies[self.training_data.config_lims_r[i]:
                                          self.training_data.config_lims_r[
                                              i + 1]])
            err_vec.append(energy - energy_true)

        return np.array(err_vec).flatten()
