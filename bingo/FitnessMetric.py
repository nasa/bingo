"""
This module encapsulates different fitness metrics that can be used for
symbolic regression in bingo
"""

import abc
import numpy as np
import warnings


class FitnessMetric(object, metaclass=abc.ABCMeta):
    """fitness metric superclass"""

    need_x = True  # required to be true at the moment
    need_dx_dt = False
    need_y = False

    @classmethod
    def evaluate_metric(cls, **kwargs):
        """ returns the fitness metric """
        return np.mean(np.abs(cls.evaluate_vector(**kwargs)))

    @staticmethod
    @abc.abstractmethod
    def evaluate_vector(**kwargs):
        """does the fitness calculation in vector form"""
        pass


class StandardRegression(FitnessMetric):
    """ Traditional fitness evaluation """

    need_y = True

    @staticmethod
    def evaluate_vector(indv, x, y):
        """
        :param x: independent variable
        :param f: test function evaluated at x
        :param y: target values of the test function

        :return f - y
        """
        f = indv.evaluate(x, StandardRegression, x=x, y=y)

        return (f - y).flatten()


class ImplicitRegression(FitnessMetric):
    """ Implicit Regression, version 2"""

    need_dx_dt = True

    @staticmethod
    def evaluate_vector(indv, x, dx_dt,
                        required_params=None,
                        normalize_dot=False):
        """
        Fitness of this metric is related cos of angle between between df_dx
        and dx_dt. Different normalization and erorr checking are available.

        :param x: independent variable
        :param df_dx: partial derivatives of the test function wrt x
        :param dx_dt: time derivative of x along trajectories
        :param required_params: minimum number of nonzero components of dot
        :param normalize_dot: normalize the terms in the dot product

        :return: the fitness for each row
        """
        f, df_dx = indv.evaluate_deriv(x, ImplicitRegression,
                                       x=x, dx_dt=dx_dt,
                                       required_params=required_params,
                                       normalize_dot=normalize_dot)

        if normalize_dot:
            dot = (df_dx/np.linalg.norm(df_dx, axis=1).reshape((-1, 1))) * \
                  (dx_dt/np.linalg.norm(dx_dt, axis=1).reshape((-1, 1)))
        else:
            dot = df_dx * dx_dt

        if required_params is not None:
            n_params_used = (abs(dot) > 1e-16).sum(1)
            enough_params_used = np.any(n_params_used >= required_params)
            if not enough_params_used:  # not enough parameters
                return np.full((x.shape[0],), np.inf)

        new = np.abs(np.sum(dot, axis=1)) / np.sum(np.abs(dot), axis=1)
        return new

    @classmethod
    def evaluate_metric(cls, **kwargs):
        """
        Fitness of this metric is related cos of angle between between df_dx
        and dx_dt. Different normalization and erorr checking are available.

        :param kwargs: dictionary of keyword args used in vector evaluation
        :return: the mean of the fitness vector, ignoring nans
        """
        vec = cls.evaluate_vector(**kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            err = np.nanmean(np.abs(vec))
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

    need_dx_dt = True

    @staticmethod
    def evaluate_vector(indv, x, dx_dt):
        """
        from schmidt and lipson's papers

        :param x: independent variable
        :param dx_dt: time derivative of x along trajectories

        :return: the fitness for each row
        """

        # NOTE: this doesnt work well right now
        #       importantly, it couldn't reproduce the papers

        f, df_dx = indv.evaluate_deriv(x, ImplicitRegressionTest,
                                       x=x, dx_dt=dx_dt)

        n_params = x.shape[1]
        # print("----------------------------------")
        worst_fit = 0
        diff_worst = np.full((x.shape[0], ), np.inf)
        for i in range(n_params):
            for j in range(n_params):
                if i != j:
                    df_dxi = np.copy(df_dx[:, i])
                    df_dxj = np.copy(df_dx[:, j])
                    dxi_dxj_2 = dx_dt[:, i]/dx_dt[:, j]
                    # print("independent:", i, "-", j)
                    for k in range(n_params):
                        if k != i and k != j:
                            # print("  dependent:", i, "-", k)
                            # df_dxi += df_dx[:, k]*dx_dt[:, k]/dx_dt[:, i]
                            df_dxj += df_dx[:, k]*dx_dt[:, k]/dx_dt[:, j]

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


class AtomicPotential(FitnessMetric):
    """ Implicit Regression, version from schmidt and lipson """

    need_y = True

    @staticmethod
    def evaluate_vector(indv, x, y):
        r_list = []
        config_lims = [0]
        for (structure, a, rcut), energy_true in zip(x, y):
            # make radius list
            natoms = structure.shape[0]
            rcutsq = rcut**2
            for atomi in range(0, natoms):
                xtmp = structure[atomi, 0]
                ytmp = structure[atomi, 1]
                ztmp = structure[atomi, 2]
                for atomj in range(atomi + 1, natoms):
                    delx = structure[atomj, 0] - xtmp
                    while delx > 0.5 * a:
                        delx -= a
                    while delx < -0.5 * a:
                        delx += a
                    dely = structure[atomj, 1] - ytmp
                    while dely > 0.5 * a:
                        dely -= a
                    while dely < -0.5 * a:
                        dely += a
                    delz = structure[atomj, 2] - ztmp
                    while delz > 0.5 * a:
                        delz -= a
                    while delz < -0.5 * a:
                        delz += a

                    rsq = delx * delx + dely * dely + delz * delz
                    if rsq <= rcutsq:
                        r_list.append(np.sqrt(rsq))
            config_lims.append(len(r_list))

        r_list = np.array(r_list).reshape([-1, 1])
        pair_energies = indv.evaluate(r_list,
                                      AtomicPotential,
                                      x=x, y=y).flatten()

        err_vec = []
        for i, energy_true in enumerate(y):
            energy = np.sum(pair_energies[config_lims[i]:config_lims[i+1]])
            err_vec.append(energy - energy_true)

        return np.array(err_vec).flatten()
