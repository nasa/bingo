"""
This module encapsulates different fitness metrics that can be used
"""

import abc
import numpy as np


class FitnessMetric(object, metaclass=abc.ABCMeta):
    """fitness metric superclass"""

    need_x = True  # required to be true at the moment
    need_dx_dt = False
    need_f = False
    need_df_dx = False
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

    need_f = True
    need_y = True

    @staticmethod
    def evaluate_vector(x, f, y):
        return f - y


class ImplicitRegression(FitnessMetric):
    """ Implicit Regression, version """

    need_df_dx = True
    need_dx_dt = True

    @staticmethod
    def evaluate_vector(x, df_dx, dx_dt, required_params=2):
        """ error = cos of angle between between df_dx and dx_dt """
        dot = df_dx * dx_dt
        n_params_used = np.count_nonzero(abs(dot) > 1e-16, axis=1)
        if np.any(n_params_used >= required_params):
            diff = np.log(1 + np.abs(np.sum(dot, axis=1)))
        else:  # not enough parameters in const regression
            diff = np.full((x.shape[0], ), np.inf)
        return diff

    @classmethod
    def evaluate_metric(cls, x, df_dx, dx_dt, required_params=2):
        """ evaluate metric but ignore some nans in vector
            error = cos of angle between between df_dx and dx_dt """
        dot = (df_dx/np.linalg.norm(df_dx, axis=1).reshape((-1, 1))) * \
              (dx_dt/np.linalg.norm(dx_dt, axis=1).reshape((-1, 1)))
        n_params_used = np.count_nonzero(abs(dot) > 1e-16, axis=1)
        if np.any(n_params_used >= required_params):
            vec = np.log(1 + np.abs(np.sum(dot, axis=1)))
        else:  # not enough parameters in const regression
            vec = np.full((x.shape[0], ), np.inf)
        nan_count = np.count_nonzero(np.isnan(vec))
        # ok
        if nan_count < 0.1 * len(vec):
            return np.nanmean(np.abs(vec))
        # too many nans
        else:
            return np.nan


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

    need_df_dx = True
    need_dx_dt = True

    @staticmethod
    def evaluate_vector(x, df_dx, dx_dt):
        """ from schmidt and lipson's papers """

        # NOTE: this doesnt work right now, and couldn't reproduce the papers
        n_params = x.shape[1]

        worst_fit = 0
        diff_worst = np.full((x.shape[0], ), np.inf)
        for i in range(n_params):
            for j in range(n_params):
                if i != j:
                    df_dxi = np.copy(df_dx[:, i])
                    df_dxj = np.copy(df_dx[:, i])
                    dxi_dxj_2 = dx_dt[:, i]/dx_dt[:, j]
                    for k in range(n_params):
                        if k != i and k != j:
                            df_dxi += df_dx[:, k]*dx_dt[:, k]/dx_dt[:, i]
                            df_dxj += df_dx[:, k]*dx_dt[:, k]/dx_dt[:, j]

                    dxi_dxj_1 = df_dxj / df_dxi
                    diff = np.log(1. + np.abs(dxi_dxj_1 - dxi_dxj_2))
                    fit = np.mean(diff)
                    if np.isfinite(fit) and fit > worst_fit:
                        # print(i, j)
                        diff_worst = np.copy(diff)
                        worst_fit = fit
        # print(diff_worst)
        return diff_worst

    @classmethod
    def evaluate_metric(cls, **kwargs):
        """ evaluate metric but ignore some nans in vector"""
        vec = cls.evaluate_vector(**kwargs)
        nan_count = np.count_nonzero(np.isnan(vec))
        # ok
        if nan_count < 0.1 * len(vec):
            return np.nanmean(np.abs(vec))
        # too many nans
        else:
            return np.nan
