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
        dot = (df_dx/np.linalg.norm(df_dx, axis=1).reshape((-1, 1))) * \
              (dx_dt/np.linalg.norm(dx_dt, axis=1).reshape((-1, 1)))
        n_params_used = np.count_nonzero(abs(dot) > 1e-16, axis=1)
        if np.any(n_params_used >= required_params):
            diff = np.log(1 + np.abs(np.sum(dot, axis=1)))
        else:  # not enough parameters in const regression
            diff = np.full((dot.shape[0], ), np.inf)
        return diff

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
