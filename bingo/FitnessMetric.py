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