"""
This module contains most of the code necessary for the representation of an
fitness predictor in the form of a subsampling list for real data
"""
import numpy as np


class FPManipulator(object):
    """
    manipulates AGraph objects for generation, crossover, mutation,
    and distance
    """

    def __init__(self, size, max_index):
        self.size = size
        self.max_index = max_index

    def generate(self):
        """generate random individual"""
        indices = np.random.randint(0, self.max_index, self.size).tolist()
        return FitnessPredictor(indices)

    def crossover(self, parent1, parent2):
        """single point crossover, returns 2 new individuals"""
        cx_point = np.random.randint(1, self.size)
        child1 = parent1.copy()
        child2 = parent2.copy()
        child1.indices[cx_point:] = parent2.indices[cx_point:]
        child2.indices[cx_point:] = parent1.indices[cx_point:]
        child1.fitness = None
        child2.fitness = None
        return child1, child2

    def mutation(self, indv):
        """performs 1pt mutation, does not create copy of indv"""
        mut_point = np.random.randint(self.size)
        indv.indices[mut_point] = np.random.randint(self.max_index)
        indv.fitness = None
        return indv

    @staticmethod
    def distance(indv1, indv2):
        """
        Calculates the distance between indv1 and indv2.
        """
        ind2 = list(indv2.indices)
        # remove matching values
        for i in indv1.indices:
            if i in ind2:
                ind2.remove(i)

        return len(ind2)

    @staticmethod
    def dump(indv):
        """
        dumps indv to picklable object
        """
        return indv.indices

    @staticmethod
    def load(indices):
        """
        loads indv from picklable object
        """
        return FitnessPredictor(indices)


class FitnessPredictor(object):
    """
    class for fitness predictor, mainly just a list of indices
    """
    def __init__(self, indices=None):
        if indices is None:
            self.indices = []
        else:
            self.indices = indices
        self.fitness = None

    def copy(self):
        """duplicates a fitness predictor via deep copy"""
        dup = FitnessPredictor(list(self.indices))
        dup. fitness = self.fitness
        return dup

    def __str__(self):
        return str(self.indices)

    def fit_func(self, indv, X, Y, standard_regression, required_params=2):
        """fitness function for standard regression type"""
        try:
            # standard symbolic regression
            if standard_regression:
                err_vec = np.abs(indv.evaluate(X[self.indices, :])
                                 - Y[self.indices])
                err = np.mean(err_vec)

            # regression to find constant combinations/laws
            else:
                df_dx = indv.evaluate_deriv(X[self.indices, :])
                dot = df_dx * Y[self.indices, :]
                n_params_used = np.count_nonzero(abs(dot) > 1e-16, axis=1)
                if np.any(n_params_used >= required_params):
                    err_vec = np.log(1 + np.abs(np.sum(dot, axis=1) /
                                                np.linalg.norm(df_dx, axis=1)))
                    err = np.mean(err_vec)
                else:
                    err = np.inf
        except (OverflowError, FloatingPointError, ValueError):
            print("fit_func error")
            err = np.nan

        return err
