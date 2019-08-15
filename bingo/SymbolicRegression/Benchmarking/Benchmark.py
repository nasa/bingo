"""This module defines a Benchmark for bingo.
Benchmarks are intended to measure the performance of bingo
on common symbolic regression example problems.
"""
import numpy as np

from ..ExplicitRegression import ExplicitTrainingData

class Benchmark:
    """ The class containing the information required to run a benchmark

    Parameters
    ----------
    name : string
        The name of the benchmark
    objective_function : string
        The target function of the benchmark
    train_set : list of ints 
        [start, stop, number_of_points] of the desired training set
    has_test_set : bool
        True if benchmark has a test set, false otherwise
    """
    def __init__(self, name, objective_function, train_set, has_test_set=False):
        self.name = name
        self.objective_function = objective_function
        self._has_test_set = has_test_set
        self.train_set = train_set
        self.test_set = []
        self.make_training_data()

    def equation_eval(self, x):
        """Evaluates the target function of a benchmark at some number x

        Parameters
        ----------
        x : float
            The input variable at which to evaluate the target function
        """
        return eval(self.objective_function)

    def make_training_data(self):
        """Makes an ExplicitTrainingData object for training data
        """
        np.random.seed(42)
        start = self.train_set[0]
        stop = self.train_set[1]
        num_points = self.train_set[2]
        x = self._init_x_vals(start, stop, num_points)
        y = self.equation_eval(x)
        noise = np.random.normal(0, 0.1, num_points)
        noise = noise.reshape((-1, 1))
        y = np.add(y, noise)
        data = ExplicitTrainingData(x, y)
        if self._has_test_set:
            self._make_test_set(data, 0.2)
        else:
            self.train_set = data
            self.test_set = []

    def _make_test_set(self, data, test_ratio):
        shuffled_indices = np.random.permutation(len(data))
        test_set_size = int(len(data) * test_ratio)
        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]
        self.train_set = data.__getitem__(train_indices)
        self.test_set = data.__getitem__(test_indices)

    def _init_x_vals(self, start, stop, num_points):
        return np.linspace(start, stop, num_points).reshape([-1, 1])
