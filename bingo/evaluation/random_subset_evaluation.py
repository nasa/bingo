from copy import deepcopy
import numpy as np

from ..util.argument_validation import argument_validation
from .evaluation import Evaluation


class RandomSubsetEvaluation(Evaluation):
    @argument_validation(subset_size={">=": 1},)
    def __init__(
        self,
        fitness_function,
        subset_size,
        redundant=False,
        multiprocess=False,
    ):
        super().__init__(fitness_function, redundant, multiprocess)
        self._subset_size = subset_size
        self._full_training_data = deepcopy(fitness_function.training_data)

    def __call__(self, individual):
        subset = np.random.choice(
            len(self._full_training_data), self._subset_size, replace=False,
        )
        self.fitness_function.training_data = self._full_training_data[subset]
        super().__call__(individual)

