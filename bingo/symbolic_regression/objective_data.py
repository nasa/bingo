"""Aligned arrays used privately by expression regression objectives."""

import numpy as np


class ObjectiveData:
    """A collection of arrays that share a sample axis."""

    def __init__(self, *arrays):
        self._arrays = tuple(np.asarray(array) for array in arrays)
        if not self._arrays:
            return
        try:
            lengths = {array.shape[0] for array in self._arrays}
        except IndexError as error:
            raise TypeError("Objective data arrays must have a first axis") from error
        if len(lengths) != 1:
            raise ValueError("Objective data arrays must have equal first-axis length")

    @property
    def arrays(self):
        """The aligned arrays in their construction order."""
        return self._arrays

    def __getitem__(self, items):
        if isinstance(items, (int, np.integer)):
            items = slice(items, items + 1)
        return type(self)(*(array[items] for array in self._arrays))

    def __len__(self):
        return len(self._arrays[0]) if self._arrays else 0
