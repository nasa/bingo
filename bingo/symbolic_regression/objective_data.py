"""Aligned arrays used privately by expression regression objectives."""

import numpy as np


class ObjectiveData:
    """A collection of arrays that share a sample axis.

    Parameters
    ----------
    *arrays : array-like
        Arrays with equal lengths along their first axes.

    Raises
    ------
    TypeError
        If an array has no first axis.
    ValueError
        If the arrays have unequal first-axis lengths.
    """

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
        """Return a collection containing each aligned array indexed by ``items``.

        Parameters
        ----------
        items : int, slice, or array-like
            Index applied to every aligned array.

        Returns
        -------
        ObjectiveData
            The indexed aligned arrays. Integer indices retain a sample axis.
        """
        if isinstance(items, (int, np.integer)):
            items = slice(items, items + 1)
        return type(self)(*(array[items] for array in self._arrays))

    def __len__(self):
        """Return the number of aligned samples.

        Returns
        -------
        int
            The common first-axis length, or zero when no arrays were supplied.
        """
        return len(self._arrays[0]) if self._arrays else 0
