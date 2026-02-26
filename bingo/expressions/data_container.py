"""Data container for expression fitting and scoring.

A thin wrapper around numpy arrays that holds feature data (``x``)
and target data (``y``), with automatic reshaping and validation.
"""

import numpy as np


class DataContainer:
    """Container for training/test data.

    Parameters
    ----------
    x : array-like
        Feature data. If 1-D, reshaped to a column vector.
    y : array-like
        Target data. If 1-D, reshaped to a column vector.

    Raises
    ------
    ValueError
        If the number of rows in ``x`` and ``y`` don't match.
    """

    def __init__(self, x, y):
        self.x = np.atleast_2d(np.asarray(x, dtype=float))
        self.y = np.atleast_2d(np.asarray(y, dtype=float))
        # If a 1-D array was given, atleast_2d makes it (1, N).
        # We want (N, 1) instead.
        if self.x.shape[0] == 1 and self.x.ndim == 2 and len(np.asarray(x).shape) == 1:
            self.x = self.x.T
        if self.y.shape[0] == 1 and self.y.ndim == 2 and len(np.asarray(y).shape) == 1:
            self.y = self.y.T
        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError(
                f"Number of rows in x ({self.x.shape[0]}) and y "
                f"({self.y.shape[0]}) must match."
            )

    def __getitem__(self, idx):
        """Row-slice the data container."""
        return DataContainer(self.x[idx], self.y[idx])

    def __len__(self):
        """Number of data points."""
        return self.x.shape[0]
