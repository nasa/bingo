"""Lightweight probability mass function using numpy's Generator API.

This is a local reimplementation of the bingo class that accepts an explicit
:class:`numpy.random.Generator` rather than relying on global
``numpy.random`` state.
"""

import numpy as np


class ProbabilityMassFunction:
    """A probability mass function that draws samples via an explicit RNG.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator used for drawing samples.
    items : list, optional
        Initial items.
    weights : list-like of float, optional
        Relative weights (need not sum to 1). Default is equal weights.

    Attributes
    ----------
    items : list
    """

    def __init__(self, rng, items=None, weights=None):
        self._rng = rng
        self.items = list(items) if items is not None else []

        if weights is None:
            n = len(self.items)
            weights = np.ones(n) / n if n > 0 else np.array([])
        else:
            weights = np.asarray(weights, dtype=float)

        if len(weights) != len(self.items):
            raise ValueError(
                f"items ({len(self.items)}) and weights ({len(weights)}) "
                "must have the same length"
            )

        self._cumulative = self._build_cumulative(weights)

    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_cumulative(weights):
        if weights.size == 0:
            return np.array([])
        total = weights.sum()
        if total <= 0 or np.any(weights < 0):
            raise ValueError(
                f"Weights must be non-negative with positive sum; got {weights}"
            )
        return np.cumsum(weights / total)

    # ------------------------------------------------------------------ #

    def add_item(self, item, weight=None):
        """Add an item to the PMF.

        Parameters
        ----------
        item
            The item to add.
        weight : float or None
            Relative weight.  *None* uses the current average weight.
        """
        if weight is None:
            weight = 1.0 / len(self.items) if self.items else 1.0

        self.items.append(item)

        # Reconstruct raw (unnormalised) weights and append
        if self._cumulative.size > 0:
            raw = np.diff(np.concatenate([[0.0], self._cumulative]))
        else:
            raw = np.array([])

        raw = np.append(raw, weight)
        self._cumulative = self._build_cumulative(raw)

    def draw_sample(self):
        """Draw a single item according to the stored weights.

        Returns
        -------
        item
            One of the stored items.
        """
        u = self._rng.random()
        idx = int(np.searchsorted(self._cumulative, u))
        # Clamp in case of floating-point edge cases
        idx = min(idx, len(self.items) - 1)
        return self.items[idx]
