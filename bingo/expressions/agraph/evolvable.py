"""Evolvable wrapper around :class:`AGraphExpression`.

:class:`EvolvableExpression` adapts a standalone
:class:`~bingo.expressions.agraph.expression.AGraphExpression` so that it
satisfies the :class:`~bingo.chromosomes.chromosome.Chromosome` interface
required by bingo's evolutionary framework (``Island``, ``Archipelago``,
selection operators, etc.).
"""

import copy

from ...chromosomes.chromosome import Chromosome


class EvolvableExpression(Chromosome):
    """Evolutionary adapter for an AGraph expression.

    Parameters
    ----------
    expression : AGraphExpression
        The underlying expression object.
    genetic_age : int, optional
        Age of the oldest genetic material.  Default 0.
    fitness : numeric or None, optional
        Starting fitness value.
    fit_set : bool, optional
        Whether ``fitness`` has been set.

    Attributes
    ----------
    expression : AGraphExpression
    """

    def __init__(self, expression, genetic_age=0, fitness=None, fit_set=False):
        super().__init__(genetic_age=genetic_age, fitness=fitness, fit_set=fit_set)
        self.expression = expression

    # ------------------------------------------------------------------ #
    #  Chromosome abstract methods                                        #
    # ------------------------------------------------------------------ #

    def __str__(self):
        return str(self.expression)

    def distance(self, other):
        """Element-wise distance between command arrays.

        Parameters
        ----------
        other : EvolvableExpression

        Returns
        -------
        int
        """
        return self.expression.distance(other.expression)

    # ------------------------------------------------------------------ #
    #  Local optimization interface                                       #
    # ------------------------------------------------------------------ #

    def needs_local_optimization(self):
        """Whether the expression has un-optimised constants.

        Returns
        -------
        bool
        """
        return (
            len(self.expression.constants) > 0
            and not self.expression.__sklearn_is_fitted__()
        )

    def get_number_local_optimization_params(self):
        """Number of optimisable constants.

        Returns
        -------
        int
        """
        return len(self.expression.constants)

    def set_local_optimization_params(self, params):
        """Set the constant values.

        Parameters
        ----------
        params : array-like of float
        """
        self.expression.constants = tuple(float(p) for p in params)

    # ------------------------------------------------------------------ #
    #  Public expression view                                             #
    # ------------------------------------------------------------------ #

    @property
    def command_array(self):
        """Nx3 uint8 array: the simplified (evaluation-ready) command stack.

        This is the canonical public view of the expression.  Genetic
        operators must access the raw (pre-simplification) layer
        directly through ``individual.expression.raw_command_array`` and
        ``individual.expression.mutable_raw_command_array``.
        """
        return self.expression.command_array

    @property
    def complexity(self):
        """Number of utilized commands in the simplified stack."""
        return self.expression.complexity

    def get_utilized_commands(self):
        """Which raw commands are utilized by the output.

        Returns
        -------
        bytearray
        """
        return self.expression.get_utilized_commands()

    # ------------------------------------------------------------------ #
    #  Hash / equality                                                    #
    # ------------------------------------------------------------------ #

    def __hash__(self):
        return hash(self.expression)

    def __eq__(self, other):
        if not isinstance(other, EvolvableExpression):
            return NotImplemented
        return self.expression == other.expression

    # ------------------------------------------------------------------ #
    #  Copy / serialization                                               #
    # ------------------------------------------------------------------ #

    def copy(self):
        """Deep copy of the evolvable expression."""
        return copy.deepcopy(self)

    def __deepcopy__(self, memodict=None):
        new_expr = self.expression.copy()
        new = EvolvableExpression.__new__(EvolvableExpression)
        new.expression = new_expr
        new._genetic_age = self._genetic_age
        new._fitness = self._fitness
        new._fit_set = self._fit_set
        return new
