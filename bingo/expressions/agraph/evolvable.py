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
    #  Public expression facade                                           #
    # ------------------------------------------------------------------ #

    def predict(self, X, *, constants=None):
        """Predict target values for *X* with optional temporary constants."""
        if constants is None:
            return self.expression.predict(X)
        return self.expression.predict(X, constants=constants)

    def gradient(self, X):
        """Return predictions and their gradients with respect to *X*."""
        return self.expression.gradient(X)

    def fit(self, X, y, *, tolerance=1e-5):
        """Fit expression constants and return this evolutionary candidate."""
        self.expression.fit(X, y, tolerance=tolerance)
        self.fit_set = False
        return self

    def fit_implicit(self, X, dx_dt, *, tolerance=1e-5):
        """Fit implicit-expression constants and return this candidate."""
        self.expression.fit_implicit(X, dx_dt, tolerance=tolerance)
        self.fit_set = False
        return self

    def loss(self, X, y, *, kind="mse"):
        """Return the lower-is-better explicit-regression loss."""
        return self.expression.loss(X, y, kind=kind)

    def score(self, X, y, *, kind="r2"):
        """Return the higher-is-better explicit-regression score."""
        return self.expression.score(X, y, kind=kind)

    def implicit_loss(self, X, dx_dt, *, required_params=None):
        """Return the lower-is-better implicit-regression loss."""
        return self.expression.implicit_loss(
            X, dx_dt, required_params=required_params
        )

    def implicit_score(self, X, dx_dt, *, required_params=None):
        """Return the higher-is-better implicit-regression score."""
        return self.expression.implicit_score(
            X, dx_dt, required_params=required_params
        )

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

    @property
    def tree_complexity(self):
        """Tree-based node count (counts shared sub-expressions multiple times)."""
        return self.expression.tree_complexity

    @property
    def constants(self):
        """Numeric constants used in the simplified equation."""
        return self.expression.constants

    @property
    def integers(self):
        """Integer values used in the simplified equation."""
        return self.expression.integers

    @property
    def constant_mapping(self):
        """Map simplified constant indices to raw constant indices."""
        return self.expression.constant_mapping

    @property
    def is_fitted(self):
        """Whether fitting has been attempted for the current structure."""
        return self.expression.is_fitted

    @property
    def console(self):
        """Human-readable expression string."""
        return self.expression.console

    @property
    def sympy(self):
        """SymPy representation of the expression."""
        return self.expression.sympy

    @property
    def latex(self):
        """LaTeX representation of the expression."""
        return self.expression.latex

    def get_utilized_commands(self):
        """Which raw commands are utilized by the output.

        Returns
        -------
        bytearray
        """
        return self.expression.get_utilized_commands()

    def get_operator_counts(self, tree=True, terminals="exclude"):
        """Count operators in the expression tree or DAG."""
        return self.expression.get_operator_counts(tree=tree, terminals=terminals)

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
