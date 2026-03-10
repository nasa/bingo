"""Component generator for AGraph expressions.

Generates random commands (or sub-components such as operators, terminals,
and their parameters) for building an AGraph command stack.
"""

import numpy as np

from .pyagraph.operators import (
    ARITY_2_IDS,
    VARIABLE,
    CONSTANT,
    OPERATOR_NAMES,
)
from .probability_mass_function import ProbabilityMassFunction


class ComponentGenerator:
    """Generate random commands for an AGraph command stack.

    Parameters
    ----------
    input_x_dimension : int
        Number of input variables (columns of *X*).
    num_initial_load_statements : int, optional
        Number of rows at the beginning of the stack that are forced to
        be terminals.  Default 1.
    terminal_probability : float, optional
        Probability that a randomly generated command is a terminal
        (as opposed to an operator).  Default 0.1.
    constant_probability : float or None, optional
        Probability that a terminal is a CONSTANT (vs VARIABLE).  When
        *None* (the default), the weight is ``1 : input_x_dimension``.
    constant_distribution : {"normal", "uniform"}, optional
        Distribution used when drawing a random initial value for a
        CONSTANT node.  ``"normal"`` samples from
        ``Normal(0, constant_scale)``; ``"uniform"`` samples uniformly
        from ``(-constant_scale, +constant_scale)``.  Default
        ``"normal"``.
    constant_scale : float, optional
        Scale parameter for the constant-value distribution.  Must be
        positive.  Default ``1.0``.
    random_state : int, numpy.random.Generator, or None, optional
        Seed or generator for reproducibility.  Default *None*
        (unseeded).

    Attributes
    ----------
    input_x_dimension : int
    """

    def __init__(
        self,
        input_x_dimension,
        num_initial_load_statements=1,
        terminal_probability=0.1,
        constant_probability=None,
        constant_distribution="normal",
        constant_scale=1.0,
        random_state=None,
    ):
        if input_x_dimension < 0:
            raise ValueError("input_x_dimension must be >= 0")
        if num_initial_load_statements < 1:
            raise ValueError("num_initial_load_statements must be >= 1")
        if not 0.0 <= terminal_probability <= 1.0:
            raise ValueError("terminal_probability must be in [0, 1]")
        if constant_distribution not in ("normal", "uniform"):
            raise ValueError("constant_distribution must be 'normal' or 'uniform'")
        if constant_scale <= 0:
            raise ValueError("constant_scale must be > 0")

        self._rng = np.random.default_rng(random_state)
        self.input_x_dimension = input_x_dimension
        self._num_initial_load_statements = num_initial_load_statements
        self._constant_distribution = constant_distribution
        self._constant_scale = constant_scale

        self._terminal_pmf = self._make_terminal_pmf(constant_probability)
        self._operator_pmf = ProbabilityMassFunction(self._rng)
        self._random_command_pmf = self._make_command_pmf(terminal_probability)

    # ------------------------------------------------------------------ #
    #  PMF construction                                                   #
    # ------------------------------------------------------------------ #

    def _make_terminal_pmf(self, constant_probability):
        if constant_probability is None:
            weights = [1, max(self.input_x_dimension, 1)]
        else:
            if not 0.0 <= constant_probability <= 1.0:
                raise ValueError("constant_probability must be in [0, 1]")
            weights = [constant_probability, 1.0 - constant_probability]
        return ProbabilityMassFunction(
            self._rng, items=[CONSTANT, VARIABLE], weights=weights
        )

    def _make_command_pmf(self, terminal_probability):
        return ProbabilityMassFunction(
            self._rng,
            items=[self.random_terminal_command, self.random_operator_command],
            weights=[terminal_probability, 1.0 - terminal_probability],
        )

    # ------------------------------------------------------------------ #
    #  Operator management                                                #
    # ------------------------------------------------------------------ #

    def add_operator(self, operator, weight=None):
        """Add an operator to the set of possible operators.

        Parameters
        ----------
        operator : int or str
            Operator ID (e.g. 3) or a recognised name / symbol
            (e.g. ``"+"``, ``"addition"``).
        weight : float or None, optional
            Relative weight.  *None* uses the current average weight.
        """
        if isinstance(operator, str):
            operator = self._operator_from_string(operator)
        self._operator_pmf.add_item(operator, weight)

    @staticmethod
    def _operator_from_string(name):
        for op_id, names in OPERATOR_NAMES.items():
            if name in names:
                return op_id
        raise ValueError(f"Unknown operator name: {name!r}")

    # ------------------------------------------------------------------ #
    #  Random command generation                                          #
    # ------------------------------------------------------------------ #

    def random_command(self, stack_location):
        """Generate a random command for *stack_location*.

        Early rows (< ``num_initial_load_statements``) are always
        terminals.

        Parameters
        ----------
        stack_location : int

        Returns
        -------
        numpy array of uint8, shape (3,)
            ``[operator, param1, param2]``
        """
        if stack_location < self._num_initial_load_statements:
            return self.random_terminal_command(stack_location)
        return self._random_command_pmf.draw_sample()(stack_location)

    def random_operator_command(self, stack_location):
        """Generate a random operator (non-terminal) command.

        Parameters
        ----------
        stack_location : int

        Returns
        -------
        numpy array of uint8, shape (3,)
        """

        op = self.random_operator()
        p1 = self.random_operator_parameter(stack_location)
        p2 = self.random_operator_parameter(stack_location) if op in ARITY_2_IDS else p1

        return np.array(
            [
                op,
                p1,
                p2,
            ],
            dtype=np.uint8,
        )

    def random_terminal_command(self, _stack_location=None):
        """Generate a random terminal command.

        Returns
        -------
        numpy array of uint8, shape (3,)
        """
        terminal = self.random_terminal()
        param = self.random_terminal_parameter(terminal)
        return np.array([terminal, param, param], dtype=np.uint8)

    # ------------------------------------------------------------------ #
    #  Component-level helpers                                            #
    # ------------------------------------------------------------------ #

    def random_operator(self):
        """Draw a random operator from the configured set.

        Returns
        -------
        int
        """
        return self._operator_pmf.draw_sample()

    def random_operator_parameter(self, stack_location):
        """Random parameter for an operator (index < *stack_location*).

        Parameters
        ----------
        stack_location : int

        Returns
        -------
        int
        """
        return int(self._rng.integers(stack_location))

    def random_terminal(self):
        """Draw a random terminal type (VARIABLE or CONSTANT).

        Returns
        -------
        int
        """
        return self._terminal_pmf.draw_sample()

    def random_terminal_parameter(self, terminal_type):
        """Random parameter for a terminal.

        For VARIABLE, returns a random variable index.  For CONSTANT,
        returns a placeholder index (``0``); callers that need a real
        constant value should call :meth:`random_constant_value` and
        insert the result into ``raw_constants`` themselves.

        Parameters
        ----------
        terminal_type : int
            VARIABLE or CONSTANT.

        Returns
        -------
        int
        """
        if terminal_type == VARIABLE:
            return int(self._rng.integers(max(self.input_x_dimension, 1)))
        # CONSTANT — placeholder index 0; caller assigns real index
        return 0

    def random_constant_value(self):
        """Draw a random initial numeric value for a CONSTANT node.

        The value is sampled from the distribution specified at
        construction time:

        * ``"normal"``  → ``Normal(0, constant_scale)``
        * ``"uniform"`` → ``Uniform(-constant_scale, +constant_scale)``

        Returns
        -------
        float
        """
        if self._constant_distribution == "normal":
            return float(self._rng.normal(0.0, self._constant_scale))
        return float(self._rng.uniform(-self._constant_scale, self._constant_scale))

    # ------------------------------------------------------------------ #
    #  Introspection                                                      #
    # ------------------------------------------------------------------ #

    def get_number_of_terminals(self):
        """Number of distinct terminal types.

        Returns
        -------
        int
        """
        return len(self._terminal_pmf.items)

    def get_number_of_operators(self):
        """Number of distinct operator types.

        Returns
        -------
        int
        """
        return len(self._operator_pmf.items)
