"""Generator of random AGraph expressions.

Creates random :class:`~bingo.expressions.agraph.evolvable.EvolvableExpression`
individuals whose command stacks are filled by a
:class:`ComponentGenerator`.
"""

import numpy as np

from bingo.chromosomes.generator import Generator

from . import get_expression_class
from .evolvable import EvolvableExpression
from .pyagraph import CONSTANT


class AGraphGenerator(Generator):
    """Generate random acyclic-graph individuals.

    Parameters
    ----------
    min_size : int
        Minimum command-array row count.
    max_size : int
        Maximum command-array row count.
    component_generator : ComponentGenerator
        Generates individual commands.
    simplification : {"reduce", "cas"}, optional
        Simplification strategy for generated expressions.  Default
        ``"cas"`` (full computer algebra simplification).
    random_state : int, numpy.random.Generator, or None, optional
        Seed or generator for reproducibility.  Default *None*.

    Raises
    ------
    ValueError
        If ``min_size < 1`` or ``max_size < min_size``.
    """

    def __init__(
        self,
        min_size,
        max_size,
        component_generator,
        simplification="cas",
        random_state=None,
    ):
        if min_size < 1:
            raise ValueError("min_size must be >= 1")
        if max_size < min_size:
            raise ValueError("max_size must be >= min_size")
        self._rng = np.random.default_rng(random_state)
        self.min_size = min_size
        self.max_size = max_size
        self.component_generator = component_generator
        self._simplification = simplification

    def __call__(self):
        """Generate a random :class:`EvolvableExpression`.

        Returns
        -------
        EvolvableExpression
        """
        size = int(self._rng.integers(self.min_size, self.max_size + 1))
        command_array = np.empty((size, 3), dtype=np.uint8)
        raw_constants = []
        for i in range(size):
            cmd = self.component_generator.random_command(i)
            if int(cmd[0]) == CONSTANT:
                idx = len(raw_constants)
                raw_constants.append(self.component_generator.random_constant_value())
                cmd[1] = cmd[2] = idx
            command_array[i] = cmd

        expr = get_expression_class()(simplification=self._simplification)
        expr.raw_command_array = command_array
        expr.raw_constants = tuple(raw_constants)
        return EvolvableExpression(expr)
