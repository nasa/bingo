"""Single-point crossover for variable-size AGraph individuals.

Supports parents of different sizes.  Parent constant and integer values
are always carried into the children and remapped to the correct
positions.  Crossover points are chosen so that both children remain
within the configured ``[min_size, max_size]`` bounds."""

import numpy as np

from .pyagraph import CONSTANT, INTEGER, TERMINAL_IDS
from bingo.chromosomes.crossover import Crossover


class AGraphCrossover(Crossover):
    """Single-point crossover between AGraph individuals.

    Parents may have different stack sizes.  Crossover points are chosen
    so that both resulting children satisfy
    ``min_size <= child_size <= max_size``.

    Parameters
    ----------
    min_size : int
        Minimum allowed command-array row count for children.
    max_size : int
        Maximum allowed command-array row count for children.
    random_state : int, numpy.random.Generator, or None, optional
        Seed or generator for reproducibility.  Default *None*.

    Attributes
    ----------
    types : list of str
    last_crossover_types : tuple(str or None, str or None)
    """

    def __init__(self, min_size, max_size, random_state=None):
        if min_size < 1:
            raise ValueError("min_size must be >= 1")
        if max_size < min_size:
            raise ValueError("max_size must be >= min_size")
        self._rng = np.random.default_rng(random_state)
        self._min_size = min_size
        self._max_size = max_size
        self.types = ["default"]
        self.last_crossover_types = (None, None)

    # ------------------------------------------------------------------ #

    def __call__(self, parent_1, parent_2):
        """Perform single-point crossover.

        Parameters
        ----------
        parent_1 : EvolvableExpression
        parent_2 : EvolvableExpression

        Returns
        -------
        tuple(EvolvableExpression, EvolvableExpression)
        """
        child_1 = parent_1.copy()
        child_2 = parent_2.copy()

        p1_stack = parent_1.expression.raw_command_array
        p2_stack = parent_2.expression.raw_command_array

        n1 = p1_stack.shape[0]
        n2 = p2_stack.shape[0]

        # Pick crossover points constrained so children stay in [min, max]
        cp1, cp2 = self._pick_crossover_points(n1, n2)

        # Build child stacks by swapping tails
        c1_stack = self._build_child_stack(p1_stack, p2_stack, cp1, cp2)
        c2_stack = self._build_child_stack(p2_stack, p1_stack, cp2, cp1)

        # Carry parent constants and integers into children
        c1_consts, c1_ints = self._renumber_and_merge_constants(
            parent_1, parent_2, c1_stack, cp1
        )
        c2_consts, c2_ints = self._renumber_and_merge_constants(
            parent_2, parent_1, c2_stack, cp2
        )

        # Apply to children
        child_1.expression.raw_command_array = c1_stack
        child_1.expression.raw_integers = c1_ints
        child_1.expression.raw_constants = c1_consts

        child_2.expression.raw_command_array = c2_stack
        child_2.expression.raw_integers = c2_ints
        child_2.expression.raw_constants = c2_consts

        # Genetic age
        child_age = max(parent_1.genetic_age, parent_2.genetic_age)
        child_1.genetic_age = child_age
        child_2.genetic_age = child_age
        child_1.fit_set = False
        child_2.fit_set = False

        self.last_crossover_types = ("default", "default")
        return child_1, child_2

    # ------------------------------------------------------------------ #
    #  Crossover point selection                                         #
    # ------------------------------------------------------------------ #

    def _pick_crossover_points(self, n1, n2):
        """Choose cp1, cp2 so both children stay within [min_size, max_size].

        Child 1 size = cp1 + (n2 - cp2)
        Child 2 size = cp2 + (n1 - cp1)

        For a given cp1 the valid cp2 range is::

            cp2_lo = max(1, n2 + cp1 - max_size, min_size + cp1 - n1)
            cp2_hi = min(n2, n2 + cp1 - min_size, max_size + cp1 - n1)

        We compute the range of cp1 values that admit a non-empty cp2
        range, pick cp1 uniformly from that range, then pick cp2
        uniformly from the valid range.
        """
        lo = self._min_size
        hi = self._max_size
        cp1_lo = max(1, lo + 1 - n2, n1 + 1 - hi)
        cp1_hi = min(n1, n1 + n2 - lo, hi)

        if cp1_lo > cp1_hi:
            # Fallback (should not happen when parents are within bounds)
            return (
                int(self._rng.integers(1, max(n1, 2))),
                int(self._rng.integers(1, max(n2, 2))),
            )

        cp1 = int(self._rng.integers(cp1_lo, cp1_hi + 1))
        cp2_lo = max(1, n2 + cp1 - hi, lo + cp1 - n1)
        cp2_hi = min(n2, n2 + cp1 - lo, hi + cp1 - n1)
        cp2 = int(self._rng.integers(cp2_lo, cp2_hi + 1))

        return cp1, cp2

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_child_stack(head_stack, tail_stack, head_cp, tail_cp):
        """Concatenate head[:head_cp] + tail[tail_cp:] with reference fix-up.

        Operator parameters in the tail that reference rows >= tail_cp
        are shifted by ``head_cp - tail_cp`` so they point to the correct
        row in the new combined stack.  References into the tail's head
        (< tail_cp) are clamped to valid head rows.
        """
        head = head_stack[:head_cp].copy()
        tail = tail_stack[tail_cp:].copy()
        shift = head_cp - tail_cp

        for i in range(tail.shape[0]):
            node = int(tail[i, 0])
            if node not in TERMINAL_IDS:
                for col in (1, 2):
                    old_ref = int(tail[i, col])
                    new_ref = old_ref + shift
                    # Clamp to valid range [0, head_cp + i - 1]
                    new_ref = max(0, min(new_ref, head_cp + i - 1))
                    tail[i, col] = new_ref

        return np.vstack([head, tail]).astype(np.uint8)

    @staticmethod
    def _renumber_and_merge_constants(head_parent, tail_parent, child_stack, head_cp):
        """Carry parent constant/integer values into the child and remap indices.

        Scans every row once, handling both CONSTANT and INTEGER nodes
        in a single pass.  Rows before ``head_cp`` source from
        ``head_parent``; rows from ``head_cp`` onward source from
        ``tail_parent``.  Each encountered terminal is appended to a new
        list and its index in the child stack is rewritten to the new
        sequential position.

        Returns
        -------
        tuple
            ``(constants_tuple, integers_tuple)``
        """
        head_consts = list(head_parent.expression.raw_constants)
        tail_consts = list(tail_parent.expression.raw_constants)
        head_ints = list(head_parent.expression.raw_integers)
        tail_ints = list(tail_parent.expression.raw_integers)

        new_consts = []
        new_ints = []
        for row in range(child_stack.shape[0]):
            op = int(child_stack[row, 0])
            if op == CONSTANT:
                src = head_consts if row < head_cp else tail_consts
                idx = int(child_stack[row, 1])
                new_consts.append(src[idx] if idx < len(src) else 0.0)
                child_stack[row, 1:] = len(new_consts) - 1
            elif op == INTEGER:
                src = head_ints if row < head_cp else tail_ints
                idx = int(child_stack[row, 1])
                new_ints.append(src[idx] if idx < len(src) else 0)
                child_stack[row, 1:] = len(new_ints) - 1

        return tuple(new_consts), tuple(new_ints)
