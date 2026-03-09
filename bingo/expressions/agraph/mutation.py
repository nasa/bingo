"""Mutation of AGraph individuals.

Five mutation strategies, selected via a probability mass function:

* **command** — replace an entire utilized row with a random command.
* **node** — swap the operator/terminal type while keeping parameters
  consistent.
* **parameter** — randomise the parameters of a utilized row.
* **prune** — remove an operator by redirecting references to one of
  its children.
* **fork** — wrap an existing row in a new operator sub-tree,
  using unutilized rows as budget.
"""

import numpy as np

from .pyagraph import (
    CONSTANT,
    INTEGER,
    VARIABLE,
    ARITY_2_IDS,
    TERMINAL_IDS,
)
from ...chromosomes.mutation import Mutation
from .probability_mass_function import ProbabilityMassFunction

COMMAND_MUTATION = "command"
NODE_MUTATION = "node"
PARAMETER_MUTATION = "parameter"
PRUNE_MUTATION = "prune"
FORK_MUTATION = "fork"

DEFAULT_MAX_FORK_SIZE = 3


# ====================================================================== #
#  Stack compaction helpers (module-level, no instance state needed)       #
# ====================================================================== #


def _compact_stack_forward(
    new_stack, fork_size, fork_target, slots_before, slots_between
):
    """Move unutilized rows from before *fork_target* to just after it.

    Utilized rows between the freed slots and *fork_target* are shifted
    down (toward index 0) to close the gaps.  All operator references
    from the first freed slot onward are renumbered.

    Parameters
    ----------
    new_stack : numpy.ndarray
        Mutable raw command array (modified in-place).
    fork_size : int
        Total number of free slots required.
    fork_target : int
        Current index of the fork target row.
    slots_before : list of int
        Sorted unutilized indices that precede *fork_target*.
    slots_between : list of int
        Sorted unutilized indices already between *fork_target* and
        the fork reference.

    Returns
    -------
    fork_target_new : int
        Updated index of the fork target after compaction.
    new_slots_between : list of int
        Updated list of free slots between the (new) fork target and
        the fork reference.
    """
    needed_slots = fork_size - len(slots_between)
    moved_slots = slots_before
    if needed_slots < len(slots_before):
        moved_slots = slots_before[-needed_slots:]
    moved_slots_set = set(moved_slots)
    first_slot = moved_slots[0]

    new_spot_map = {}
    dest = first_slot
    for src in range(first_slot + 1, fork_target + 1):
        if src not in moved_slots_set:
            if src != dest:
                new_stack[dest] = new_stack[src]
                new_spot_map[src] = dest
            dest += 1

    if new_spot_map:
        for i in range(first_slot + 1, len(new_stack)):
            if new_stack[i, 0] not in TERMINAL_IDS:
                p1, p2 = int(new_stack[i, 1]), int(new_stack[i, 2])
                new_stack[i, 1] = new_spot_map.get(p1, p1)
                new_stack[i, 2] = new_spot_map.get(p2, p2)

    new_slots_between = list(range(dest, fork_target + 1)) + slots_between
    fork_target_new = new_spot_map.get(fork_target, fork_target)
    return fork_target_new, new_slots_between


def _compact_stack_backward(new_stack, fork_size, fork_ref, slots_between, slots_after):
    """Move unutilized rows from after *fork_ref* to just before it.

    Utilized rows between *fork_ref* and the freed slots are shifted
    up (toward higher indices) to close the gaps.  All operator
    references from the compaction region onward are renumbered.

    Parameters
    ----------
    new_stack : numpy.ndarray
        Mutable raw command array (modified in-place).
    fork_size : int
        Total number of free slots required.
    fork_ref : int
        Current index of the fork reference row.
    slots_between : list of int
        Sorted unutilized indices already between the fork target and
        *fork_ref*.
    slots_after : list of int
        Sorted unutilized indices that follow *fork_ref*.

    Returns
    -------
    fork_ref_new : int
        Updated index of the fork reference after compaction.
    new_slots_between : list of int
        Updated list of free slots between the fork target and the
        (new) fork reference.
    """
    needed_slots = fork_size - len(slots_between)
    moved_slots = slots_after[:needed_slots]
    moved_slots_set = set(moved_slots)
    last_slot = moved_slots[-1]

    new_spot_map = {}
    dest = last_slot
    for src in range(last_slot, fork_ref - 1, -1):
        if src not in moved_slots_set:
            if src != dest:
                new_stack[dest] = new_stack[src]
                new_spot_map[src] = dest
            dest -= 1

    if new_spot_map:
        for i in range(dest, len(new_stack)):
            if new_stack[i, 0] not in TERMINAL_IDS:
                p1, p2 = int(new_stack[i, 1]), int(new_stack[i, 2])
                new_stack[i, 1] = new_spot_map.get(p1, p1)
                new_stack[i, 2] = new_spot_map.get(p2, p2)

    new_slots_between = slots_between + list(range(fork_ref, dest + 1))
    fork_ref_new = new_spot_map.get(fork_ref, fork_ref)
    return fork_ref_new, new_slots_between


class AGraphMutation(Mutation):
    """Mutation of acyclic-graph individuals.

    Parameters
    ----------
    component_generator : ComponentGenerator
        Generates random commands / sub-components.
    command_probability : float, optional
    node_probability : float, optional
    parameter_probability : float, optional
    prune_probability : float, optional
    fork_probability : float, optional
    random_state : int, numpy.random.Generator, or None, optional
        Seed or generator for reproducibility.  Default *None*.

    Attributes
    ----------
    types : list of str
    last_mutation_type : str or None
    """

    def __init__(
        self,
        component_generator,
        command_probability=0.2,
        node_probability=0.2,
        parameter_probability=0.2,
        prune_probability=0.2,
        fork_probability=0.2,
        random_state=None,
    ):
        self._rng = np.random.default_rng(random_state)
        self._cgen = component_generator
        self._mutation_pmf = ProbabilityMassFunction(
            self._rng,
            [
                self._mutate_command,
                self._mutate_node,
                self._mutate_parameters,
                self._prune_branch,
                self._fork_mutation,
            ],
            [
                command_probability,
                node_probability,
                parameter_probability,
                prune_probability,
                fork_probability,
            ],
        )
        self.last_mutation_type = None
        self.types = [
            COMMAND_MUTATION,
            NODE_MUTATION,
            PARAMETER_MUTATION,
            PRUNE_MUTATION,
            FORK_MUTATION,
        ]

    # ------------------------------------------------------------------ #
    #  Public interface                                                   #
    # ------------------------------------------------------------------ #

    def __call__(self, parent):
        """Mutate *parent* and return a child.

        Parameters
        ----------
        parent : EvolvableExpression

        Returns
        -------
        EvolvableExpression
        """
        child = parent.copy()
        mutation_fn = self._mutation_pmf.draw_sample()
        mutation_fn(child)
        self._prune_raw_constants(child)
        child.fit_set = False
        return child

    # ------------------------------------------------------------------ #
    #  Command mutation                                                   #
    # ------------------------------------------------------------------ #

    def _mutate_command(self, individual):
        self.last_mutation_type = COMMAND_MUTATION
        loc = self._random_utilized_location(individual)
        raw = individual.expression.raw_command_array
        old = raw[loc].copy()
        new = self._cgen.random_command(loc)
        # Avoid no-op; CONSTANT->CONSTANT is always accepted (new value drawn below)
        attempts = 0
        while np.array_equal(new, old) and int(new[0]) != CONSTANT:
            new = self._cgen.random_command(loc)
            attempts += 1
            if attempts > 100:
                break
        self._append_constant(individual, new)
        individual.expression.mutable_raw_command_array[loc] = new

    # ------------------------------------------------------------------ #
    #  Node mutation                                                      #
    # ------------------------------------------------------------------ #

    def _mutate_node(self, individual):
        self.last_mutation_type = NODE_MUTATION
        loc = self._random_node_mutation_location(individual)
        raw = individual.expression.raw_command_array
        old_cmd = raw[loc].copy()
        new_cmd = old_cmd.copy()
        attempts = 0
        while old_cmd[0] == new_cmd[0]:
            self._randomize_node(new_cmd)
            attempts += 1
            if attempts > 100:
                break
        self._append_constant(individual, new_cmd)
        individual.expression.mutable_raw_command_array[loc] = new_cmd

    def _random_node_mutation_location(self, individual):
        utilized = individual.get_utilized_commands()
        raw = individual.expression.raw_command_array
        terminals_ok = self._cgen.get_number_of_terminals() > 1
        operators_ok = self._cgen.get_number_of_operators() > 1
        indices = []
        for i, (u, node) in enumerate(zip(utilized, raw[:, 0])):
            if u:
                if (node in TERMINAL_IDS and terminals_ok) or (
                    node not in TERMINAL_IDS and operators_ok
                ):
                    indices.append(i)
        if not indices:
            # Fallback — pick any utilized
            indices = [i for i, u in enumerate(utilized) if u]
        return indices[int(self._rng.integers(len(indices)))]

    def _randomize_node(self, command):
        if command[0] in TERMINAL_IDS:
            command[0] = self._cgen.random_terminal()
            command[1] = self._cgen.random_terminal_parameter(command[0])
            command[2] = command[1]
        else:
            command[0] = self._cgen.random_operator()
            # Fixup params if arity changed
            if command[0] not in ARITY_2_IDS:
                command[2] = command[1]

    # ------------------------------------------------------------------ #
    #  Parameter mutation                                                 #
    # ------------------------------------------------------------------ #

    def _mutate_parameters(self, individual):
        self.last_mutation_type = PARAMETER_MUTATION
        loc = self._random_param_location(individual)
        if loc is None:
            return
        raw = individual.expression.raw_command_array
        cmd = raw[loc]
        if int(cmd[0]) == CONSTANT:
            # Replace the constant's numeric value at its existing index
            idx = int(cmd[1])
            new_val = self._cgen.random_constant_value()
            consts = list(individual.expression.raw_constants)
            # Ensure the list is large enough (guard against stale indices)
            while idx >= len(consts):
                consts.append(0.0)
            consts[idx] = new_val
            individual.expression.raw_constants = tuple(consts)
            return
        old_cmd = cmd.copy()
        new_cmd = old_cmd.copy()
        attempts = 0
        while np.array_equal(old_cmd, new_cmd):
            self._randomize_parameters(new_cmd, loc)
            attempts += 1
            if attempts > 100:
                break
        individual.expression.mutable_raw_command_array[loc] = new_cmd

    def _random_param_location(self, individual):
        utilized = individual.get_utilized_commands()
        raw = individual.expression.raw_command_array
        # INTEGER has no meaningful numeric parameter to mutate here
        no_param_mut = {INTEGER}
        if self._cgen.input_x_dimension <= 1:
            no_param_mut.add(VARIABLE)

        indices = [
            i
            for i, u in enumerate(utilized)
            if u and int(raw[i, 0]) not in no_param_mut
        ]
        # Operator at row 1 can't change params if it only has row 0
        if 1 in indices and int(raw[1, 0]) not in TERMINAL_IDS:
            indices.remove(1)

        if not indices:
            return None
        return indices[int(self._rng.integers(len(indices)))]

    def _randomize_parameters(self, command, stack_location):
        if command[0] in TERMINAL_IDS:
            command[1] = self._cgen.random_terminal_parameter(command[0])
            command[2] = command[1]
        else:
            command[1] = self._cgen.random_operator_parameter(stack_location)
            if command[0] in ARITY_2_IDS:
                command[2] = self._cgen.random_operator_parameter(stack_location)

    # ------------------------------------------------------------------ #
    #  Prune mutation                                                     #
    # ------------------------------------------------------------------ #

    def _prune_branch(self, individual):
        self.last_mutation_type = PRUNE_MUTATION
        loc = self._random_prune_location(individual)
        if loc is None:
            return

        stack = individual.expression.mutable_raw_command_array
        node = int(stack[loc, 0])
        if node in ARITY_2_IDS:
            keep_col = 1 + int(self._rng.integers(2))
        else:
            keep_col = 1
        replacement = int(stack[loc, keep_col])

        # Redirect all references to *loc* -> *replacement*
        n = stack.shape[0]
        for i in range(loc, n):
            op = int(stack[i, 0])
            if op not in TERMINAL_IDS:
                if int(stack[i, 1]) == loc:
                    stack[i, 1] = replacement
                if int(stack[i, 2]) == loc:
                    stack[i, 2] = replacement

    def _random_prune_location(self, individual):
        utilized = individual.get_utilized_commands()
        raw = individual.expression.raw_command_array
        indices = [
            i
            for i, u in enumerate(utilized[:-1])
            if u and int(raw[i, 0]) not in TERMINAL_IDS
        ]
        if not indices:
            return None
        return indices[int(self._rng.integers(len(indices)))]

    # ------------------------------------------------------------------ #
    #  Fork mutation                                                      #
    # ------------------------------------------------------------------ #

    def _fork_mutation(self, individual):
        """Insert a new operator sub-tree after a chosen utilized row.

        The stack size is **never** changed.  Unutilized rows provide the
        "budget" for the new fork rows.

        Algorithm
        ---------
        1. Collect unutilized rows in ``[1 .. n-2]`` (row 0 must remain a
           terminal; row ``n-1`` is the output and always stays in place).
           Return immediately if there are none.
        2. Choose ``fork_size`` (1–``DEFAULT_MAX_FORK_SIZE``, capped by the
           number of available unutilized rows).
        3. Pick ``fork_target``: any utilized row in ``[0 .. n-2]``.
        4a. Compact the interior of the stack so that the unutilized rows
            sit in ``[fork_target+1 .. fork_target+N]`` (contiguous free
            slots) and the utilized interior rows are packed into
            ``[fork_target+N+1 .. n-2]``, where ``N`` is the number of
            unutilized rows found in step 1.  The output row stays at
            ``n-1``.  All operator-row references are renumbered via the
            old-to-new index map.
        """
        self.last_mutation_type = FORK_MUTATION

        utilized = individual.get_utilized_commands()
        n = len(utilized)

        # Row 0 must remain a terminal; row n-1 is the output and must stay
        # in place.  Only rows 1..n-2 are available as unutilized budget.
        unutilized = [i for i in range(1, n - 1) if not utilized[i]]
        if not unutilized:
            return

        # --- Step 2: choose fork size ---
        max_fork = min(len(unutilized), DEFAULT_MAX_FORK_SIZE)
        fork_size = int(self._rng.integers(1, max_fork + 1))

        # --- Step 3: pick fork target and reference ---
        valid_targets = [i for i, u in enumerate(utilized) if u and i < n - 1]
        if not valid_targets:
            return
        fork_target = valid_targets[int(self._rng.integers(len(valid_targets)))]

        raw = individual.expression.raw_command_array
        valid_refs = [
            i
            for i in range(fork_target + 1, n)
            if utilized[i]
            and int(raw[i, 0]) not in TERMINAL_IDS
            and (int(raw[i, 1]) == fork_target or int(raw[i, 2]) == fork_target)
        ]
        if not valid_refs:
            return
        fork_ref = valid_refs[int(self._rng.integers(len(valid_refs)))]

        # --- Step 4: classify unutilized slots relative to fork window ---
        slots_before = [i for i in unutilized if i < fork_target]
        slots_between = [i for i in unutilized if fork_target < i < fork_ref]
        slots_after = [i for i in unutilized if i > fork_ref]

        # --- Step 4b: compact if not enough free slots in the window ---
        new_stack = individual.expression.mutable_raw_command_array
        if len(slots_between) >= fork_size:
            fork_slots = slots_between[:fork_size]
        else:
            if slots_before:
                fork_target, slots_between = _compact_stack_forward(
                    new_stack,
                    fork_size,
                    fork_target,
                    slots_before,
                    slots_between,
                )
            if len(slots_between) < fork_size and slots_after:
                fork_ref, slots_between = _compact_stack_backward(
                    new_stack,
                    fork_size,
                    fork_ref,
                    slots_between,
                    slots_after,
                )
            fork_slots = slots_between[:fork_size]

        if len(fork_slots) < fork_size:
            return  # not enough space after compaction

        # --- Step 5: generate a new sub-tree ---
        fork = self._generate_fork_subtree(fork_size)

        # --- Step 6: write fork into slots, remap local refs to absolute ---
        fork_internal_refs = [fork_target] + fork_slots
        for i in range(fork_size):
            cmd = fork[i]
            if cmd[0] not in TERMINAL_IDS:
                cmd[1] = fork_internal_refs[cmd[1]]
                cmd[2] = fork_internal_refs[cmd[2]]
            self._append_constant(individual, cmd)
            new_stack[fork_slots[i]] = cmd

        # Redirect one operand of fork_ref: fork_target -> fork_output
        fork_output = fork_slots[-1]
        if int(new_stack[fork_ref, 0]) in ARITY_2_IDS:
            possible_params = [
                c for c in (1, 2) if int(new_stack[fork_ref, c]) == fork_target
            ]
            col = possible_params[int(self._rng.integers(len(possible_params)))]
            new_stack[fork_ref, col] = fork_output
        else:
            new_stack[fork_ref, 1] = fork_output
            new_stack[fork_ref, 2] = fork_output

    def _generate_fork_subtree(self, fork_size):
        connected_set = {0}
        fork = np.empty((fork_size, 3), dtype=np.uint8)
        for i in range(fork_size - 1):
            cmd = self._cgen.random_command(i + 1)
            if int(cmd[0]) not in TERMINAL_IDS and (
                int(cmd[1]) in connected_set or int(cmd[2]) in connected_set
            ):
                connected_set.add(i + 1)
            fork[i] = cmd
        fork[-1][0] = self._cgen.random_operator()
        connected_list = sorted(connected_set)
        fork[-1][1] = connected_list[int(self._rng.integers(len(connected_list)))]
        if fork[-1][0] in ARITY_2_IDS:
            fork[-1][2] = int(self._rng.integers(fork_size))
        else:
            fork[-1][2] = fork[-1][1]
        return fork

    def _get_arity_operator(self, arity):
        """Try to draw an operator of the given arity.

        Returns
        -------
        int or None
        """
        for _ in range(100):
            op = self._cgen.random_operator()
            is_arity_2 = op in ARITY_2_IDS
            if (arity == 2 and is_arity_2) or (arity == 1 and not is_arity_2):
                return op
        return None

    # ------------------------------------------------------------------ #
    #  Shared helpers                                                     #
    # ------------------------------------------------------------------ #

    def _append_constant(self, individual, command):
        """Assign a random value to a newly-generated CONSTANT command.

        If *command* is a CONSTANT node, draws a value from the
        component generator's constant distribution, appends it to
        ``individual.expression.raw_constants``, and updates the
        command's parameter to the new index.  No-op for all other
        node types.

        Parameters
        ----------
        individual : EvolvableExpression
        command : numpy.ndarray, shape (3,)
            The proposed new command (modified in-place).
        """
        if int(command[0]) != CONSTANT:
            return
        new_idx = len(individual.expression.raw_constants)
        value = self._cgen.random_constant_value()
        individual.expression.raw_constants = individual.expression.raw_constants + (
            value,
        )
        command[1] = command[2] = new_idx

    def _prune_raw_constants(self, individual):
        """Remove unused entries from ``raw_constants`` and ``raw_integers``
        and renumber their indices.

        After a mutation, CONSTANT or INTEGER nodes may reference slots
        that are no longer used.  This method compacts both pools to only
        the values actually referenced in the raw command array and
        rewrites all parameter indices in a single pass.

        Parameters
        ----------
        individual : EvolvableExpression
        """
        old_consts = individual.expression.raw_constants
        old_ints = individual.expression.raw_integers
        if not old_consts and not old_ints:
            return

        mraw = individual.expression.mutable_raw_command_array
        const_map = {}
        int_map = {}
        new_consts = []
        new_ints = []

        for i in range(mraw.shape[0]):
            op = int(mraw[i, 0])
            if op == CONSTANT:
                old_idx = int(mraw[i, 1])
                if old_idx not in const_map:
                    const_map[old_idx] = len(new_consts)
                    new_consts.append(
                        old_consts[old_idx] if old_idx < len(old_consts) else 0.0
                    )
                mraw[i, 1] = mraw[i, 2] = const_map[old_idx]
            elif op == INTEGER:
                old_idx = int(mraw[i, 1])
                if old_idx not in int_map:
                    int_map[old_idx] = len(new_ints)
                    new_ints.append(old_ints[old_idx] if old_idx < len(old_ints) else 0)
                mraw[i, 1] = mraw[i, 2] = int_map[old_idx]

        if tuple(new_consts) != old_consts:
            individual.expression.raw_constants = tuple(new_consts)
        if tuple(new_ints) != old_ints:
            individual.expression.raw_integers = tuple(new_ints)

    def _random_utilized_location(self, individual):
        utilized = individual.get_utilized_commands()
        indices = [i for i, u in enumerate(utilized) if u]
        return indices[int(self._rng.integers(len(indices)))]


if __name__ == "__main__":
    """Illustrate :meth:`_fork_mutation` across four distinct code paths.

    Run with::

        python -m bingo.expressions.agraph.mutation
    """
    import textwrap
    import numpy as np

    from .component_generator import ComponentGenerator
    from .pyagraph.expression import AGraphExpression
    from .evolvable import EvolvableExpression
    from .pyagraph.operators import (
        VARIABLE,
        CONSTANT,
        ADDITION,
        MULTIPLICATION,
        SIN,
        OPERATOR_NAMES,
    )

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _opname(op):
        names = OPERATOR_NAMES.get(op)
        if names:
            return names[0]
        return {VARIABLE: "VAR", CONSTANT: "CONST"}.get(op, str(op))

    def _make(rows, constants=()):
        expr = AGraphExpression()
        expr.raw_command_array = np.array(rows, dtype=np.uint8)
        expr.raw_constants = constants
        return EvolvableExpression(expr)

    def _show(label, indv):
        raw = indv.expression.raw_command_array
        util = indv.get_utilized_commands()
        rc = indv.expression.raw_constants
        print(f"  {label}:")
        for i, row in enumerate(raw):
            op, p1, p2 = int(row[0]), int(row[1]), int(row[2])
            flag = "U" if util[i] else " "
            print(f"    [{i}] {flag}  {_opname(op):12s}({p1}, {p2})")
        if rc:
            print(f"    raw_constants = {tuple(round(v, 4) for v in rc)}")
        else:
            print(f"    raw_constants = ()")

    def _mut(cgen, seed=0):
        return AGraphMutation(
            cgen,
            command_probability=0.0,
            node_probability=0.0,
            parameter_probability=0.0,
            prune_probability=0.0,
            fork_probability=1.0,
            random_state=seed,
        )

    def _section(title, body):
        print()
        print("=" * 60)
        print(title)
        print(textwrap.fill(body, width=60, initial_indent="  "))
        print()

    # # ------------------------------------------------------------------ #
    # #  Case 1 – no unutilized rows: immediate no-op                      #
    # # ------------------------------------------------------------------ #
    # _section(
    #     "CASE 1  No unutilized rows -> immediate no-op",
    #     "Every row is in the utilized sub-tree.  The first check "
    #     "(unutilized == []) fires and the stack is returned unchanged.",
    # )
    # # X0 + X1: all three rows are utilized.
    # cgen1 = ComponentGenerator(input_x_dimension=3, random_state=0)
    # cgen1.add_operator(ADDITION)
    # cgen1.add_operator(SIN)
    # c1 = _make([[VARIABLE, 0, 0],
    #             [VARIABLE, 1, 1],
    #             [ADDITION, 0, 1]])
    # _show("before", c1)
    # child1 = _mut(cgen1)(c1)
    # _show("after ", child1)
    # assert np.array_equal(
    #     c1.expression.raw_command_array,
    #     child1.expression.raw_command_array,
    # ), "Case 1: expected no-op"

    # # ------------------------------------------------------------------ #
    # #  Case 2 – fork_size = 1, tail reference redirected                 #
    # # ------------------------------------------------------------------ #
    # _section(
    #     "CASE 2  fork_size=1, downstream reference redirected",
    #     "One unutilized row provides the budget for a single-row fork "
    #     "inserted immediately after the fork target.  The first tail "
    #     "row that referenced the fork target is redirected to the new "
    #     "fork-output row (step 6).",
    # )
    # # sin(X0) + X0, with row 1 dead.
    # # Output = ADD(0,2), so row 2 = SIN(0) and row 0 = VAR are both utilized;
    # # row 1 (VAR X1) is NOT referenced by anything -> unutilized.
    # cgen2 = ComponentGenerator(input_x_dimension=3, random_state=5)
    # for op in (ADDITION, MULTIPLICATION, SIN):
    #     cgen2.add_operator(op)
    # c2 = _make([[VARIABLE, 0, 0],
    #             [VARIABLE, 1, 1],
    #             [SIN,      0, 0],
    #             [ADDITION, 0, 2]])
    # _show("before", c2)
    # child2 = _mut(cgen2, seed=5)(c2)
    # _show("after ", child2)
    # n2_before = c2.expression.raw_command_array.shape[0]
    # n2_after  = child2.expression.raw_command_array.shape[0]
    # assert n2_before == n2_after, "Case 2: size must not change"

    # ------------------------------------------------------------------ #
    #  Case 3 – fork_size = 2, two intermediate rows, no filler needed   #
    # ------------------------------------------------------------------ #
    _section(
        "CASE 3  Interior utilized rows packed and renumbered",
        "After the fork slot is inserted, the utilized interior rows that "
        "live between fork_target+1 and the output row are compacted "
        "into the positions immediately following the fork rows; each "
        "operand reference in those rows (and in the output row) is "
        "updated via an old->new index map built in step 4a. "
        "The remaining interior slots become filler VARIABLE terminals.",
    )
    for i in range(0, 5):
        print("=" * 60)
        print(" " * 20, "seed =", i)
        print("=" * 60)
        # sin(X0) * X1, with rows 1–3 dead.
        cgen3 = ComponentGenerator(input_x_dimension=3, random_state=i)
        for op in (ADDITION, MULTIPLICATION, SIN):
            cgen3.add_operator(op)
        c3 = _make(
            [
                [VARIABLE, 0, 0],
                [VARIABLE, 1, 1],
                [SIN, 0, 0],
                [VARIABLE, 2, 2],
                [SIN, 2, 2],
                [VARIABLE, 2, 2],
                [MULTIPLICATION, 0, 4],
            ]
        )
        _show("before", c3)
        child3 = _mut(cgen3, seed=i)(c3)
        _show("after ", child3)
        print("\n")

    # # ------------------------------------------------------------------ #
    # #  Case 4 – filler rows are CONSTANTs: raw_constants extended        #
    # # ------------------------------------------------------------------ #
    # _section(
    #     "CASE 4  Filler rows produce CONSTANT nodes, raw_constants grows",
    #     "When the number of unutilized rows exceeds fork_size + 1 the "
    #     "leftover slots are filled with random terminal commands.  Here "
    #     "constant_probability=1 forces every filler terminal to be a "
    #     "CONSTANT; _append_constant is called for each, extending "
    #     "raw_constants and assigning fresh sequential indices.",
    # )
    # # sin(X0): only rows 0 and 6 utilized; rows 1–5 are dead interior budget
    # # (6 rows excluding the output), so fork_size (max 3) always leaves at
    # # least one filler slot no matter which fork_size is chosen.
    # cgen4 = ComponentGenerator(
    #     input_x_dimension=3,
    #     constant_probability=1.0,   # all terminal draws => CONSTANT
    #     constant_scale=2.0,
    #     random_state=1,
    # )
    # for op in (ADDITION, MULTIPLICATION, SIN):
    #     cgen4.add_operator(op)
    # c4 = _make([[VARIABLE, 0, 0],
    #             [VARIABLE, 0, 0],
    #             [VARIABLE, 0, 0],
    #             [VARIABLE, 0, 0],
    #             [VARIABLE, 0, 0],
    #             [VARIABLE, 0, 0],
    #             [SIN,      0, 0]])
    # _show("before", c4)
    # child4 = _mut(cgen4, seed=1)(c4)
    # _show("after ", child4)
    # n4_before = c4.expression.raw_command_array.shape[0]
    # n4_after  = child4.expression.raw_command_array.shape[0]
    # assert n4_before == n4_after, "Case 4: size must not change"
    # assert len(child4.expression.raw_constants) > 0, \
    #     "Case 4: expected new CONSTANT values in raw_constants"
    # print()
    # print("All assertions passed.")
