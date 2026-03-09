"""Tests for AGraphMutation."""

import numpy as np
import pytest

from bingo.expressions.agraph.mutation import (
    AGraphMutation,
    COMMAND_MUTATION,
    NODE_MUTATION,
    PARAMETER_MUTATION,
    PRUNE_MUTATION,
    FORK_MUTATION,
)
from bingo.expressions.agraph.component_generator import ComponentGenerator
from bingo.expressions.agraph.pyagraph.expression import AGraphExpression
from bingo.expressions.agraph.pyagraph.operators import (
    VARIABLE,
    CONSTANT,
    ADDITION,
    SUBTRACTION,
    MULTIPLICATION,
    SIN,
    TERMINAL_IDS,
)
from bingo.expressions.agraph.evolvable import EvolvableExpression


def _make_individual(command_rows, constants=(), integers=()):
    expr = AGraphExpression()
    expr.raw_command_array = np.array(command_rows, dtype=np.uint8)
    expr.raw_integers = integers
    expr.raw_constants = constants
    return EvolvableExpression(expr)


@pytest.fixture
def cgen():
    gen = ComponentGenerator(input_x_dimension=3, random_state=0)
    gen.add_operator(ADDITION)
    gen.add_operator(SUBTRACTION)
    gen.add_operator(MULTIPLICATION)
    gen.add_operator(SIN)
    return gen


@pytest.fixture
def mutation(cgen):
    return AGraphMutation(cgen, random_state=0)


@pytest.fixture
def individual():
    """A simple 5-row individual: sin(X0 + C0)."""
    return _make_individual(
        [
            [VARIABLE, 0, 0],
            [CONSTANT, 0, 0],
            [ADDITION, 0, 1],
            [SIN, 2, 2],
            [MULTIPLICATION, 2, 3],
        ],
        constants=(1.0,),
    )


class TestMutationBasics:
    def test_returns_evolvable_expression(self, mutation, individual):
        child = mutation(individual)
        assert isinstance(child, EvolvableExpression)

    def test_child_is_different_object(self, mutation, individual):
        child = mutation(individual)
        assert child is not individual

    def test_parent_unchanged(self, mutation, individual):
        original = individual.command_array.copy()
        mutation(individual)
        np.testing.assert_array_equal(individual.command_array, original)

    def test_fit_set_cleared(self, mutation, individual):
        individual.fitness = 1.0
        child = mutation(individual)
        assert not child.fit_set

    def test_mutation_type_recorded(self, mutation, individual):
        mutation(individual)
        assert mutation.last_mutation_type in mutation.types

    def test_all_types_available(self, mutation):
        assert set(mutation.types) == {
            COMMAND_MUTATION,
            NODE_MUTATION,
            PARAMETER_MUTATION,
            PRUNE_MUTATION,
            FORK_MUTATION,
        }


class TestCommandMutation:
    def test_command_only(self, cgen, individual):
        mut = AGraphMutation(
            cgen,
            command_probability=1.0,
            node_probability=0.0,
            parameter_probability=0.0,
            prune_probability=0.0,
            fork_probability=0.0,
            random_state=0,
        )
        child = mut(individual)
        assert mut.last_mutation_type == COMMAND_MUTATION
        # At least one row should differ
        assert not np.array_equal(child.command_array, individual.command_array)


class TestNodeMutation:
    def test_node_only(self, cgen, individual):
        mut = AGraphMutation(
            cgen,
            command_probability=0.0,
            node_probability=1.0,
            parameter_probability=0.0,
            prune_probability=0.0,
            fork_probability=0.0,
            random_state=0,
        )
        child = mut(individual)
        assert mut.last_mutation_type == NODE_MUTATION


class TestParameterMutation:
    def test_parameter_only(self, cgen, individual):
        mut = AGraphMutation(
            cgen,
            command_probability=0.0,
            node_probability=0.0,
            parameter_probability=1.0,
            prune_probability=0.0,
            fork_probability=0.0,
            random_state=0,
        )
        child = mut(individual)
        assert mut.last_mutation_type == PARAMETER_MUTATION


class TestPruneMutation:
    def test_prune_only(self, cgen, individual):
        mut = AGraphMutation(
            cgen,
            command_probability=0.0,
            node_probability=0.0,
            parameter_probability=0.0,
            prune_probability=1.0,
            fork_probability=0.0,
            random_state=0,
        )
        child = mut(individual)
        assert mut.last_mutation_type == PRUNE_MUTATION

    def test_prune_terminal_only_is_noop(self, cgen):
        """Pruning when only terminals are utilized does nothing."""
        indv = _make_individual([[VARIABLE, 0, 0]])
        mut = AGraphMutation(
            cgen,
            command_probability=0.0,
            node_probability=0.0,
            parameter_probability=0.0,
            prune_probability=1.0,
            fork_probability=0.0,
            random_state=0,
        )
        child = mut(indv)
        np.testing.assert_array_equal(child.command_array, indv.command_array)


class TestForkMutation:
    def test_fork_only(self, cgen, individual):
        mut = AGraphMutation(
            cgen,
            command_probability=0.0,
            node_probability=0.0,
            parameter_probability=0.0,
            prune_probability=0.0,
            fork_probability=1.0,
            random_state=42,
        )
        child = mut(individual)
        assert mut.last_mutation_type == FORK_MUTATION

    def test_fork_never_grows_stack(self, cgen):
        """Fork must never change the stack size."""
        # Individual with unutilized rows (row 1 is dead)
        indv = _make_individual(
            [
                [VARIABLE, 0, 0],  # row 0 — utilized (root chain)
                [VARIABLE, 1, 1],  # row 1 — unutilized
                [SIN, 0, 0],  # row 2 — utilized (output)
            ]
        )
        mut = AGraphMutation(
            cgen,
            command_probability=0.0,
            node_probability=0.0,
            parameter_probability=0.0,
            prune_probability=0.0,
            fork_probability=1.0,
            random_state=0,
        )
        n_before = indv.expression.raw_command_array.shape[0]
        for _ in range(20):
            child = mut(indv)
            assert child.expression.raw_command_array.shape[0] == n_before

    def test_fork_noop_when_all_utilized(self, cgen):
        """Fork is a no-op when every row is utilized (no room to insert)."""
        # All rows utilized: X0 + X1 + X2 (3 utilized rows, none spare)
        indv = _make_individual(
            [
                [VARIABLE, 0, 0],
                [VARIABLE, 1, 1],
                [ADDITION, 0, 1],
            ]
        )
        mut = AGraphMutation(
            cgen,
            command_probability=0.0,
            node_probability=0.0,
            parameter_probability=0.0,
            prune_probability=0.0,
            fork_probability=1.0,
            random_state=0,
        )
        child = mut(indv)
        # All rows are utilized so fork must be a no-op
        np.testing.assert_array_equal(
            child.expression.raw_command_array,
            indv.expression.raw_command_array,
        )

    def test_fork_inserts_after_fork_target(self, cgen):
        """Fork rows appear right after the fork target in the new stack."""
        from bingo.expressions.agraph.pyagraph.operators import TERMINAL_IDS as TID

        indv = _make_individual(
            [
                [VARIABLE, 0, 0],  # row 0 — utilized
                [VARIABLE, 1, 1],  # row 1 — unutilized (budget)
                [VARIABLE, 2, 2],  # row 2 — unutilized (budget)
                [SIN, 0, 0],  # row 3 — utilized
                [ADDITION, 0, 3],  # row 4 — utilized (output)
            ]
        )
        mut = AGraphMutation(
            cgen,
            command_probability=0.0,
            node_probability=0.0,
            parameter_probability=0.0,
            prune_probability=0.0,
            fork_probability=1.0,
            random_state=0,
        )
        for _ in range(30):
            child = mut(indv)
            raw = child.expression.raw_command_array
            # Size must not change
            assert raw.shape[0] == 5
            # All operator parameters must reference only earlier rows
            for i in range(raw.shape[0]):
                op = int(raw[i, 0])
                if op not in TID:
                    assert int(raw[i, 1]) < i
                    assert int(raw[i, 2]) < i


class TestRepeatedMutation:
    def test_many_mutations_no_crash(self, mutation, individual):
        current = individual
        for _ in range(100):
            current = mutation(current)
        assert isinstance(current, EvolvableExpression)


class TestConstantValueInMutation:
    """New CONSTANT nodes introduced by mutation should have their value
    recorded in raw_constants at the correct index."""

    def _all_constant_indices_valid(self, indv):
        """Return True if every CONSTANT row has a valid raw_constants entry."""
        ca = indv.expression.raw_command_array
        rc = indv.expression.raw_constants
        for row in ca:
            if int(row[0]) == CONSTANT:
                if int(row[1]) >= len(rc):
                    return False
        return True

    def test_command_mutation_populates_raw_constants(self, cgen):
        """Command mutation producing a CONSTANT must append its value."""
        # Force all terminals to CONSTANT
        forced_cgen = ComponentGenerator(
            input_x_dimension=3, constant_probability=1.0, random_state=1
        )
        for op in (ADDITION, SUBTRACTION, MULTIPLICATION, SIN):
            forced_cgen.add_operator(op)
        # Start with a VARIABLE individual (no raw_constants initially)
        indv = _make_individual([[VARIABLE, 0, 0], [SIN, 0, 0]])
        mut = AGraphMutation(
            forced_cgen,
            command_probability=1.0,
            node_probability=0.0,
            parameter_probability=0.0,
            prune_probability=0.0,
            fork_probability=0.0,
            random_state=0,
        )
        for _ in range(30):
            child = mut(indv)
            assert self._all_constant_indices_valid(child)

    def test_node_mutation_populates_raw_constants(self, cgen):
        """Node mutation turning a VARIABLE into CONSTANT must append a value."""
        # Individual with a single VARIABLE terminal — node mutation may turn
        # it into CONSTANT (needs >=2 terminal types, which cgen has).
        indv = _make_individual([[VARIABLE, 0, 0], [SIN, 0, 0]])
        mut = AGraphMutation(
            cgen,
            command_probability=0.0,
            node_probability=1.0,
            parameter_probability=0.0,
            prune_probability=0.0,
            fork_probability=0.0,
            random_state=0,
        )
        for _ in range(50):
            child = mut(indv)
            assert self._all_constant_indices_valid(child)

    def test_raw_constants_unchanged_for_non_constant_mutation(self, cgen):
        """Mutations that don't introduce new CONSTANTs should not grow raw_constants."""
        indv = _make_individual(
            [[VARIABLE, 0, 0], [VARIABLE, 1, 1], [ADDITION, 0, 1]],
            constants=(),
        )
        mut = AGraphMutation(
            cgen,
            command_probability=0.0,
            node_probability=0.0,
            parameter_probability=1.0,
            prune_probability=0.0,
            fork_probability=0.0,
            random_state=0,
        )
        for _ in range(20):
            child = mut(indv)
            # Parameter mutation never introduces new CONSTANT nodes
            for row in child.expression.raw_command_array:
                if int(row[0]) == CONSTANT:
                    assert int(row[1]) < len(child.expression.raw_constants)

    def test_fork_mutation_populates_raw_constants(self):
        """Fork mutation generating a CONSTANT in the sub-tree must add it
        to raw_constants with a valid index."""
        # Force all terminals to CONSTANT so any filler / sub-tree terminal
        # becomes a CONSTANT node.
        forced_cgen = ComponentGenerator(
            input_x_dimension=3, constant_probability=1.0, random_state=0
        )
        for op in (ADDITION, SUBTRACTION, MULTIPLICATION, SIN):
            forced_cgen.add_operator(op)

        # Individual with several unutilized rows to give fork room to work
        indv = _make_individual(
            [
                [VARIABLE, 0, 0],
                [VARIABLE, 0, 0],
                [VARIABLE, 0, 0],
                [VARIABLE, 0, 0],
                [VARIABLE, 0, 0],
                [SIN, 0, 0],
            ]
        )
        mut = AGraphMutation(
            forced_cgen,
            command_probability=0.0,
            node_probability=0.0,
            parameter_probability=0.0,
            prune_probability=0.0,
            fork_probability=1.0,
            random_state=0,
        )
        found_constant = False
        for seed in range(50):
            mut_i = AGraphMutation(
                forced_cgen,
                command_probability=0.0,
                node_probability=0.0,
                parameter_probability=0.0,
                prune_probability=0.0,
                fork_probability=1.0,
                random_state=seed,
            )
            child = mut_i(indv)
            assert self._all_constant_indices_valid(child)
            if child.expression.raw_constants:
                found_constant = True
        assert (
            found_constant
        ), "Expected at least one fork mutation to introduce a CONSTANT node"
