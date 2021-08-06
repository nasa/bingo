"""Mutation of acyclic graph individuals.

This module contains the implementation of mutation for acyclic graph
individuals, which is composed of 4 possible mutation strategies: command
mutation, node mutation, parameter mutation and pruning.
"""
from operator import itemgetter

import numpy as np

from .operator_definitions import IS_ARITY_2_MAP, IS_TERMINAL_MAP, \
                                  CONSTANT, VARIABLE, INTEGER
from ...chromosomes.mutation import Mutation
from ...util.argument_validation import argument_validation
from ...util.probability_mass_function import ProbabilityMassFunction

COMMAND_MUTATION = 0
NODE_MUTATION = 1
PARAMETER_MUTATION = 2
PRUNE_MUTATION = 3
FORK_MUTATION = 4


class AGraphMutation(Mutation):
    """Mutation of acyclic graph individual

    Mutation of an agraph individual my modification of its command array.
    Mutation randomly takes one of the following 4 forms
      * command mutation: An entire command (row) of the command array is
        replaced by a new random one
      * node mutation: The node of a command is replaced by a random new one.
        Change from a terminal to operator (or reversed) will automatically
        have updated parameters to maintain consistency.
      * parameter mutation: The parameters of a command are randomly changed.
      * pruning: The command array is adjusted to remove an operator from
        the evaluation of the `AGraph`. Pruning a terminal does nothing.

    Parameters
    ----------
    command_probability : float
        probability of command mutation. Default 0.2
    node_probability : float
        probability of node mutation. Default 0.2
    parameter_probability : float
        probability of parameter mutation. Default 0.2
    prune_probability : float
        probability of pruning. Default 0.2
    fork_probability : float
        probability of forking. Default 0.2

    Notes
    -----
    The input probabilities are normalized if their sum is not equal to 1.

    Mutation can result in no change if, for instance,
      * a prune mutation is executed on a `AGraph` utilizing only a single
        terminal.
      * a parameter mutation occurs on a `AGraph` utilizing only a single
        constant.
    """

    @argument_validation(command_probability={">=": 0, "<=": 1},
                         node_probability={">=": 0, "<=": 1},
                         parameter_probability={">=": 0, "<=": 1},
                         prune_probability={">=": 0, "<=": 1},
                         fork_probability={">=": 0, "<=": 1})
    def __init__(self, component_generator, command_probability=0.2,
                 node_probability=0.2, parameter_probability=0.2,
                 prune_probability=0.2, fork_probability=0.2):
        self._component_generator = component_generator
        self._mutation_function_pmf = \
            ProbabilityMassFunction([self._mutate_command,
                                     self._mutate_node,
                                     self._mutate_parameters,
                                     self._prune_branch,
                                     self._fork_mutation_2],
                                    [command_probability,
                                     node_probability,
                                     parameter_probability,
                                     prune_probability,
                                     fork_probability])
        self._last_mutation_location = None
        self._last_mutation_type = None

    def __call__(self, parent):
        """Single point mutation.

        Parameters
        ----------
        parent : `AGraph`
            The parent individual

        Returns
        -------
        `AGraph` :
            The child of the mutation
        """
        child = parent.copy()
        mutation_algorithm = self._mutation_function_pmf.draw_sample()
        mutation_algorithm(child)

        for i, (command, op1, op2) in enumerate(child.command_array):
            if not IS_TERMINAL_MAP[command]:
                if op1 >= i or op2 >= i:
                    print("bad!", self._last_mutation_type, self._last_mutation_location)
                    print("parent", parent.command_array)
                    print("child", child.command_array)
                    raise(RuntimeError)

        return child

    def _mutate_command(self, individual):
        mutation_location = \
            self._get_random_command_mutation_location(individual)
        self._last_mutation_location = mutation_location
        self._last_mutation_type = COMMAND_MUTATION

        old_command = individual.command_array[mutation_location]
        new_command = \
            self._component_generator.random_command(mutation_location)
        while np.array_equal(new_command, old_command) \
                or old_command[0] == new_command[0] == CONSTANT:
            new_command = \
                self._component_generator.random_command(mutation_location)

        individual.mutable_command_array[mutation_location] = new_command

    @staticmethod
    def _get_random_command_mutation_location(child):
        utilized_commands = child.get_utilized_commands()
        indices = [n for n, x in enumerate(utilized_commands) if x]
        index = np.random.randint(len(indices))
        return indices[index]

    def _mutate_node(self, individual):
        mutation_location = self._get_random_node_mutation_location(individual)
        self._last_mutation_location = mutation_location
        self._last_mutation_type = NODE_MUTATION

        old_command = individual.command_array[mutation_location]
        new_command = old_command.copy()
        while old_command[0] == new_command[0]:
            self._randomize_node(new_command)

        individual.mutable_command_array[mutation_location] = new_command

    def _get_random_node_mutation_location(self, child):
        utilized_commands = child.get_utilized_commands()
        terminals_ok = self._component_generator.get_number_of_terminals() > 1
        operators_ok = self._component_generator.get_number_of_operators() > 1
        indices = []
        for i, (x, node) in enumerate(zip(utilized_commands,
                                          child.command_array[:, 0])):
            if x:
                if (IS_TERMINAL_MAP[node] and terminals_ok) or \
                        (not IS_TERMINAL_MAP[node] and operators_ok):
                    indices.append(i)
        index = np.random.randint(len(indices))
        return indices[index]

    def _randomize_node(self, command):
        if IS_TERMINAL_MAP[command[0]]:
            command[0] = self._component_generator.random_terminal()
            command[1] = \
                self._component_generator.random_terminal_parameter(command[0])
            command[2] = command[1]
        else:
            command[0] = self._component_generator.random_operator()

    def _mutate_parameters(self, individual):
        mutation_location = self._get_random_param_mut_location(individual)
        self._last_mutation_location = mutation_location
        self._last_mutation_type = PARAMETER_MUTATION
        if mutation_location is None:
            return

        old_command = individual.command_array[mutation_location]
        new_command = old_command.copy()
        while np.array_equal(old_command, new_command):
            self._randomize_parameters(new_command, mutation_location)

        individual.mutable_command_array[mutation_location] = new_command

    def _get_random_param_mut_location(self, individual):
        utilized_commands = individual.get_utilized_commands()
        no_param_mut = [CONSTANT, INTEGER]
        if self._component_generator.input_x_dimension <= 1:
            no_param_mut += [VARIABLE]

        indices = [i for i, x in enumerate(utilized_commands)
                   if x and
                   individual.command_array[i, 0] not in no_param_mut]

        if 1 in indices and \
                not IS_TERMINAL_MAP[individual.command_array[1, 0]]:
            indices.remove(1)

        if not indices:
            return None
        index = np.random.randint(len(indices))
        return indices[index]

    def _randomize_parameters(self, command, mutation_location):
        if IS_TERMINAL_MAP[command[0]]:
            command[1] = \
                self._component_generator.random_terminal_parameter(command[0])
            command[2] = command[1]
        else:
            command[1] = \
                self._component_generator.random_operator_parameter(
                    mutation_location)
            if IS_ARITY_2_MAP[command[0]]:
                command[2] = \
                    self._component_generator.random_operator_parameter(
                        mutation_location)

    def _prune_branch(self, individual):
        mutation_location = self._get_random_prune_location(individual)
        self._last_mutation_location = mutation_location
        self._last_mutation_type = PRUNE_MUTATION
        if mutation_location is None:
            return

        mutated_node = individual.command_array[mutation_location, 0]
        if IS_ARITY_2_MAP[mutated_node]:
            pruned_param_num = np.random.randint(2)
        else:
            pruned_param_num = 0
        pruned_param = individual.command_array[mutation_location,
                                                1 + pruned_param_num]

        for i, (node, p_1, p_2) in \
                enumerate(individual.command_array[mutation_location:]):
            if not IS_TERMINAL_MAP[node]:
                if p_1 == mutation_location:
                    individual.mutable_command_array[mutation_location + i, 1] = \
                        pruned_param
                if p_2 == mutation_location:
                    individual.mutable_command_array[mutation_location + i, 2] = \
                        pruned_param

    @staticmethod
    def _get_random_prune_location(individual):
        utilized_commands = individual.get_utilized_commands()
        indices = [i for i, x in enumerate(utilized_commands[:-1])
                   if x and
                   not IS_TERMINAL_MAP[individual.command_array[i, 0]]]
        if not indices:
            return None
        index = np.random.randint(len(indices))
        return indices[index]

    def _fork_mutation(self, individual):
        MAX_FORK_SIZE = 4

        utilized_commands = individual.get_utilized_commands()
        num_unusused_commands = utilized_commands.count(False)

        if num_unusused_commands < 2:
            self._last_mutation_location = None
            self._last_mutation_type = FORK_MUTATION
            return

        indices = [n for n, x in enumerate(utilized_commands) if x]
        mutation_location = np.random.choice(indices)

        max_fork = min(num_unusused_commands, MAX_FORK_SIZE)
        fork_size = np.random.randint(2, max_fork + 1)

        fork_space, new_mut_location = self._make_space_for_fork(
            fork_size, individual, mutation_location, utilized_commands)

        fork_commands = self._generate_fork(fork_size, fork_space,
                                            new_mut_location)

        individual.mutable_command_array[fork_space] = fork_commands.copy()

        # this check is needed to account for the case when an arity 1 node is
        # mutated/shifted, then the op2 argument can get messed up
        # TODO: this is messy
        for i, (command, _, op2) in enumerate(individual.command_array):
            if not IS_TERMINAL_MAP[command]:
                if op2 >= i:
                    individual.mutable_command_array[i, 2] = \
                        self._component_generator.random_operator_parameter(i)

        self._last_mutation_location = mutation_location
        self._last_mutation_type = FORK_MUTATION

    @staticmethod
    def _make_space_for_fork(fork_size, individual, mutation_location,
                             utilized_commands):
        new_permutation = np.arange(len(utilized_commands))
        # backwards
        fork_space = []
        for i in range(mutation_location, -1, -1):
            if not utilized_commands[i]:
                if len(fork_space) == 0:
                    fork_space = [mutation_location]
                else:
                    fork_space.insert(0, fork_space[0] - 1)
                new_permutation[i:fork_space[0] + 1] = \
                    np.roll(new_permutation[i:fork_space[0] + 1], -1)
                if len(fork_space) == fork_size:
                    break
        # forwards
        if len(fork_space) < fork_size:
            for i in range(mutation_location + 1, len(utilized_commands)):
                if not utilized_commands[i]:
                    if len(fork_space) == 0:
                        fork_space = [mutation_location + 1]
                    else:
                        fork_space.append(fork_space[-1] + 1)
                    new_permutation[fork_space[-1]:i + 1] = \
                        np.roll(new_permutation[fork_space[-1]:i + 1], 1)
                    if len(fork_space) == fork_size:
                        break
        # update parameters
        terminals = np.vectorize(IS_TERMINAL_MAP.get)(
                individual.command_array[:, 0])
        used_operators = np.logical_and(~terminals, utilized_commands)
        parameter_conversion = np.argsort(new_permutation)
        new_mut_location = parameter_conversion[mutation_location]
        parameter_conversion[mutation_location] = fork_space[-1]
        individual.mutable_command_array[used_operators, 1] = \
            parameter_conversion[individual.command_array[used_operators, 1]]
        individual.mutable_command_array[used_operators, 2] = \
            parameter_conversion[individual.command_array[used_operators, 2]]
        # move commands
        individual.mutable_command_array[:, :] = \
            individual.command_array[new_permutation, :].copy()
        return fork_space, new_mut_location

    def _generate_fork(self, fork_size, fork_space, new_mut_location):
        fork_commands = np.empty((fork_size, 3), dtype=int)
        num_terminals = np.random.randint(1, fork_size // 2 + 1)
        for i in range(num_terminals):
            fork_commands[i] = \
                self._component_generator.random_terminal_command(i)
        for i in range(num_terminals, fork_size):
            fork_commands[i] = \
                self._component_generator.random_operator_command(i)
            fork_commands[i, 1] = fork_space[fork_commands[i, 1]]
            fork_commands[i, 2] = fork_space[fork_commands[i, 2]]
        attempt_count = 0
        while not IS_ARITY_2_MAP[fork_commands[-1, 0]]:
            attempt_count += 1
            fork_commands[-1] = \
                self._component_generator.random_operator_command(
                    fork_size - 1)
            fork_commands[-1, 1] = fork_space[fork_commands[-1, 1]]
            fork_commands[-1, 2] = fork_space[fork_commands[-1, 2]]
            if attempt_count > 100:
                break
        fork_commands[-1, np.random.randint(1, 3)] = new_mut_location
        return fork_commands

    def _fork_mutation_2(self, individual):
        MAX_FORK_SIZE = 4

        utilized_commands = individual.get_utilized_commands()
        n_unutilized_commands = utilized_commands.count(False)

        if n_unutilized_commands < 2:
            self._last_mutation_location = None
            self._last_mutation_type = FORK_MUTATION
            return

        max_fork = min(n_unutilized_commands, MAX_FORK_SIZE)
        fork_size = np.random.randint(2, max_fork + 1)

        indices = [n for n, x in enumerate(utilized_commands) if x]
        original_mutation_location = mutation_location = np.random.choice(indices)

        mutation_location, utilized_commands, index_shifts = self._move_unutilized_to_top(individual, utilized_commands, mutation_location)
        new_mutation_location = self._move_utilized_to_top(individual, utilized_commands, mutation_location, index_shifts)
        self._insert_fork(individual, mutation_location, new_mutation_location, fork_size)

        # this check is needed to account for the case when an arity 1 node is
        # mutated/shifted, then the op2 argument can get messed up
        # TODO: this is messy
        for i, (command, _, op2) in enumerate(individual.command_array):
            if not IS_TERMINAL_MAP[command]:
                if op2 >= i:
                    individual.mutable_command_array[i, 2] = \
                        self._component_generator.random_operator_parameter(i)

        self._last_mutation_location = original_mutation_location
        self._last_mutation_type = FORK_MUTATION

    def _move_unutilized_to_top(self, individual, utilized_commands, mutation_location):
        new_stack, new_utilized_commands, index_shifts = \
            self._move_utilized_commands(individual.mutable_command_array,
                                         utilized_commands,
                                         to_front=True)
        individual.command_array = new_stack
        new_mutation_location = index_shifts[mutation_location]
        return new_mutation_location, new_utilized_commands, index_shifts

    def _move_utilized_to_top(self, individual, utilized_commands, mutation_location, old_index_shifts):
        new_stack, new_utilized_commands, index_shifts =\
            self._move_utilized_commands(individual.mutable_command_array[:mutation_location + 1],
                                         utilized_commands[:mutation_location + 1],
                                         to_front=False)

        new_mutation_location = index_shifts[mutation_location]
        new_stack = np.vstack((new_stack, individual.command_array[mutation_location+1:]))

        new_utilized_commands.extend(utilized_commands[mutation_location + 1:])

        # update index shifts with the indices that didn't change
        no_change_indices = dict(zip(range(len(individual.mutable_command_array))[mutation_location:],
                                     range(len(individual.mutable_command_array))[mutation_location:]))
        index_shifts.update(no_change_indices)

        combined_index_shifts = dict(zip(old_index_shifts.keys(), itemgetter(*old_index_shifts.values())(index_shifts)))

        self._fix_indices(new_stack, new_utilized_commands, combined_index_shifts, new_mutation_location)
        individual.command_array = new_stack

        return new_mutation_location

    def _insert_fork(self, individual, mutation_location, new_mutation_location, fork_size):
        # start with an arity 2 operator (to link the mutated command with another command)
        # (starting with an arity 2 operator makes the fork_size - 1)
        # generate operators until the fork_size reaches 3, 2, or 1
        # where if fork_size = 3 -> ar2 operator, fork_size = 2 -> ar1 operator,
        # fork_size = 1 -> terminal

        insertion_index = mutation_location
        start_operator = self._get_arity_operator(2)

        # add start arity 2 operator that links to mutated command
        start_command = np.array([start_operator, new_mutation_location, insertion_index - 1], dtype=int)
        individual.mutable_command_array[insertion_index] = start_command
        fork_size -= 1
        insertion_index -= 1

        while fork_size > 3:
            new_operator = self._component_generator.random_operator()
            if IS_ARITY_2_MAP[new_operator]:
                new_operator_command = np.array([new_operator, insertion_index - 1, insertion_index - 2], dtype=int)
            else:
                new_operator_command = np.array([new_operator, insertion_index - 1, insertion_index - 1], dtype=int)
            individual.mutable_command_array[insertion_index] = new_operator_command
            fork_size -= 1
            insertion_index -= 1

        if fork_size == 3:
            new_operator_command = np.array([self._get_arity_operator(2), insertion_index - 1, insertion_index - 2])
            individual.mutable_command_array[insertion_index] = new_operator_command
            individual.mutable_command_array[insertion_index - 1] = self._component_generator.random_terminal_command()
            individual.mutable_command_array[insertion_index - 2] = self._component_generator.random_terminal_command()
        elif fork_size == 2:
            new_operator_command = np.array([self._get_arity_operator(1), insertion_index - 1, insertion_index - 1])
            individual.mutable_command_array[insertion_index] = new_operator_command
            individual.mutable_command_array[insertion_index - 1] = self._component_generator.random_terminal_command()
        else:
            individual.mutable_command_array[insertion_index] = self._component_generator.random_terminal_command()

    @staticmethod
    def _move_utilized_commands(stack, utilized_commands, to_front=True):
        indices = range(len(stack))
        new_tuples = []
        counter = 0
        for stack_util_index_tuple in zip(stack, utilized_commands, indices):
            if to_front:
                if stack_util_index_tuple[1]:  # if utilized
                    new_tuples.append(stack_util_index_tuple)
                else:
                    new_tuples.insert(counter, stack_util_index_tuple)
                    counter += 1
            else:
                if not stack_util_index_tuple[1]:  # if not utilized
                    new_tuples.append(stack_util_index_tuple)
                else:
                    new_tuples.insert(counter, stack_util_index_tuple)
                    counter += 1
        new_stack, new_utilized_commands, new_indices = zip(*new_tuples)
        index_shifts = dict(zip(new_indices, indices))
        return np.array(new_stack), list(new_utilized_commands), index_shifts

    def _fix_indices(self, stack, utilized_commands, index_shifts, new_mutation_location):
        non_terminals = ~np.vectorize(IS_TERMINAL_MAP.get)(stack[:, 0])
        utilized_operators = np.logical_and(non_terminals, utilized_commands)
        index_shifts = dict(sorted(index_shifts.items(), key=lambda pair: pair[0]))
        index_values = np.array(list(index_shifts.values()))

        for j in range(1, 3):
            stack[utilized_operators, j] = index_values[stack[utilized_operators, j]]

        i = utilized_commands.index(False)
        for command in stack[~np.array(utilized_commands)]:
            if non_terminals[i]:
                for j in range(1, 3):
                    if command[j] >= i:
                        stack[i][j] = self._component_generator.random_operator_parameter(i)
            i += 1

    def _get_arity_operator(self, arity):
        attempts = 0
        operator = None
        if arity == 1:
            while operator is None or IS_ARITY_2_MAP[operator]:
                if attempts >= 100:
                    raise RuntimeError("Could not generate arity {} operator".format(arity))
                operator = self._component_generator.random_operator()
                attempts += 1
        else:
            while operator is None or not IS_ARITY_2_MAP[operator]:
                if attempts >= 100:
                    raise RuntimeError("Could not generate arity {} operator".format(arity))
                operator = self._component_generator.random_operator()
                attempts += 1
        return operator
