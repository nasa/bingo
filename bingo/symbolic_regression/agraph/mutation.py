"""Mutation of acyclic graph individuals.

This module contains the implementation of mutation for acyclic graph
individuals, which is composed of 4 possible mutation strategies: command
mutation, node mutation, parameter mutation and pruning.
"""
import numpy as np

from .maps import IS_ARITY_2_MAP, IS_TERMINAL_MAP
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
      * node mutation: The node of a command is replaced by a random
                           new one.  Terminals will automatically have updated
                           parameters to maintain consistency.
      * parameter mutation: The parameters of a command are randomly changed.
      * pruning: The command array is adjusted to remove an operator from
                 the evaluation of the agraph. Pruning a terminal does nothing.

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
      * a prune mutation is executed on a Agraph utilizing only a single
        terminal.
      * a parameter mutation occurs on a Agraph utilizing only a single
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
                                     self._fork_mutation],
                                    [command_probability,
                                     node_probability,
                                     parameter_probability,
                                     prune_probability,
                                     fork_probability])
        self._manual_constants = \
            not component_generator.automatic_constant_optimization
        self._last_mutation_location = None
        self._last_mutation_type = None

    def __call__(self, parent):
        """Single point mutation.

        Parameters
        ----------
        parent : Agraph
                 The parent individual

        Returns
        -------
        Agraph :
            The child of the mutation
        """
        child = parent.copy()
        mutation_algorithm = self._mutation_function_pmf.draw_sample()
        mutation_algorithm(child)

        if self._manual_constants:
            self._track_constants(parent, child)

        # TODO can we shift this responsibility to agraph?
        child.notify_command_array_modification()

        return child

    @staticmethod
    def _get_random_mutation_location(child):
        utilized_commands = child.get_utilized_commands()
        indices = [n for n, x in enumerate(utilized_commands) if x]
        return np.random.choice(indices)

    def _mutate_command(self, individual):
        mutation_location = self._get_random_mutation_location(individual)
        old_command = np.copy(individual.command_array[mutation_location])

        continue_search_for_new_command = True
        while continue_search_for_new_command:
            individual.command_array[mutation_location] = \
                self._component_generator.random_command(mutation_location)

            continue_search_for_new_command = \
                np.array_equal(individual.command_array[mutation_location],
                               old_command)
        self._last_mutation_location = mutation_location
        self._last_mutation_type = COMMAND_MUTATION

    def _track_constants(self, parent, child):
        child.force_renumber_constants()
        child.constants = [0., ]*child.num_constants
        for i, (command, param1, _) in enumerate(child.command_array):
            if command == 1 and param1 != -1:
                if i == self._last_mutation_location:
                    if self._last_mutation_type == PARAMETER_MUTATION:
                        old_constant_num = parent.command_array[i, 1]
                        constant = \
                            self._component_generator.random_numerical_constant(
                                parent.constants[old_constant_num])
                    else:
                        constant = \
                            self._component_generator.random_numerical_constant()
                elif self._last_mutation_type == FORK_MUTATION:
                    constant = \
                        self._component_generator.random_numerical_constant()
                else:
                    old_constant_num = parent.command_array[i, 1]
                    if old_constant_num == -1:
                        constant = \
                            self._component_generator.random_numerical_constant()
                    else:
                        constant = parent.constants[old_constant_num]
                child.constants[param1] = constant

    def _mutate_node(self, individual):
        mutation_location = self._get_random_mutation_location(individual)
        old_node = individual.command_array[mutation_location, 0]
        mutated_command = individual.command_array[mutation_location]
        is_terminal = IS_TERMINAL_MAP[old_node]

        if self._is_new_node_possible(is_terminal):
            self._force_mutated_node(is_terminal, mutated_command, old_node)

        self._last_mutation_location = mutation_location
        self._last_mutation_type = NODE_MUTATION

    def _is_new_node_possible(self, is_terminal):
        if is_terminal:
            return self._component_generator.get_number_of_terminals() > 1
        return self._component_generator.get_number_of_operators() > 1

    def _force_mutated_node(self, is_terminal, mutated_command, old_node):
        unique_node = False
        while not unique_node:
            self._randomize_node(is_terminal, mutated_command)
            unique_node = mutated_command[0] != old_node

    def _randomize_node(self, is_terminal, mutated_command):
        if is_terminal:
            new_terminal = \
                self._component_generator.random_terminal()
            mutated_command[0] = new_terminal
            mutated_command[1] = \
                self._component_generator.random_terminal_parameter(
                    new_terminal)
            # if IS_ARITY_2_MAP[new_terminal]:
            #     mutated_command[2] = \
            #         self._component_generator.random_terminal_parameter(
            #                 new_terminal)
        else:
            mutated_command[0] = \
                self._component_generator.random_operator()

    def _mutate_parameters(self, individual):
        mutation_location = self._get_random_param_mut_location(individual)
        if mutation_location is None:
            return
        old_command = np.copy(individual.command_array[mutation_location])
        mutated_command = individual.command_array[mutation_location]

        if self._is_new_param_possible(old_command[0], mutation_location):
            self._force_mutated_parameters(mutated_command,
                                           old_command,
                                           mutation_location)

        self._last_mutation_location = mutation_location
        self._last_mutation_type = PARAMETER_MUTATION

    def _get_random_param_mut_location(self, individual):
        utilized_commands = individual.get_utilized_commands()
        acceptable_indices = []
        for i, (util, node) in enumerate(zip(utilized_commands,
                                             individual.command_array[:, 0])):
            if util:
                if self._manual_constants or node != 1:  # TODO hard coded info about node map
                    acceptable_indices.append(i)

        if not acceptable_indices:
            return None

        return np.random.choice(acceptable_indices)

    def _force_mutated_parameters(self,
                                  mutated_command,
                                  old_command,
                                  mutation_location):
        is_terminal = IS_TERMINAL_MAP[old_command[0]]
        unique_params = False
        while not unique_params:
            self._randomize_parameters(is_terminal,
                                       mutated_command,
                                       mutation_location)
            unique_params = not np.array_equal(mutated_command,
                                               old_command)

    def _is_new_param_possible(self, node, mutation_location):
        # TODO hard coded info about node map
        if node == 0:
            return self._component_generator.input_x_dimension > 1
        if node == 1:
            return True
        return mutation_location > 1

    def _randomize_parameters(self,
                              is_terminal,
                              mutated_command,
                              mutation_location):
        if is_terminal:
            mutated_command[1] = \
                self._component_generator.random_terminal_parameter(
                    mutated_command[0])
            # if IS_ARITY_2_MAP[mutated_command[0]]:
            #     mutated_command[2] = \
            #         self._component_generator.random_terminal_parameter(
            #                 mutated_command[0])
        else:
            mutated_command[1] = \
                self._component_generator.random_operator_parameter(
                    mutation_location)
            if IS_ARITY_2_MAP[mutated_command[0]]:
                mutated_command[2] = \
                    self._component_generator.random_operator_parameter(
                        mutation_location)

    def _prune_branch(self, individual):
        mutation_location = \
            AGraphMutation._get_random_prune_location(individual)
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
                    individual.command_array[mutation_location + i, 1] = \
                        pruned_param
                if p_2 == mutation_location:
                    individual.command_array[mutation_location + i, 2] = \
                        pruned_param

        self._last_mutation_location = mutation_location
        self._last_mutation_type = PRUNE_MUTATION

    @staticmethod
    def _get_random_prune_location(individual):
        utilized_commands = individual.get_utilized_commands()
        operators = []
        for i, (util, node) in enumerate(zip(utilized_commands,
                                             individual.command_array[:-1,
                                                                      0])):
            if util:
                if not IS_TERMINAL_MAP[node]:
                    operators.append(i)

        if not operators:
            return None

        index = np.random.randint(len(operators))
        return operators[index]

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

        individual.command_array[fork_space] = fork_commands.copy()

        self._last_mutation_location = 0
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
        individual.command_array[used_operators, 1] = \
            parameter_conversion[individual.command_array[used_operators, 1]]
        individual.command_array[used_operators, 2] = \
            parameter_conversion[individual.command_array[used_operators, 2]]
        # move commands
        individual.command_array[:, :] = \
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
