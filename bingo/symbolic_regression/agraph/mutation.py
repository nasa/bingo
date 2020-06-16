"""Mutation of acyclic graph individuals.

This module contains the implementation of mutation for acyclic graph
individuals, which is composed of 4 possible mutation strategies: command
mutation, node mutation, parameter mutation and pruning.
"""
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
        probability of node mutation. Default 0.3
    parameter_probability : float
        probability of parameter mutation. Default 0.3
    prune_probability : float
        probability of pruning. Default 0.2

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
                         prune_probability={">=": 0, "<=": 1})
    def __init__(self, component_generator, command_probability=0.2,
                 node_probability=0.3, parameter_probability=0.3,
                 prune_probability=0.2):
        self._component_generator = component_generator
        self._mutation_function_pmf = \
            ProbabilityMassFunction([self._mutate_command,
                                     self._mutate_node,
                                     self._mutate_parameters,
                                     self._prune_branch],
                                    [command_probability,
                                     node_probability,
                                     parameter_probability,
                                     prune_probability])
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
