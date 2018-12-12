"""Generator of Agraph individuals

This module covers the random generation of acyclic graph individuals.
"""
import numpy as np

from .AGraph import AGraph
from .BackendNodes import IS_TERMINAL_MAP, IS_ARITY_2_MAP
from .. import IndividualVariation
from ..Util.ProbabilityMassFunction import ProbabilityMassFunction
from ..Util.ArgumentValidation import argument_validation


class Generation(IndividualVariation.Generation):
    """Generates acyclic graph individuals

    Parameters
    ----------
    agraph_size : int
                  command array size of the generated acyclic graphs
    component_generator : AGraph.ComponentGenerator
                          Generator of stack components of agraphs
    """
    @argument_validation(agraph_size={">=": 1})
    def __init__(self, agraph_size, component_generator):
        self.agraph_size = agraph_size
        self.component_generator = component_generator

    def __call__(self):
        """Generates random agraph individual.

        Fills stack based on random commands from the component generator.

        Returns
        -------
        Agraph
            new random acyclic graph individual
        """
        individual = AGraph()
        individual.command_array = self._create_command_array()
        return individual

    def _create_command_array(self):
        command_array = np.empty((self.agraph_size, 3), dtype=int)
        for i in range(self.agraph_size):
            command_array[i] = self.component_generator.random_command(i)
        return command_array


class Crossover(IndividualVariation.Crossover):
    """Crossover between acyclic graph individuals"""

    def __init__(self):
        pass

    def __call__(self, parent_1, parent_2):
        """Single point crossover.

        Parameters
        ----------
        parent_1 : Agraph
                   The first parent individual
        parent_2 : Agraph
                   The second parent individual

        Returns
        -------
        tuple(Agraph, Agraph) :
            The two children from the crossover.
        """

        child_1 = parent_1.copy()
        child_2 = parent_2.copy()

        ag_size = parent_1.command_array.shape[0]
        cross_point = np.random.randint(1, ag_size-1)
        child_1.command_array[cross_point:] = \
            parent_2.command_array[cross_point:]
        child_2.command_array[cross_point:] = \
            parent_1.command_array[cross_point:]

        # TODO can we shift this responsibility to agraph?
        child_1.notify_command_array_modification()
        child_2.notify_command_array_modification()

        child_age = max(parent_1.genetic_age, parent_2.genetic_age)
        child_1.genetic_age = child_age
        child_2.genetic_age = child_age

        return child_1, child_2


class Mutation(IndividualVariation.Mutation):
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
                          probability of command mutation
    node_probability : float
                           probability of node mutation
    parameter_probability : float
                            probability of parameter mutation
    prune_probability : float
                        probability of pruning

    Notes
    -----
    The input probabilities are normalized if their sum is not equal to 1.

    Mutation can result in no change if, for instance, a prune mutation is
    executed on a Agraph utilizing only a single terminal.
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

        # TODO can we shift this responsibility to agraph?
        child.notify_command_array_modification()

        return child

    @staticmethod
    def _get_random_mutation_location(child):
        utilized_commands = child.get_utilized_commands()
        index = np.random.randint(sum(utilized_commands))
        return [n for n, x in enumerate(utilized_commands) if x][index]

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

    def _mutate_node(self, individual):
        mutation_location = self._get_random_mutation_location(individual)
        old_node = individual.command_array[mutation_location, 0]
        mutated_command = individual.command_array[mutation_location]
        is_terminal = IS_TERMINAL_MAP[old_node]

        if self._is_new_node_possible(is_terminal):
            self._force_mutated_node(is_terminal, mutated_command, old_node)

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
        mutation_location = self._get_random_mutation_location(individual)
        old_command = np.copy(individual.command_array[mutation_location])
        mutated_command = individual.command_array[mutation_location]

        if self._is_new_param_possible(old_command[0], mutation_location):
            self._force_mutated_parameters(mutated_command, old_command)

    def _force_mutated_parameters(self, mutated_command, old_command):
        is_terminal = IS_TERMINAL_MAP[old_command[0]]
        unique_params = False
        while not unique_params:
            self._randomize_parameters(is_terminal, mutated_command)
            if mutated_command[0] == 1:  # TODO hard coded info about node map
                break
            unique_params = not np.array_equal(mutated_command,
                                               old_command)

    def _is_new_param_possible(self, node, mutation_location):
        # TODO hard coded info about node map
        if node == 0:
            return self._component_generator.input_x_dimension > 1
        if node == 1:
            return True
        return mutation_location > 1

    def _randomize_parameters(self, is_terminal, mutated_command):
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
                    mutated_command[0])
            if IS_ARITY_2_MAP[mutated_command[0]]:
                mutated_command[2] = \
                    self._component_generator.random_operator_parameter(
                        mutated_command[0])

    @staticmethod
    def _prune_branch(individual):
        mutation_location = Mutation._get_random_prune_location(individual)
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
