"""Definition of crossover between two acyclic graph individuals

This module contains the implementation of single point crossover between
acyclic graph individuals.
"""
import numpy as np

from ...Base.Crossover import Crossover


class AGraphCrossover(Crossover):
    """Crossover between acyclic graph individuals

    Parameters
    ----------
    component_generator : ComponentGenerator
        Component generator used for generating numerical constants.
    """

    def __init__(self, component_generator):
        self._component_generator = component_generator
        self._manual_constants = \
            not component_generator.automatic_constant_optimization

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

        if self._manual_constants:
            self._track_constants(parent_1, parent_2, child_1, cross_point)
            self._track_constants(parent_2, parent_1, child_2, cross_point)

        # TODO can we shift this responsibility to agraph?
        child_1.notify_command_array_modification()
        child_2.notify_command_array_modification()

        child_age = max(parent_1.genetic_age, parent_2.genetic_age)
        child_1.genetic_age = child_age
        child_2.genetic_age = child_age

        return child_1, child_2

    def _track_constants(self, parent_start, parent_end, child, cross_point):
        child.force_renumber_constants()
        child.constants = [0., ]*child.num_constants
        for i, (command, param1, _) in enumerate(child.command_array):
            if command == 1 and param1 != -1:
                if i < cross_point:
                    parent = parent_start
                else:
                    parent = parent_end
                old_constant_num = parent.command_array[i, 1]
                if old_constant_num == -1:
                    constant = \
                        self._component_generator.random_numerical_constant()
                else:
                    constant = parent.constants[old_constant_num]
                child.constants[param1] = constant
