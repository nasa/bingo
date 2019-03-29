"""Generator of acyclic graph individuals.

This module contains the implementation of the generation of random acyclic
graph individuals.
"""
import numpy as np

from .AGraph import AGraph
from ...Base.Generator import Generator
from ...Util.ArgumentValidation import argument_validation


class AGraphGenerator(Generator):
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
