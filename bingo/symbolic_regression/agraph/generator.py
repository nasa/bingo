"""Generator of acyclic graph individuals.

This module contains the implementation of the generation of random acyclic
graph individuals.
"""
import warnings

import numpy as np

from .agraph import AGraph
from ...chromosomes.generator import Generator
from ...util.argument_validation import argument_validation

try:
    from bingocpp.build import bingocpp as bingocpp
except ImportError:
    bingocpp = None

class AGraphGenerator(Generator):
    """Generates acyclic graph individuals

    Parameters
    ----------
    agraph_size : int
                  command array size of the generated acyclic graphs
    component_generator : agraph.ComponentGenerator
                          Generator of stack components of agraphs
    """
    @argument_validation(agraph_size={">=": 1})
    def __init__(self, agraph_size, component_generator, cpp=False):
        self.agraph_size = agraph_size
        self.component_generator = component_generator
        self._manual_constants = \
            not component_generator.automatic_constant_optimization
        if cpp and not bingocpp:
            warnings.WarningMessage('Import error: bingocpp not loaded. \
                                     Program will default to python backend.')
            self._backend_generator_function = self._python_generator_function
        else:
            self._backend_generator_function = self._cpp_generator_function \
                if cpp else self._python_generator_function

    def __call__(self):
        """Generates random agraph individual.

        Fills stack based on random commands from the component generator.

        Returns
        -------
        Agraph
            new random acyclic graph individual
        """
        individual = self._backend_generator_function()
        individual.command_array = self._create_command_array()
        if self._manual_constants:
            individual.constants = self._generate_manual_constants(individual)
        return individual

    def _python_generator_function(self):
        return AGraph(self._manual_constants)

    def _cpp_generator_function(self):
        return bingocpp.AGraph(self._manual_constants)

    def _generate_manual_constants(self, individual):
        individual.force_renumber_constants()
        individual.notify_command_array_modification()
        num_consts = individual.get_number_local_optimization_params()
        return [self.component_generator.random_numerical_constant()
                for _ in range(num_consts)]

    def _create_command_array(self):
        command_array = np.empty((self.agraph_size, 3), dtype=int)
        for i in range(self.agraph_size):
            command_array[i] = self.component_generator.random_command(i)
        return command_array
