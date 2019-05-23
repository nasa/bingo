"""Component Generator for Agraph equations.

This module covers the random generation of components of an Acyclic graph
command stack. It can generate full commands or sub-components such as
operators, terminals, and their associated parameters.
"""
import logging
import numpy as np

from .AGraph import OPERATOR_NAMES
from ...Util.ProbabilityMassFunction import ProbabilityMassFunction
from ...Util.ArgumentValidation import argument_validation

LOGGER = logging.getLogger(__name__)


class ComponentGenerator:
    """Generates commands or components of a command for an AGraph stack

    Parameters
    ----------
    input_x_dimension : int
        number of independent variables
    num_initial_load_statements : int
        number of commands at the beginning of stack which are required to be
        "load" commands. Default 1
    terminal_probability : float [0.0-1.0]
        probability that a new node will be a terminal. Default 0.1
    constant_probability : float [0.0-1.0] (optional)
        probability that a new terminal will be a constant
    automatic_constant_optimization : bool
        Whether automatic constant optimization is used. Default True
    numerical_constant_range : float
        maximum and -minimum value for randomly generated numerical constants.
        Only used if automatic constant optimization is off in
        mutation/generation/crossover. Default 100.0
    numerical_constant_std : float
        standard deviation of modifications of numerical constants. Default
        numerical_constant_range / 100

    Attributes
    ----------
    input_x_dimension : int
        number of independent variables
    automatic_constant_optimization : bool
        Whether automatic constant optimization is used.
    """
    @argument_validation(input_x_dimension={">=": 0},
                         num_initial_load_statements={">=": 1},
                         terminal_probability={">=": 0.0, "<=": 1.0},
                         constant_probability={">=": 0.0, "<=": 1.0})
    def __init__(self, input_x_dimension, num_initial_load_statements=1,
                 terminal_probability=0.1,
                 constant_probability=None,
                 automatic_constant_optimization=True,
                 numerical_constant_range=100,
                 numerical_constant_std=None):

        self.input_x_dimension = input_x_dimension
        self._num_initial_load_statements = num_initial_load_statements

        self._terminal_pmf = self._make_terminal_pdf(constant_probability)
        self._operator_pmf = ProbabilityMassFunction()
        self._random_command_function_pmf = \
            self._make_random_command_pmf(terminal_probability)

        self.automatic_constant_optimization = automatic_constant_optimization
        self._numerical_constant_range = numerical_constant_range
        if numerical_constant_std is None:
            numerical_constant_std = numerical_constant_range / 100
        self._numerical_constant_std = numerical_constant_std

    def _make_terminal_pdf(self, constant_probability):
        if constant_probability is None:
            terminal_weight = [1, self.input_x_dimension]
        else:
            terminal_weight = [constant_probability,
                               1.0 - constant_probability]
        return ProbabilityMassFunction(items=[1, 0], weights=terminal_weight)

    def _make_random_command_pmf(self, terminal_probability):
        command_weights = [terminal_probability,
                           1.0 - terminal_probability]
        return ProbabilityMassFunction(items=[self._random_terminal_command,
                                              self._random_operator_command],
                                       weights=command_weights)

    def add_operator(self, operator_to_add, operator_weight=None):
        """Add an operator number to the set of possible operators

        Parameters
        ----------
        operator_to_add : int, str
            operator integer code (e.g. 2, 3) defined in Agraph operator maps
            or an operator string description (e.g. "+", "addition")
        operator_weight : number
                          relative weight of operator probability
        """
        if isinstance(operator_to_add, str):
            operator_number = self._get_operator_number_from_string(
                operator_to_add)
        else:
            operator_number = operator_to_add

        self._operator_pmf.add_item(operator_number, operator_weight)

    @staticmethod
    def _get_operator_number_from_string(operator_string):
        for operator_number, operator_names in OPERATOR_NAMES.items():
            if operator_string in operator_names:
                return operator_number
        raise ValueError("Could not find operator %s. " % operator_string)

    def random_command(self, stack_location):
        """Get a random command

        Parameters
        ----------
        stack_location : int
                         location in the stack for the command

        Returns
        -------
        array of int
            a random command in the form [node, parameter 1, parameter 2]

        """
        if stack_location < self._num_initial_load_statements:
            return self._random_terminal_command(stack_location)
        return self._random_command_function_pmf.draw_sample()(stack_location)

    def _random_operator_command(self, stack_location):
        return np.array([self.random_operator(),
                         self.random_operator_parameter(stack_location),
                         self.random_operator_parameter(stack_location)],
                        dtype=int)

    def random_operator(self):
        """Get a random operator

         Get a random operator from the list of possible operators.

        Returns
        -------
        int
            an operator number
        """
        return self._operator_pmf.draw_sample()

    @staticmethod
    def random_operator_parameter(stack_location):
        """Get random operator parameter

        Parameters
        ----------
        stack_location : int
                         location of command in stack

        Returns
        -------
        int
            parameter to be used in an operator command


        Notes
        -----
        The returned random operator parameter is guranteed to be less than
        stack_location.
        """
        return np.random.randint(stack_location)

    def _random_terminal_command(self, _=None):
        terminal = self.random_terminal()
        return np.array([terminal,
                         self.random_terminal_parameter(terminal),
                         self.random_terminal_parameter(terminal)],
                        dtype=int)

    def random_terminal(self):
        """Get a random terminal

         Get a random load-X or load-C terminal.

        Returns
        -------
        int
            terminal number (0 or 1)
        """
        return self._terminal_pmf.draw_sample()

    def random_terminal_parameter(self, terminal_number):
        """Get random terminal parameter

        Parameters
        ----------
        terminal_number : int
                          terminal number for which random parameter should be
                          generated

        Returns
        -------
        int
            parameter to be used in a terminal command
        """
        if terminal_number == 0:
            param = np.random.randint(self.input_x_dimension)
        else:
            param = -1
        return param

    def get_number_of_terminals(self):
        """Gets number of possible terminals

        Returns
        -------
        int :
            number of terminals
        """
        return len(self._terminal_pmf.items)

    def get_number_of_operators(self):
        """Gets number of possible operators

        Returns
        -------
        int :
            number of operators
        """
        return len(self._operator_pmf.items)

    def random_numerical_constant(self, near=None):
        """Gets a random numerical constant

        Parameters
        ----------
        near : float (optional)
            Value near which to generate a new random numerical value

        Returns
        -------
        float :
            A random numerical constant
        """
        if near is not None:
            return np.random.normal(near, self._numerical_constant_std)

        return np.random.uniform(-self._numerical_constant_range,
                                 self._numerical_constant_range)
