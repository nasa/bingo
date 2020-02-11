"""Acyclic graph representation of an equation.


This module contains most of the code necessary for the representation of an
acyclic graph (linear stack) in symbolic regression. 
Stack
-----

The stack is represented as Nx3 integer array. Where each row of the array
corresponds to a single command with form:

========  ===========  ===========
Node      Parameter 1  Parameter 2
========  ===========  ===========

Where the parameters are a reference to the result of previously executed
commands (referenced by row number in the stack). The result of the last (N'th)
command in the stack is the evaluation of the equation.

Note: Parameter values have special meaning for two of the nodes (0 and 1).

Nodes
---------

An integer to node mapping is how the command stack is parsed.
The current map is outlined below.

========  =======================================  =================
Node      Name                                     Math
========  =======================================  =================
0         load p1'th column of x                   :math:`x_{p1}`
1         load p1'th constant                      :math:`c_{p1}`
2         addition                                 :math:`p1 + p2`
3         subtraction                              :math:`p1 - p2`
4         multiplication                           :math:`p1 - p2`
5         division (not divide-by-zero protected)  :math:`p1 / p2`
6         sine                                     :math:`sin(p1)`
7         cosine                                   :math:`cos(p1)`
8         exponential                              :math:`exp(p1)`
9         logarithm                                :math:`log(|p1|)`
10        power                                    :math:`|p1|^{p2}`
11        absolute value                           :math:`|p1|`
12        square root                              :math:`sqrt(|p1|)`
========  =======================================  =================
"""
import logging
import numpy as np

from .string_generation import get_formatted_string
from ..equation import Equation
from ...local_optimizers import continuous_local_opt

try:
    from bingocpp.build import bingocpp as Backend
except ImportError as e:
    print(e)
    from . import backend as Backend

LOGGER = logging.getLogger(__name__)


class AGraph(Equation, continuous_local_opt.ChromosomeInterface):
    """Acyclic graph representation of an equation.

    Agraph is initialized with with empty command array and no constants.

    Attributes
    ----------
    command_array
    constants
    num_constants
    """
    def __init__(self):
        super().__init__()
        self._command_array = np.empty([0, 3], dtype=int)

        self._simplified_command_array = np.empty([0, 3], dtype=int)
        self._simplified_constants = []

        self._needs_opt = False
        self._modified = False
        self._used_constant_commands = []

    @staticmethod
    def is_cpp():
        return False

    @property
    def command_array(self):
        """Nx3 array of int: acyclic graph stack.

        Notes
        -----
        Setting the command stack automatically resets fitness
        """
        self._command_array.flags.writeable = False
        return self._command_array

    @command_array.setter
    def command_array(self, command_array):
        self._command_array = command_array
        self._notify_modification()

    @property
    def mutable_command_array(self):
        """Nx3 array of int: acyclic graph stack.

        Notes
        -----
        Setting the command stack automatically resets fitness
        """
        self._command_array.flags.writeable = True
        self._notify_modification()
        return self._command_array

    def _notify_modification(self):
        self._modified = True
        self._fitness = None
        self._fit_set = False

    def _update(self):
        self._simplified_command_array = \
            Backend.simplify_stack(self._command_array)
        # TODO hard coded info about node map
        const_commands = self._simplified_command_array[:, 0] == 1
        num_const = np.count_nonzero(const_commands)
        self._simplified_command_array[const_commands, 1] = np.arange(num_const)
        self._simplified_command_array[const_commands, 2] = np.arange(num_const)

        optimization_aggression = 0
        if optimization_aggression == 0 \
                and num_const <= len(self._simplified_constants):
            self._simplified_constants = self._simplified_constants[:num_const]
        elif optimization_aggression == 1 \
                and num_const == len(self._simplified_constants):
            self._simplified_constants = self._simplified_constants[:num_const]
        else:
            self._simplified_constants = (1.0,) * num_const
            if num_const > 0:
                self._needs_opt = True
        self._modified = False

    def needs_local_optimization(self):
        """The Agraph needs local optimization.

        Find out whether constants need optimization.

        Returns
        -------
        bool
            Constants need optimization
        """
        if self._modified:
            self._update()
        return self._needs_opt

    def get_number_local_optimization_params(self):
        """number of parameters for local optimization

        Count constants and set up for optimization

        Returns
        -------
        int
            Number of constants that need to be optimized
        """
        if self._modified:
            self._update()
        return len(self._simplified_constants)

    def set_local_optimization_params(self, params):
        """Set the local optimization parameters.

        Manually set optimized constants.

        Parameters
        ----------
        params : list of numeric
                 Values to set constants
        """
        self._simplified_constants = tuple(params)
        self._needs_opt = False

    def get_utilized_commands(self):
        """"Find which commands are utilized.

        Find the commands in the command array of the agraph upon which the
        last command relies. This is inclusive of the last command.

        Returns
        -------
        list of bool of length N
            Boolean values for whether each command is utilized.
        """
        return Backend.get_utilized_commands(self._command_array)

    def evaluate_equation_at(self, x):
        """Evaluate the agraph equation.

        evaluation of the Agraph equation at points x.

        Parameters
        ----------
        x : MxD array of numeric.
            Values at which to evaluate the equations. D is the number of
            dimensions in x and M is the number of data points in x.

        Returns
        -------
        Mx1 array of numeric
            :math:`f(x)`
        """
        if self._modified:
            self._update()
        try:
            f_of_x = Backend.evaluate(self._simplified_command_array,
                                      x, self._simplified_constants)
            return f_of_x
        except (ArithmeticError, OverflowError, ValueError,
                FloatingPointError) as err:
            LOGGER.warning("%s in stack evaluation", err)
            return np.full(x.shape, np.nan)

    def evaluate_equation_with_x_gradient_at(self, x):
        """Evaluate Agraph and get its derivatives.

        Evaluate the agraph equation at x and the gradient of x.

        Parameters
        ----------
        x : MxD array of numeric.
            Values at which to evaluate the equations. D is the number of
            dimensions in x and M is the number of data points in x.

        Returns
        -------
        tuple(Mx1 array of numeric, MxD array of numeric)
            :math:`f(x)` and :math:`df(x)/dx_i`
        """
        if self._modified:
            self._update()
        try:
            f_of_x, df_dx = Backend.evaluate_with_derivative(
                self._simplified_command_array, x,
                self._simplified_constants, True)
            return f_of_x, df_dx
        except (ArithmeticError, OverflowError, ValueError,
                FloatingPointError) as err:
            LOGGER.warning("%s in stack evaluation/deriv", err)
            nan_array = np.full(x.shape, np.nan)
            return nan_array, np.array(nan_array)

    def evaluate_equation_with_local_opt_gradient_at(self, x):
        """Evaluate Agraph and get its derivatives.

        Evaluate the agraph equation at x and get the gradient of constants.
        Constants are of length L.

        Parameters
        ----------
        x : MxD array of numeric.
            Values at which to evaluate the equations. D is the number of
            dimensions in x and M is the number of data points in x.

        Returns
        -------
        tuple(Mx1 array of numeric, MxL array of numeric)
            :math:`f(x)` and :math:`df(x)/dc_i`
        """
        if self._modified:
            self._update()
        try:
            f_of_x, df_dc = Backend.evaluate_with_derivative(
                self._simplified_command_array, x,
                self._simplified_constants, False)
            return f_of_x, df_dc
        except (ArithmeticError, OverflowError, ValueError,
                FloatingPointError) as err:
            LOGGER.warning("%s in stack evaluation/const-deriv", err)
            nan_array = np.full((x.shape[0], len(self._simplified_constants)),
                                np.nan)
            return nan_array, np.array(nan_array)

    def __str__(self):
        """Console string output of Agraph equation.

        Returns
        -------
        str
            equation in console form
        """
        return self.get_console_string()

    def get_latex_string(self):
        """Latex interpretable version of Agraph equation.

        Returns
        -------
        str
            Equation in latex form
        """
        if self._modified:
            self._update()
        return get_formatted_string("latex", self._simplified_command_array,
                                    self._simplified_constants)

    def get_console_string(self):
        """Console version of Agraph equation.

        Returns
        -------
        str
            Equation in simple form
        """
        if self._modified:
            self._update()
        return get_formatted_string("console", self._simplified_command_array,
                                    self._simplified_constants)

    def get_stack_string(self):
        """Stack output of Agraph equation.

        Returns
        -------
        str
            equation in stack form and simplified stack form
        """
        if self._modified:
            self._update()
        print_str = "---full stack---\n"
        print_str += get_formatted_string("stack", self._command_array,
                                          tuple())
        print_str += "---small stack---\n"
        print_str += get_formatted_string("stack",
                                          self._simplified_command_array,
                                          self._simplified_constants)
        return print_str

    def get_complexity(self):
        """Calculate complexity of agraph equation.

        Returns
        -------
        int
            number of utilized commands in stack
        """
        if self._modified:
            self._update()
        return self._simplified_command_array.shape[0]

    def distance(self, chromosome):
        """Computes the distance to another Agraph

        Distance is a measure of similarity of the two command_arrays

        Parameters
        ----------
        chromosome : Agraph
                     The individual to which distance will be calculated

        Returns
        -------
         : int
            distance from self to individual
        """
        dist = np.sum(self.command_array != chromosome.command_array)
        return dist

    def __deepcopy__(self, memodict=None):
        duplicate = AGraph()
        self._copy_agraph_values_to_new_graph(duplicate)
        return duplicate

    def _copy_agraph_values_to_new_graph(self, agraph_duplicate):
        agraph_duplicate._genetic_age = self._genetic_age
        agraph_duplicate._fitness = self._fitness
        agraph_duplicate._fit_set = self._fit_set
        agraph_duplicate._command_array = np.copy(self.command_array)
        agraph_duplicate._simplified_command_array = \
            np.copy(self._simplified_command_array)
        agraph_duplicate._simplified_constants = \
            tuple(self._simplified_constants)
        agraph_duplicate._needs_opt = self._needs_opt
        agraph_duplicate._modified = self._modified
