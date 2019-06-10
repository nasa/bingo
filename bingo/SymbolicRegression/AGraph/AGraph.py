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


Attributes
----------
IS_ARITY_2_MAP : dict {int: bool}
                 A map of node number to boolean that states whether the
                 node has arity 2 (as opposed to 1)
IS_TERMINAL_MAP : dict {int: bool}
                 A map of node number to boolean that states whether the
                 node is a terminal
STACK_PRINT_MAP : dict {int: str}
                  A map of node number to a format string for stack output
LATEX_PRINT_MAP : dict {int: str}
                  A map of node number to a format string for latex output
CONSOLE_PRINT_MAP : dict {int: str}
                  A map of node number to a format string for console output
"""
import logging
import numpy as np

from ..Equation import Equation
from ...Base import ContinuousLocalOptimization

try:
    from bingocpp.build import bingocpp as Backend
except ImportError:
    from . import Backend

LOGGER = logging.getLogger(__name__)


STACK_PRINT_MAP = {2: "({}) + ({})",
                   3: "({}) - ({})",
                   4: "({}) * ({})",
                   5: "({}) / ({}) ",
                   6: "sin ({})",
                   7: "cos ({})",
                   8: "exp ({})",
                   9: "log ({})",
                   10: "({}) ^ ({})",
                   11: "abs ({})",
                   12: "sqrt ({})"}

LATEX_PRINT_MAP = {2: "{} + {}",
                   3: "{} - ({})",
                   4: "({})({})",
                   5: "\\frac{{ {} }}{{ {} }}",
                   6: "sin{{ {} }}",
                   7: "cos{{ {} }}",
                   8: "exp{{ {} }}",
                   9: "log{{ {} }}",
                   10: "({})^{{ ({}) }}",
                   11: "|{}|",
                   12: "\\sqrt{{ {} }}"}

CONSOLE_PRINT_MAP = {2: "{} + {}",
                     3: "{} - ({})",
                     4: "({})({})",
                     5: "({})/({}) ",
                     6: "sin({})",
                     7: "cos({})",
                     8: "exp({})",
                     9: "log({})",
                     10: "({})^({})",
                     11: "|{}|",
                     12: "sqrt({})"}

IS_ARITY_2_MAP = {0: False,
                  1: False,
                  2: True,
                  3: True,
                  4: True,
                  5: True,
                  6: False,
                  7: False,
                  8: False,
                  9: False,
                  10: True,
                  11: False,
                  12: False}

IS_TERMINAL_MAP = {0: True,
                   1: True,
                   2: False,
                   3: False,
                   4: False,
                   5: False,
                   6: False,
                   7: False,
                   8: False,
                   9: False,
                   10: False,
                   11: False,
                   12: False}

OPERATOR_NAMES = {0: ["load", "x"],
                  1: ["constant", "c"],
                  2: ["add", "addition", "+"],
                  3: ["subtract", "subtraction", "-"],
                  4: ["multiply", "multiplication", "*"],
                  5: ["divide", "division", "/"],
                  6: ["sine", "sin"],
                  7: ["cosine", "cos"],
                  8: ["exponential", "exp", "e"],
                  9: ["logarithm", "log"],
                  10: ["power", "pow", "^"],
                  11: ["absolute value", "||", "|"],
                  12: ["square root", "sqrt"]}


class AGraph(Equation, ContinuousLocalOptimization.ChromosomeInterface):
    """Acyclic graph representation of an equation.

    Agraph is initialized with with empty command array and no constants.

    Attributes
    ----------
    command_array
    constants
    num_constants
    """
    def __init__(self, genetic_age=0, fitness=None, fit_set=False,
                 command_array=np.empty([0, 3], dtype=int),
                 short_command_array=np.empty([0, 3], dtype=int),
                 constants=None,
                 needs_opt=False,
                 num_constants=0,
                 manual_constants=False):
        super().__init__(genetic_age, fitness, fit_set)
        self._command_array = command_array
        self._short_command_array = short_command_array
        if constants is None:
            constants = []
        self.constants = constants
        self._needs_opt = needs_opt
        self.num_constants = num_constants
        self._manual_constants = manual_constants

    @property
    def command_array(self):
        """Nx3 array of int: acyclic graph stack.

        Notes
        -----
        Setting the command stack automatically resets fitness
        """
        return self._command_array

    @command_array.setter
    def command_array(self, command_array):
        self._command_array = command_array
        self._fitness = None
        self.fit_set = False
        self._process_modified_command_array()

    def notify_command_array_modification(self):
        """Notify individual of inplace modification of its command array"""
        self._fitness = None
        self.fit_set = False
        self._process_modified_command_array()

    def force_renumber_constants(self):
        """force the renumbering of constants"""
        util = self.get_utilized_commands()
        self._renumber_constants(util)

    def _process_modified_command_array(self):
        if not self._manual_constants:
            util = self.get_utilized_commands()

            self._needs_opt = self._check_optimization_requirement(util)
            if self._needs_opt:
                self._renumber_constants(util)

        self._short_command_array = Backend.simplify_stack(self._command_array)

    def _check_optimization_requirement(self, util):
        for i in range(self._command_array.shape[0]):
            if util[i]:
                if self._command_array[i][0] == 1:
                    if self._command_array[i][1] == -1 or \
                            self._command_array[i][1] >= len(self.constants):
                        return True
        return False

    def _renumber_constants(self, util):
        const_num = 0
        for i in range(self._command_array.shape[0]):
            if self._command_array[i][0] == 1:
                if util[i]:
                    self._command_array[i] = (1, const_num, const_num)
                    const_num += 1
                else:
                    self._command_array[i] = (1, -1, -1)
        self.num_constants = const_num

    def needs_local_optimization(self):
        """The Agraph needs local optimization.

        Find out whether constants need optimization.

        Returns
        -------
        bool
            Constants need optimization
        """
        return self._needs_opt

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

    def get_number_local_optimization_params(self):
        """number of parameters for local optimization

        Count constants and set up for optimization

        Returns
        -------
        int
            Number of constants that need to be optimized
        """
        return self.num_constants

    def set_local_optimization_params(self, params):
        """Set the local optimization parameters.

        Manually set optimized constants.

        Parameters
        ----------
        params : list of numeric
                 Values to set constants
        """
        self.constants = params
        self._needs_opt = False

    def evaluate_equation_at(self, x):
        """Evaluate the AGraph equation.

        Evaluation of the Agraph equation at points x.

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
        try:
            f_of_x = Backend.evaluate(self._short_command_array,
                                      x, self.constants)
            return f_of_x
        except (ArithmeticError, OverflowError, ValueError,
                FloatingPointError) as err:
            LOGGER.warning("%s in stack evaluation", err)
            return np.full(x.shape, np.nan)

    def evaluate_equation_with_x_gradient_at(self, x):
        """Evaluate Agraph and get its derivatives.

        Evaluate the AGraph equation at x and the gradient of x.

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
        try:
            f_of_x, df_dx = Backend.evaluate_with_derivative(
                self._short_command_array, x, self.constants, True)
            return f_of_x, df_dx
        except (ArithmeticError, OverflowError, ValueError,
                FloatingPointError) as err:
            LOGGER.warning("%s in stack evaluation/deriv", err)
            nan_array = np.full(x.shape, np.nan)
            return nan_array, np.array(nan_array)

    def evaluate_equation_with_local_opt_gradient_at(self, x):
        """Evaluate Agraph and get its derivatives.

        Evaluate the AGraph equation at x and get the gradient of constants.
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
        try:
            f_of_x, df_dc = Backend.evaluate_with_derivative(
                self._short_command_array, x, self.constants, False)
            return f_of_x, df_dc
        except (ArithmeticError, OverflowError, ValueError,
                FloatingPointError) as err:
            LOGGER.warning("%s in stack evaluation/const-deriv", err)
            nan_array = np.full((x.shape[0], len(self.constants)), np.nan)
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
        return self._get_formatted_string_using(LATEX_PRINT_MAP)

    def get_console_string(self):
        """Console version of Agraph equation.

        Returns
        -------
        str
            Equation in simple form
        """
        return self._get_formatted_string_using(CONSOLE_PRINT_MAP)

    def get_stack_string(self):
        """Stack output of Agraph equation.

        Returns
        -------
        str
            equation in stack form and simplified stack form
        """
        print_str = "---full stack---\n"
        print_str += self._get_stack_string(short=False)
        print_str += "---small stack---\n"
        print_str += self._get_stack_string(short=True)
        return print_str

    def get_complexity(self):
        """Calculate complexity of AGraph equation.

        Returns
        -------
        int
            number of utilized commands in stack
        """
        return self._short_command_array.shape[0]

    def _get_stack_string(self, short=False):
        if short:
            stack = self._short_command_array
        else:
            stack = self._command_array
        tmp_str = ""
        for i, stack_element in enumerate(stack):
            tmp_str += self._get_stack_element_string(i, stack_element)

        return tmp_str

    def _get_stack_element_string(self, command_index, stack_element):
        node, param1, param2 = stack_element
        tmp_str = "(%d) <= " % command_index
        if node == 0:
            tmp_str += "X_%d" % param1
        elif node == 1:
            if param1 == -1 or param1 >= len(self.constants):
                tmp_str += "C"
            else:
                tmp_str += "C_{} = {}".format(param1,
                                              self.constants[param1])
        else:
            tmp_str += STACK_PRINT_MAP[node].format(param1,
                                                    param2)
        tmp_str += "\n"
        return tmp_str

    def _get_formatted_string_using(self, format_dict):
        str_list = []
        for stack_element in self._short_command_array:
            tmp_str = self._get_formatted_element_string(stack_element,
                                                         str_list,
                                                         format_dict)
            str_list.append(tmp_str)
        return str_list[-1]

    def _get_formatted_element_string(self, stack_element, str_list,
                                      format_dict):
        node, param1, param2 = stack_element
        if node == 0:
            tmp_str = "X_%d" % param1
        elif node == 1:
            if param1 == -1 or param1 >= len(self.constants):
                tmp_str = "?"
            else:
                tmp_str = str(self.constants[param1])
        else:
            tmp_str = format_dict[node].format(str_list[param1],
                                               str_list[param2])
        return tmp_str

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
        duplicate = AGraph(genetic_age=self.genetic_age,
                           fitness=self._fitness, fit_set=self.fit_set,
                           command_array=np.copy(self._command_array),
                           short_command_array=np.copy(self._short_command_array),
                           constants=list(self.constants),
                           needs_opt=self._needs_opt,
                           num_constants=self.num_constants,
                           manual_constants=self._manual_constants)
        return duplicate
