"""Acyclic graph representation of an equation.


This module contains most of the code necessary for the representation of an
acyclic graph (linear stack) in symbolic regression.

Stack
-----

The stack is represented as Nx3 integer array. Where each row of the array
corresponds to a single command with form:

========  ===========  ===========
Operator  Parameter 1  Parameter 2
========  ===========  ===========

Where the parameters are a reference to the result of previously executed
commands (referenced by row number in the stack). The result of the last (N'th)
command in the stack is the evaluation of the equation.

Note: Parameter values have special meaning for two of the operators (0 and 1).

Operators
---------

An integer to operator mapping is how the command stack is parsed.
The current map is outlined below.

========  =======================================  =================
Operator  Name                                     Math
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

from ..GeneticIndividual import EquationIndividual
from . import Backend

LOGGER = logging.getLogger(__name__)

# try:
#     sys.path.append("..")
#     from bingocpp.build import bingocpp as Backend
# except ImportError:
#     from . import AGraphBackend as Backend


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


class AGraph(EquationIndividual):
    """Acyclic graph representation of an equation.

    Agraph is initialized with with empty stack and no constants.
    """
    def __init__(self):
        super().__init__()
        self._command_array = np.empty([0, 3])
        self._constants = []

    def needs_local_optimization(self):
        """The Agraph needs local optimization.

        Find out whether constants need optimization.

        Returns
        -------
        bool
            Constants need optimization
        """
        util = Backend.get_utilized_commands(self._command_array)
        for i in range(self._command_array.shape[0]):
            if util[i]:
                if self._command_array[i][0] == 1:
                    if self._command_array[i][1] == -1 or \
                            self._command_array[i][1] >= len(self._constants):
                        return True
        return False

    def get_number_local_optimization_params(self):
        """number of parameters for local optimization

        Count constants and set up for optimization

        Returns
        -------
        int
            Number of constants that need to be optimized
        """

        # compile fitness function for optimization
        util = Backend.get_utilized_commands(self._command_array)
        const_num = 0
        for i in range(self._command_array.shape[0]):
            if util[i]:
                if self._command_array[i][0] == 1:
                    self._command_array[i] = (1, const_num, const_num)
                    const_num += 1
        return const_num

    def set_local_optimization_params(self, params):
        """Set the local optimization parameters.

        Manually set optimized constants.

        Parameters
        ----------
        params : list of numeric
                 Values to set constants
        """
        self._constants = params

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
            f_of_x = Backend.simplify_and_evaluate(self._command_array,
                                                   x,
                                                   self._constants)
            return f_of_x
        except Exception as ex:
            LOGGER.error("Error in stack evaluation")
            self._raise_runtime_error(ex)

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
            f_of_x, df_dx = Backend.simplify_and_evaluate_with_derivative(
                self._command_array, x, self._constants, True)
            return f_of_x, df_dx
        except Exception as ex:
            LOGGER.error("Error in stack evaluation/deriv")
            self._raise_runtime_error(ex)

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
            f_of_x, df_dc = Backend.simplify_and_evaluate_with_derivative(
                self._command_array, x, self._constants, False)
            return f_of_x, df_dc
        except Exception as ex:
            LOGGER.error("Error in stack evaluation/const-deriv")
            self._raise_runtime_error(ex)

    def __str__(self):
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

    def get_complexity(self):
        """Calculate complexity of AGraph equation.

        Returns
        -------
        int
            number of utilized commands in stack
        """
        return np.count_nonzero(Backend.get_utilized_commands(
            self._command_array))

    def _get_stack_string(self, short=False):
        if short:
            rows_to_show = Backend.get_utilized_commands(self._command_array)
        else:
            rows_to_show = np.ones(self._command_array.shape[0], bool)
        tmp_str = ""
        for i, show_command in enumerate(rows_to_show):
            if show_command:
                tmp_str += self._get_stack_element_string(i)

        return tmp_str

    def _get_stack_element_string(self, command_index):
        node, param1, param2 = self._command_array[command_index]
        tmp_str = "(%d) <= " % command_index
        if node == 0:
            tmp_str += "X_%d" % param1
        elif node == 1:
            if param1 == -1:
                tmp_str += "C"
            else:
                tmp_str += "C_{} = {}".format(param1,
                                              self._constants[param1])
        else:
            tmp_str += STACK_PRINT_MAP[node].format(param1,
                                                    param2)
        tmp_str += "\n"
        return tmp_str

    def _get_formatted_string_using(self, format_dict):
        utilized_rows = Backend.get_utilized_commands(self._command_array)
        str_list = []
        for i, show_command in enumerate(utilized_rows):
            if show_command:
                tmp_str = self._get_latex_element_string(i, str_list,
                                                         format_dict)
            else:
                tmp_str = ""
            str_list.append(tmp_str)
        return str_list[-1]

    def _get_latex_element_string(self, command_index, str_list, format_dict):
        node, param1, param2 = self._command_array[command_index]
        if node == 0:
            tmp_str = "X_%d" % param1
        elif node == 1:
            if param1 == -1:
                tmp_str = "0"
            else:
                tmp_str = str(self._constants[param1])
        else:
            tmp_str = format_dict[node].format(str_list[param1],
                                               str_list[param2])
        return tmp_str

    def _raise_runtime_error(self, ex):
        LOGGER.error(str(self))
        LOGGER.error(str(ex))
        raise RuntimeError
