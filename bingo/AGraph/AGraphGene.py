"""
This module contains most of the code necessary for the representation of an
acyclic graph (linear stack) in symbolic regression.  This version of the
Acyclic graph utilizes the bingocpp C++ library to do the function and
derivative evaluations

The current implementation has many hard coded sections. At the moment an
integer to operator mapping is how the command stack is parsed.
the current map is:
0: load column of X
1: load constant
2: + addition
3: - subtraction
4: * multiplication
5: / division (currently not divide-by-zero protected)
6: sin
7: cos
8: exp
9: log
10: pow
11: abs
12: sqrt
"""
import logging

import numpy as np

from ..GeneticRepresentation import EquationRepresentation

LOGGER = logging.getLogger(__name__)

# try:
#     sys.path.append("..")
#     from bingocpp.build import bingocpp as Backend
# except ImportError:
#     from . import AGraphBackend as Backend

from . import Backend as Backend

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


class AGraphGene(EquationRepresentation):
    """
    Acyclic Graph representation of an equation
    """
    def __init__(self):
        super().__init__()
        self._command_array = np.empty([0, 3])
        self._constants = []

    def needs_local_optimization(self):
        """find out whether constants need optimization"""
        util = Backend.get_utilized_commands(self._command_array)
        for i in range(self._command_array.shape[0]):
            if util[i]:
                if self._command_array[i][0] == 1:   # TODO hard coded (next too)
                    if self._command_array[i][1] == -1 or \
                            self._command_array[i][1] >= len(self._constants):
                        return True
        return False

    def get_number_local_optimization_params(self):
        """count constants and set up for  optimization"""

        # compile fitness function for optimization
        util = Backend.get_utilized_commands(self._command_array)
        const_num = 0
        for i in range(self._command_array.shape[0]):
            if util[i]:
                if self._command_array[i][0] == 1:              # TODO hard coded
                    self._command_array[i] = (1, const_num, const_num)
                    const_num += 1
        return const_num

    def set_local_optimization_params(self, consts):
        """manually set constants"""
        self._constants = consts

    def evaluate_equation_at(self, x):
        """evaluate the compiled stack"""
        try:
            f_of_x = Backend.simplify_and_evaluate(self._command_array,
                                                   x,
                                                   self._constants)
            return f_of_x
        except Exception as ex:
            LOGGER.error("Error in stack evaluation")
            self._raise_runtime_error(ex)

    def evaluate_equation_derivative_at(self, x):
        """evaluate the compiled stack"""
        try:
            f_of_x, df_dx = Backend.simplify_and_evaluate_with_derivative(
                self._command_array, x, self._constants, True)
            return f_of_x, df_dx
        except Exception as ex:
            LOGGER.error("Error in stack evaluation/deriv")
            self._raise_runtime_error(ex)

    def evaluate_equation_with_local_optimization_gradient_at(self, x):
        """evaluate the compiled stack"""
        try:
            f_of_x, df_dc = Backend.simplify_and_evaluate_with_derivative(
                self._command_array, x, self._constants, False)
            return f_of_x, df_dc
        except Exception as ex:
            LOGGER.error("Error in stack evaluation/const-deriv")
            self._raise_runtime_error(ex)

    def __str__(self):
        """overloaded string output"""
        print_str = "---full stack---\n"
        print_str += self._get_stack_string(short=False)
        print_str += "---small stack---\n"
        print_str += self._get_stack_string(short=True)
        return print_str

    def get_latex_string(self):
        """conversion to simplified latex string"""
        return self._get_formatted_string_using(LATEX_PRINT_MAP)

    def get_console_string(self):
        """conversion to simplified latex string"""
        return self._get_formatted_string_using(CONSOLE_PRINT_MAP)

    def get_complexity(self):
        """find number of commands that are utilized"""
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
