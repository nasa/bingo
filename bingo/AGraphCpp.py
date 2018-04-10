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
import random
import logging
from bingocpp.build import bingocpp
from scipy import optimize

import numpy as np

np.seterr(all='ignore')
LOGGER = logging.getLogger(__name__)


COMMAND_PRINT_MAP = {0: "X",
                     1: "C",
                     2: "+",
                     3: "-",
                     4: "*",
                     5: "/",
                     6: "sin",
                     7: "cos",
                     8: "exp",
                     9: "log",
                     10: "pow",
                     11: "abs",
                     12: "sqrt"}


class AGraphCppManipulator(object):
    """
    Manipulates AGraph objects for generation, crossover, mutation,
    and distance
    """

    def __init__(self, nvars, ag_size,
                 nloads=1, float_lim=10.0, terminal_prob=0.1):
        """
        Initialization of acyclic graph gene manipulator

        :param nvars: number of independent variables
        :param ag_size: length of command stack
        :param nloads: number of load operation which are required at the start
                       of stack
        :param float_lim: (0, max)  of floats which are generated
        :param terminal_prob: probability that a new node will be a terminal
        """
        self.nvars = nvars
        self.ag_size = ag_size
        self.nloads = nloads
        self.float_lim = float_lim
        self.terminal_prob = terminal_prob

        self.node_type_list = []
        self.terminal_inds = []
        self.operator_inds = []
        self.num_node_types = 0

        self.add_node_type(0)
        for _ in range(nvars-1):
            self.terminal_inds.append(0)
        self.add_node_type(1)

    def add_node_type(self, node_type):
        """
        Add a type of node to the set of allowed types

        :param node_type: acyclic graph node type which will be added to the
                          allowable set
        """
        if node_type not in self.node_type_list:
            self.node_type_list.append(node_type)
            if node_type <= 1:
                self.terminal_inds.append(self.num_node_types)
            else:
                self.operator_inds.append(self.num_node_types)
            self.num_node_types += 1

    def generate(self):
        """
        Generates random individual. Fills stack based on random
        nodes/terminals and random parameters

        :return: new random acyclic graph individual
        """
        indv = AGraphCpp()
        command_list = []
        for stack_loc in range(self.ag_size):
            if np.random.random() < self.terminal_prob \
                    or stack_loc < self.nloads:
                command_list.append(self.rand_terminal())
            else:
                command_list.append(self.rand_operator(stack_loc))
        indv.command_array = np.array(command_list, dtype=int)
        return indv

    def crossover(self, parent1, parent2):
        """
        Single point crossover

        :param parent1: first parent
        :param parent2: second parent
        :return: two children (new copies)
        """
        cx_point = np.random.randint(1, self.ag_size)
        child1 = parent1.copy()
        child2 = parent2.copy()
        child1.command_array[cx_point:, :] = parent2.command_array[cx_point:, :]
        child2.command_array[cx_point:, :] = parent1.command_array[cx_point:, :]
        child1.fitness = None
        child2.fitness = None
        child1.fit_set = False
        child2.fit_set = False
        return child1, child2

    def mutation(self, indv):
        """
        performs 1pt mutation, does not create copy of individual

        :param indv: individual which is mutated
        :return: mutated individual (not a new copy)
        """
        # pick mutation point within currently utilized commands
        util = indv.utilized_commands()
        loc = np.random.randint(sum(util))
        mut_point = [n for n, x in enumerate(util) if x][loc]
        orig_node_type, new_param1, new_param2 = indv.command_array[mut_point]

        # mutate operator (0.4) mutate params (0.4) prune branch (0.2)
        rand_val = np.random.random()
        # mutate operator
        if  rand_val < 0.4 and mut_point > self.nloads:
            new_type_found = False
            while not new_type_found:
                if np.random.random() < self.terminal_prob:
                    new_node_type, new_param1, new_param2 = self.rand_terminal()
                else:
                    new_node_type, new_param1, new_param2 = \
                    self.rand_operator(mut_point)
                new_type_found = new_node_type != orig_node_type or \
                                 orig_node_type <= 1           # TODO hardcoded
            indv.command_array[mut_point] = (new_node_type, new_param1,
                                             new_param2)

        # mutate parameters
        elif rand_val < 0.8:
            new_node_type = orig_node_type
            if orig_node_type <= 1:  # terminals               # TODO hardcoded
                new_params = self.mutate_terminal_param(new_node_type)
                new_param1 = new_params[0]
                new_param2 = new_param1
            else:  # operators
                new_param1, new_param2 = \
                self.rand_operator_params(2, mut_point)  # TODO hc

            indv.command_array[mut_point] = (new_node_type, new_param1,
                                             new_param2)

        # prune branch
        else:
            if orig_node_type > 1:  # operators only           # TODO hardcoded
                pruned_param = random.choice((new_param1, new_param2))
                for i in range(mut_point, len(indv.command_array)):
                    if mut_point in indv.command_array[i, 1:]:
                        p_0 = indv.command_array[i][1]        # TODO hardcoded
                        p_1 = indv.command_array[i][2]
                        if p_0 == mut_point:
                            p_0 = pruned_param
                        if p_1 == mut_point:
                            p_1 = pruned_param
                        indv.command_array[i] = (indv.command_array[i][0],
                                                 p_0, p_1)
        indv.fitness = None
        indv.fit_set = False
        return indv

    @staticmethod
    def distance(indv1, indv2):
        """
        Computes the distance (a measure of similarity) between two individuals

        :param indv1: first individual
        :param indv2: second individual
        :return: distance
        """
        dist = np.sum(indv1.command_array != indv2.command_array)

        return dist

    def dump(self, indv):
        """
        Dumps an individual to a pickleable object

        :param indv: individual which will be dumped
        :return: the individual in a pickleable format
        """

        return indv.command_array, indv.constants

    def load(self, indv_list):
        """
        Loads the individual from a pickleable object

        :param indv_list: individual in pickleable form
        :return: individual in normal form
        """
        indv = AGraphCpp()
        indv.command_array = indv_list[0]
        indv.constants = indv_list[1]

        i = 0
        while i < indv.command_array.shape[0]:
            if (indv.command_array[i, 0]) not in self.node_type_list:
                raise RuntimeError
            i += 1
        return indv

    @staticmethod
    def rand_operator_params(arity, stack_loc):
        """
        Produces random tuple for use as operator parameters

        :param arity: number of parameters needed
        :param stack_loc: location of command in stack
        :return: tuple of parameters
        """
        if stack_loc > 1:
            return tuple(np.random.randint(0, stack_loc, arity))
        else:
            return (0,)*arity

    def rand_operator_type(self):
        """
        Picks a random operator from the operator list

        :return: operator (acyclic graph node type)
        """
        node = self.node_type_list[random.choice(self.operator_inds)]
        return node

    def rand_operator(self, stack_loc):
        """
        Produces random operator and parameters. Chooses operator from list of
        allowable node types

        :param stack_loc: location of command in stack
        :return: random operator with parameters
        """
        node_type = self.rand_operator_type()
        params = self.rand_operator_params(2, stack_loc)       # TODO hardcoded
        return node_type, params[0], params[1]

    def rand_terminal_param(self, terminal):
        """
        Produces random terminal value, either input variable or float

        :return: terminal parameter
        """
        if terminal == 0:                                      # TODO hardcoded
            param = np.random.randint(self.nvars)
        else:
            param = -1                                         # TODO hardcoded
        return param,

    def mutate_terminal_param(self, terminal):
        """
        Produces random terminal value, either input variable or float
        Mutates floats by getting random variation of old param

        :return: terminal parameter
        """
        if terminal == 0:                                      # TODO hardcoded
            param = np.random.randint(self.nvars)
        else:
            param = -1                                         # TODO hardcoded
        return param,

    def rand_terminal(self):
        """
        Produces random terminal node and value

        :return: Load node, data index or constant index
        """
        node = self.node_type_list[random.choice(self.terminal_inds)]
        param = self.rand_terminal_param(node)
        return node, param[0], param[0]


class AGraphCpp(object):
    """
    Acyclic Graph representation of an equation
    """
    def __init__(self):
        self.command_array = np.empty([0, 3])
        self.constants = []
        self.fitness = None
        self.fit_set = False

    def copy(self):
        """return a deep copy"""
        dup = AGraphCpp()
        dup.fitness = self.fitness
        dup.fit_set = self.fit_set
        dup.constants = list(self.constants)
        dup.command_array = np.array(self.command_array)
        return dup

    def needs_optimization(self):
        """find out whether constants need optimization"""
        util = self.utilized_commands()
        for i in range(self.command_array.shape[0]):
            if util[i]:
                if self.command_array[i][0] == 1:   # TODO hard coded (next too)
                    if self.command_array[i][1] == -1 or \
                            self.command_array[i][1] >= len(self.constants):
                        return True
        return False

    def count_constants(self):
        """count constants and set up for  optimization"""

        # compile fitness function for optimization
        util = self.utilized_commands()
        const_num = 0
        for i in range(self.command_array.shape[0]):
            if util[i]:
                if self.command_array[i][0] == 1:              # TODO hard coded
                    self.command_array[i] = (1, const_num, const_num)
                    const_num += 1
        return const_num

    def set_constants(self, consts):
        """manually set constants"""
        self.constants = consts

    def evaluate(self, x):
        """evaluate the compiled stack"""
        try:
            # stack = bingocpp.CommandStack(self.command_list)
            f_of_x = bingocpp.simplify_and_evaluate(self.command_array,
                                                    x,
                                                    self.constants)
        except:
            LOGGER.error("Error in stack evaluation")
            LOGGER.error(str(self))
            exit(-1)
        return f_of_x

    def evaluate_deriv(self, x):
        """evaluate the compiled stack"""
        try:
            f_of_x, df_dx = bingocpp.simplify_and_evaluate_with_derivative(
                self.command_array, x, self.constants)
        except:
            LOGGER.error("Error in stack evaluation/deriv")
            LOGGER.error(str(self))
            exit(-1)
        return f_of_x, df_dx

    def __str__(self):
        """overloaded string output"""
        util = self.utilized_commands()
        print_str = "---full stack---\n"
        i = 0
        for node, param1, param2 in self.command_array:
            print_str += "(%d) <= " % i
            if node == 0:                                     # TODO hard coded
                print_str += COMMAND_PRINT_MAP[node] + "_%d\n" % param1
            elif node == 1:                                   # TODO hard coded
                print_str += COMMAND_PRINT_MAP[node]
                if param1 == -1:                           # TODO hard coded
                    print_str += "\n"
                else:
                    print_str += " = " + str(self.constants[param1]) + "\n"
            else:
                print_str += "(%d)" % param1 + " " + \
                             COMMAND_PRINT_MAP[node] \
                             + " " + "(%d)" % param2 + "\n"
            i += 1
        print_str += "---small stack---\n"
        i = 0
        for node, param1, param2 in self.command_array:
            if util[i]:
                print_str += "(%d) <= " % i
                if node == 0:                                 # TODO hard coded
                    print_str += COMMAND_PRINT_MAP[node] + "_%d\n" % param1
                elif node == 1:                               # TODO hard coded
                    print_str += COMMAND_PRINT_MAP[node]
                    if param1 == -1:                       # TODO hard coded
                        print_str += "\n"
                    else:
                        print_str += " = " + str(self.constants[param1]) + \
                                     "\n"
                else:
                    print_str += "(%d)"%param1 + " " + \
                                 COMMAND_PRINT_MAP[node] \
                                 + " " + "(%d)"%param2 + "\n"
            i += 1
        return print_str

    def latexstring(self):                           # TODO need to update this
        """conversion to simplified latex string"""
        util = self.utilized_commands()
        str_list = []
        for i, (node, param1, param2) in enumerate(self.command_array):
                                                   # TODO hard coded whole func
            tmp_str = ""
            if util[i]:
                if node == 0:
                    tmp_str = "X_%d" % param1
                elif node == 1:
                    if param1 == -1:
                        tmp_str = "0"
                    else:
                        tmp_str = str(self.constants[param1])
                elif node == 2:
                    tmp_str = "%s + %s" % (str_list[param1],
                                           str_list[param2])
                elif node == 3:
                    tmp_str = "%s - (%s)" % (str_list[param1],
                                             str_list[param2])
                elif node == 4:
                    tmp_str = "(%s)(%s)" % (str_list[param1],
                                            str_list[param2])
                elif node == 5:
                    tmp_str = "\\frac{%s}{%s}" % (str_list[param1],
                                                  str_list[param2])
                elif node == 6:
                    tmp_str = "\\sin{%s}" % (str_list[param1])
                elif node == 7:
                    tmp_str = "\\cos{%s}" % (str_list[param1])
                elif node == 8:
                    tmp_str = "\\exp{%s}" % (str_list[param1])
                elif node == 9:
                    tmp_str = "\\log{%s}" % (str_list[param1])
                elif node == 10:
                    tmp_str = "(%s)^{(%s)}" % (str_list[param1],
                                               str_list[param2])
                elif node == 11:
                    tmp_str = "|{%s}|" % (str_list[param1])
                elif node == 12:
                    tmp_str = "\\sqrt{%s}" % (str_list[param1])
            str_list.append(tmp_str)
        return str_list[-1]

    def utilized_commands(self):
        """find which commands are utilized"""
        util = [False]*self.command_array.shape[0]
        util[-1] = True
        for i in range(1, self.command_array.shape[0]):
            if util[-i] and self.command_array[-i][0] > 1:
                util[self.command_array[-i][1]] = True
                util[self.command_array[-i][2]] = True
        return util

    def complexity(self):
        """find number of commands that are utilized"""
        return sum(self.utilized_commands())
