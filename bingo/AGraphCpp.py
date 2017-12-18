"""
This module contains most of the code necessary for the representation of an
acyclic graph (linear stack) in symbolic regression.  This version of the
Acyclic graph utilizes the bingocpp C++ library to do the function and
derivative evaluations

The current implementation has many hard coded sections. At the moment an
integer to operator mapping is haw the command stack is parsed.
the current map is:
0: load column of X
1: load constant
2: + addition
3: - subtraction
4: * multiplication
5: / division (currently not divide-by-zero protected)
"""
import random
from bingo import bingocpp
from scipy import optimize

import numpy as np
np.seterr(all='ignore')


COMMAND_PRINT_MAP = {0: "X",
                     1: "C",
                     2: "+",
                     3: "-",
                     4: "*",
                     5: "/"}


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
        for stack_loc in range(self.ag_size):
            if np.random.random() < self.terminal_prob \
                    or stack_loc < self.nloads:
                indv.command_list.append(self.rand_terminal())
            else:
                indv.command_list.append(self.rand_operator(stack_loc))
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
        child1.command_list[cx_point:] = parent2.command_list[cx_point:]
        child2.command_list[cx_point:] = parent1.command_list[cx_point:]
        child1.compiled = False
        child2.compiled = False
        child1.fitness = None
        child2.fitness = None
        return child1, child2

    def mutation(self, indv):
        """
        performs 1pt mutation, does not create copy of individual

        :param indv: individual which is mutated
        :return: mutated individual (not a new copy)
        """
        indv.compiled = False
        # pick mutation point within currently utilized commands
        util = indv.utilized_commands()
        loc = np.random.randint(sum(util))
        mut_point = [n for n, x in enumerate(util) if x][loc]
        orig_node_type, _ = indv.command_list[mut_point]

        # randomly change operation or parameter with equal prob
        if np.random.random() < 0.5 and mut_point > self.nloads:  # op change
            new_type_found = False
            while not new_type_found:
                if np.random.random() < self.terminal_prob:
                    new_node_type, new_params = self.rand_terminal()
                else:
                    new_node_type, new_params = self.rand_operator(mut_point)
                new_type_found = new_node_type != orig_node_type or \
                                 orig_node_type <= 1           # TODO hardcoded
            # only use new params if needed
            # if not new_node_type.terminal:  # don't worry about this 4 terms
            #     tmp = ()
            #     for i in range(new_node_type.arity):
            #         if i < orig_node_type.arity:
            #             tmp += (orig_params[i],)
            #         else:
            #             tmp += (new_params[i],)
            #     new_params = tmp

        else:  # parameter change
            new_node_type = orig_node_type
            if orig_node_type <= 1:  # terminals               # TODO hardcoded
                new_params = self.mutate_terminal_param(new_node_type)
            else:  # operators
                new_params = self.rand_operator_params(2, mut_point)  # TODO hc

        indv.command_list[mut_point] = (new_node_type, new_params)
        indv.compiled = False
        indv.fitness = None
        return indv

    @staticmethod
    def distance(indv1, indv2):
        """
        Computes the distance (a measure of similarity) between two individuals

        :param indv1: first individual
        :param indv2: second individual
        :return: distance
        """
        dist = 0
        for (command1, params1), (command2, params2) in \
                zip(indv1.command_list, indv2.command_list):
            if command1 != command2:
                dist += 1

            if len(params1) == 1 or len(params2) == 1:
                if params1[0] != params2[0]:
                    dist += 2
            else:
                if params1[0] != params2[0]:
                    dist += 1
                if params1[1] != params2[1]:
                    dist += 1

            # if params1 != params2:
            #     dist += 1

            # for p_1, p_2 in zip((params1+params1)[:2],(params2+params2)[:2]):
            #     if p_1 != p_2:
            #         dist += 1

        return dist

    def dump(self, indv):
        """
        Dumps an individual to a pickleable object

        :param indv: individual which will be dumped
        :return: the individual in a pickleable format
        """
        command_list = []
        for node, params in indv.command_list:
            ind = self.node_type_list.index(node)
            command_list.append((ind, params))
        return command_list, indv.constants

    def load(self, indv_list):
        """
        Loads the individual from a pickleable object

        :param indv_list: individual in pickleable form
        :return: individual in normal form
        """
        indv = AGraphCpp()
        indv.constants = indv_list[1]
        for node_num, params in indv_list[0]:
            if node_num in range(len(self.node_type_list)):  # node
                indv.command_list.append((self.node_type_list[node_num],
                                          params))
            else:
                raise RuntimeError
        return indv

    @staticmethod
    def rand_operator_params(arity, stack_loc):
        """
        Aroduces random tuple for use as operator parameters

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
        return node_type, params

    def rand_terminal_param(self, terminal):
        """
        Produces random terminal value, either input variable or float

        :return: terminal parameter
        """
        if terminal is 0:                                      # TODO hardcoded
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
        if terminal is 0:                                      # TODO hardcoded
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
        return node, param


class AGraphCpp(object):
    """
    Acyclic Graph representation of an equation
    """
    def __init__(self):
        self.command_list = []
        self.constants = []
        self.fitness = None

    def copy(self):
        """return a deep copy"""
        dup = AGraphCpp()
        dup.fitness = self.fitness
        dup.constants = list(self.constants)
        dup.command_list = list(self.command_list)
        return dup

    def needs_optimization(self):
        """find out whether constants need optimization"""
        util = self.utilized_commands()
        for i in range(len(self.command_list)):
            if util[i]:
                if self.command_list[i][0] == 1:   # TODO hard coded (next too)
                    if self.command_list[i][1][0] is -1 or \
                            self.command_list[i][1][0] >= len(self.constants):
                        return True
        return False

    def optimize_constants(self, fitness_metric, **kwargs):
        """optimize constants"""

        # compile fitness function for optimization
        util = self.utilized_commands()
        const_num = 0
        for i in range(len(self.command_list)):
            if util[i]:
                if self.command_list[i][0] == 1:              # TODO hard coded
                    self.command_list[i] = (1, (const_num,))
                    const_num += 1

        # define fitness function for optimization
        def const_opt_fitness(consts):
            """ fitness function for constant optimization"""
            self.constants = consts
            return fitness_metric.evaluate_vector(indv=self, **kwargs)

        # do optimization
        sol = optimize.root(const_opt_fitness,
                            np.random.uniform(-100, 100, const_num),
                            method='lm')

        # put optimal values in command list
        self.constants = sol.x

    def evaluate(self, fitness_metric, **kwargs):
        """evaluate the compiled stack"""
        if self.needs_optimization():
            self.optimize_constants(fitness_metric, **kwargs)
        try:
            # stack = bingocpp.CommandStack(self.command_list)
            f_of_x = bingocpp.simplify_and_evauluate(self.command_list,
                                                     kwargs['x'],
                                                     self.constants)
        except:
            print("***ERROR***")
            print(self)
            exit(-1)
        return f_of_x

    def evaluate_deriv(self, fitness_metric, **kwargs):
        """evaluate the compiled stack"""
        if self.needs_optimization():
            self.optimize_constants(fitness_metric, **kwargs)
        try:
            f_of_x, df_dx = bingocpp.simplify_and_evauluate_with_derivative(
                self.command_list, kwargs['x'], self.constants)
        except:
            print("***ERROR***")
            print(self)
            exit(-1)
        return f_of_x, df_dx

    def __str__(self):
        """overloaded string output"""
        util = self.utilized_commands()
        print_str = "---full stack---\n"
        i = 0
        for node, params in self.command_list:
            print_str += "(%d) <= " % i
            if node == 0:                                     # TODO hard coded
                print_str += COMMAND_PRINT_MAP[node] + "_%d\n" % params[0]
            elif node == 1:                                   # TODO hard coded
                print_str += COMMAND_PRINT_MAP[node]
                if params[0] == -1:                           # TODO hard coded
                    print_str += "\n"
                else:
                    print_str += " = " + str(self.constants[params[0]]) + "\n"
            else:
                print_str += "(%d)" % params[0] + " " + \
                             COMMAND_PRINT_MAP[node] \
                             + " " + "(%d)" % params[1] + "\n"
            i += 1
        print_str += "---small stack---\n"
        i = 0
        for node, params in self.command_list:
            if util[i]:
                print_str += "(%d) <= " % i
                if node == 0:                                 # TODO hard coded
                    print_str += COMMAND_PRINT_MAP[node] + "_%d\n" % params[0]
                elif node == 1:                               # TODO hard coded
                    print_str += COMMAND_PRINT_MAP[node]
                    if params[0] == -1:                       # TODO hard coded
                        print_str += "\n"
                    else:
                        print_str += " = " + str(self.constants[params[0]]) + \
                                     "\n"
                else:
                    print_str += "(%d)"%params[0] + " " + \
                                 COMMAND_PRINT_MAP[node] \
                                 + " " + "(%d)"%params[1] + "\n"
            i += 1
        return print_str

    def latexstring(self):                           # TODO need to update this
        """conversion to simplified latex string"""
        util = self.utilized_commands()
        str_list = []
        for i, (node, params) in enumerate(self.command_list):
                                                   # TODO hard coded whole func
            tmp_str = ""
            if util[i]:
                if node == 0:
                    tmp_str = "X_%d" % params[0]
                elif node == 1:
                    if params[0] == -1:
                        tmp_str = "0"
                    else:
                        tmp_str = str(self.constants[params[0]])
                elif node == 2:
                    tmp_str = "%s + %s" % (str_list[params[0]],
                                           str_list[params[1]])
                elif node == 3:
                    tmp_str = "%s - (%s)" % (str_list[params[0]],
                                             str_list[params[1]])
                elif node == 4:
                    tmp_str = "(%s)(%s)" % (str_list[params[0]],
                                            str_list[params[1]])
                elif node == 5:
                    tmp_str = "\\frac{%s}{%s}" % (str_list[params[0]],
                                                  str_list[params[1]])
            str_list.append(tmp_str)
        return str_list[-1]

    def utilized_commands(self):
        """find which commands are utilized"""
        util = [False]*len(self.command_list)
        util[-1] = True
        for i in range(1, len(self.command_list)):
            if util[-i] and self.command_list[-i][0] > 1:     # TODO hard coded
                for j in self.command_list[-i][1]:
                    util[j] = True
        return util

    def complexity(self):
        """find number of commands that are utilized"""
        return sum(self.utilized_commands())
