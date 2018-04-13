"""
This module contains most of the code necessary for the representation of an
acyclic graph (linear stack) in symbolic regression
"""
import abc
import random
import logging
from scipy import optimize

import numpy as np

np.seterr(all='ignore')
LOGGER = logging.getLogger(__name__)


class AGraphManipulator(object):
    """
    Manipulates AGraph objects for generation, crossover, mutation,
    and distance
    """

    def __init__(self, nvars, ag_size,
                 nloads=1, float_lim=10.0, terminal_prob=0.1,
                 # constant_optimization=False
                ):
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

        self.namespace = {}

        self.add_node_type(AGNodes.Load_Data)
        for _ in range(nvars-1):
            self.terminal_inds.append(0)
        self.add_node_type(AGNodes.Load_Const)
        self.namespace['np'] = np

    def add_node_type(self, node_type):
        """
        Add a type of node to the set of allowed types

        :param node_type: acyclic graph node type which will be added to the
                          allowable set
        """
        if node_type not in self.node_type_list:
            self.node_type_list.append(node_type)
            if node_type.terminal:
                self.terminal_inds.append(self.num_node_types)
            else:
                self.operator_inds.append(self.num_node_types)
            self.num_node_types += 1
            if node_type.shorthand is not None:
                self.namespace[node_type.shorthand] = node_type.call
            if node_type.shorthand_deriv is not None:
                self.namespace[node_type.shorthand_deriv] = \
                    node_type.call_deriv

    def generate(self):
        """
        Generates random individual. Fills stack based on random
        nodes/terminals and random parameters

        :return: new random acyclic graph individual
        """
        indv = AGraph(self.namespace)
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
        child1.fit_set = False
        child2.fit_set = False
        child_age = max(parent1.genetic_age, parent2.genetic_age)
        child1.genetic_age = child_age
        child2.genetic_age = child_age
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
        orig_node_type, orig_params = indv.command_list[mut_point]


        # mutate operator (0.4) mutate params (0.4) prune branch (0.2)
        rand_val = np.random.random()

        # mutate operator
        if  rand_val < 0.4 and mut_point > self.nloads:
            new_type_found = False
            while not new_type_found:
                if np.random.random() < self.terminal_prob:
                    new_node_type, new_params = self.rand_terminal()
                else:
                    new_node_type, new_params = self.rand_operator(mut_point)
                new_type_found = new_node_type != orig_node_type or \
                                 orig_node_type.terminal
            # only use new params if needed
            if not new_node_type.terminal:  # don't worry about this for terms
                tmp = ()
                for i in range(new_node_type.arity):
                    if i < orig_node_type.arity:
                        tmp += (orig_params[i],)
                    else:
                        tmp += (new_params[i],)
                new_params = tmp
            indv.command_list[mut_point] = (new_node_type, new_params)

        # mutate parameters
        elif rand_val < 0.8:
            new_node_type = orig_node_type
            if orig_node_type.terminal:  # terminals
                new_params = self.mutate_terminal_param(new_node_type,
                                                        orig_params)
            else:  # operators
                new_params = self.rand_operator_params(new_node_type.arity,
                                                       mut_point)
            indv.command_list[mut_point] = (new_node_type, new_params)

        # prune branch
        else:
            if not orig_node_type.terminal:  # operators only
                pruned_param = random.choice(orig_params)
                for i in range(mut_point, len(indv.command_list)):
                    if mut_point in indv.command_list[i][1]:
                        mod_params = ()
                        for old_p in indv.command_list[i][1]:
                            if old_p == mut_point:
                                mod_params += (pruned_param,)
                            else:
                                mod_params += (old_p,)
                        indv.command_list[i] = (indv.command_list[i][0],
                                                mod_params)
        indv.compiled = False
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
        dist = 0
        for command1, command2 in zip(indv1.command_list, indv2.command_list):
            if command1[0] != command2[0]:
                dist += 0.5
            maxp = max(len(command1[1]), len(command2[1]))
            minp = min(len(command1[1]), len(command2[1]))
            for i in range(minp):
                if command1[1][i] != command2[1][i]:
                    dist += 0.5/maxp
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
        return command_list, indv.constants, indv.genetic_age

    def load(self, indv_list):
        """
        Loads the individual from a pickleable object

        :param indv_list: individual in pickleable form
        :return: individual in normal form
        """
        indv = AGraph(self.namespace)
        indv.constants = indv_list[1]
        indv.genetic_age = indv_list[2]
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
        params = self.rand_operator_params(node_type.arity, stack_loc)
        return node_type, params

    def rand_terminal_param(self, terminal):
        """
        Produces random terminal value, either input variable or float

        :return: terminal parameter
        """
        if terminal is AGNodes.Load_Data:
            param = np.random.randint(self.nvars)
        else:
            param = None
        return param,

    def mutate_terminal_param(self, terminal, old_params):
        """
        Produces random terminal value, either input variable or float
        Mutates floats by getting random variation of old param

        :return: terminal parameter
        """
        if terminal is AGNodes.Load_Data:
            param = np.random.randint(self.nvars)
        else:
            param = None
        return param,

    def rand_terminal(self):
        """
        Produces random terminal node and value

        :return: Load node, data index or constant index
        """
        node = self.node_type_list[random.choice(self.terminal_inds)]
        param = self.rand_terminal_param(node)
        return node, param


class AGraph(object):
    """
    Acyclic Graph representation of an equation
    """
    def __init__(self, namespace=None):
        self.command_list = []
        self.constants = []
        self.compiled = False
        self.compiled_deriv = False
        self.fitness = None
        self.fit_set = False
        self.genetic_age = 0
        if namespace is not None:
            self.namespace = namespace.copy()
        else:
            self.namespace = {}

    def copy(self):
        """return a deep copy"""
        dup = AGraph(self.namespace)
        dup.compiled = self.compiled
        dup.fitness = self.fitness
        dup.fit_set = self.fit_set
        dup.constants = list(self.constants)
        dup.command_list = list(self.command_list)
        dup.genetic_age = self.genetic_age
        return dup

    def compile(self):
        """compile the stack of commands"""
        util = self.utilized_commands()
        code_str = ("def evaluate(x, consts):\n"
                    "    stack = [None]*%d\n" % len(self.command_list))
        for i, (node, params) in enumerate(self.command_list):
            if util[i]:
                code_str += ("    stack[%d] = " % i +
                             node.funcstring(params) + "\n")
        code_str += "    return stack[-1].reshape([-1,1])\n"
        exec(compile(code_str, '<string>', 'exec'), self.namespace)
        self.compiled = True

    def compile_deriv(self):
        """compile the stack of commands and derivatives"""
        util = self.utilized_commands()
        code_str = "def evaluate_deriv(x, consts):\n"
        code_str += "    stack = [None]*%d\n" % len(self.command_list)
        code_str += "    deriv = [None]*%d\n" % len(self.command_list)
        for i, (node, params) in enumerate(self.command_list):
            if util[i]:
                code_str += ("    stack[%d] = " % i +
                             node.funcstring(params) + "\n")
                code_str += ("    deriv[%d] = " % i +
                             node.derivstring(params) + "\n")
        code_str += "    return stack[-1].reshape([-1,1]), deriv[-1]\n"

        exec(compile(code_str, '<string>', 'exec'), self.namespace)
        self.compiled_deriv = True

    def needs_optimization(self):
        """find out whether constants need optimization"""
        util = self.utilized_commands()
        for i in range(len(self.command_list)):
            if util[i]:
                if self.command_list[i][0] == AGNodes.Load_Const:
                    if self.command_list[i][1][0] is None:
                        return True
                    elif self.command_list[i][1][0] >= len(self.constants):
                        return True
        return False

    def count_constants(self):
        """count constants and set up for optimization"""

        # compile fitness function for optimization
        util = self.utilized_commands()
        const_num = 0
        for i, (node, _) in enumerate(self.command_list):
            if util[i]:
                if node is AGNodes.Load_Const:
                    self.command_list[i] = (node, (const_num,))
                    const_num += 1
        return const_num

    def set_constants(self, consts):
        """set individual's constants"""
        self.constants = consts

    def evaluate(self, x):
        """evaluate the compiled stack"""
        if not self.compiled:
            self.compile()
        try:
            f_of_x = self.namespace['evaluate'](x, self.constants)
        except:
            LOGGER.error("Error in stack evaluation")
            LOGGER.error(str(self))
            exit(-1)
        return f_of_x

    def evaluate_deriv(self, x):
        """evaluate the compiled stack"""
        if not self.compiled_deriv:
            self.compile_deriv()
        try:
            f_of_x, df_dx = self.namespace['evaluate_deriv'](x, self.constants)
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
        for node, params in self.command_list:
            print_str += "(%d) <= " % i
            print_str += node.printstring(params) + "\n"
            i += 1
        print_str += "---small stack---\n"
        i = 0
        for node, params in self.command_list:
            if util[i]:
                print_str += "(%d) <= " % i
                print_str += node.printstring(params) + "\n"
            i += 1
        return print_str

    def latexstring(self):
        """conversion to simplified latex string"""
        util = self.utilized_commands()
        str_list = [None]*len(self.command_list)
        for i, (node, params) in enumerate(self.command_list):
            if util[i]:
                str_list[i] = node.latexstring(params, str_list)
        indv_str = str_list[-1]
        if self.constants is not None:
            for i, c in enumerate(self.constants):
                indv_str = indv_str.replace("c_" + str(i), "{:.4f}".format(c))
        return indv_str

    def utilized_commands(self):
        """find which commands are utilized"""
        util = [False]*len(self.command_list)
        util[-1] = True
        for i in range(1, len(self.command_list)):
            if util[-i] and not self.command_list[-i][0].terminal:
                for j in self.command_list[-i][1]:
                    util[j] = True
        return util

    def complexity(self):
        """find number of commands that are utilized"""
        return sum(self.utilized_commands())


class AGNodes(object):
    """class that contains node types used in acyclic graphs"""

    class Node(object, metaclass=abc.ABCMeta):
        """node superclass"""

        terminal = False
        arity = 0
        shorthand = None
        shorthand_deriv = None

        @staticmethod
        @abc.abstractmethod
        def funcstring(params):
            """creates a string for parsing"""
            pass

        @staticmethod
        @abc.abstractmethod
        def derivstring(params):
            """creates a string for parsing derivatives"""
            pass

        @staticmethod
        @abc.abstractmethod
        def printstring(params):
            """creates a string for printing to terminal"""
            pass

        @staticmethod
        @abc.abstractmethod
        def latexstring(params, str_list):
            """creates a string for outputting latex"""
            pass

    class Load_Data(Node):
        """load"""
        terminal = True
        shorthand_deriv = "deriv_x"

        @staticmethod
        def call_deriv(index, xshape):
            """gets derivative array of loaded value"""
            tmp = np.zeros(xshape)
            tmp[:, index] = 1
            return tmp

        @staticmethod
        def funcstring(params):
            return "x[:, %d]" % params[0]

        @staticmethod
        def derivstring(params):
            return "deriv_x(%d, x.shape)" % params

        @staticmethod
        def printstring(params):
            return "x[:, %d]" % params[0]

        @staticmethod
        def latexstring(params, str_list):
            return "x_%d" % params

    class Load_Const(Node):
        """load constant for optimization"""
        terminal = True
        shorthand_deriv = "deriv_c"

        @staticmethod
        def call_deriv(xshape):
            """gets derivative array of loaded value"""
            return np.zeros(xshape)

        @staticmethod
        def funcstring(params):
            return "consts[%d]" % params[0]

        @staticmethod
        def derivstring(params):
            return "deriv_c(x.shape)"

        @staticmethod
        def printstring(params):
            if params[0] is None:
                return "consts[None]"
            else:
                return "consts[%d]" % params[0]

        @staticmethod
        def latexstring(params, str_list):
            return "c_%d" % params[0]

    class Add(Node):
        """
        addition
        """
        arity = 2
        shorthand = "add"
        call = np.add

        @staticmethod
        def printstring(params):
            return "(%d) + (%d)" % params

        @staticmethod
        def latexstring(params, str_list):
            return "%s + %s" % (str_list[params[0]], str_list[params[1]])

        @staticmethod
        def funcstring(params):
            return "add(stack[%d], stack[%d])" % params

        @staticmethod
        def derivstring(params):
            return "add(deriv[%d], deriv[%d])" % params

    class Subtract(Node):
        """
        subtraction
        """
        arity = 2
        shorthand = "subtract"
        call = np.subtract

        @staticmethod
        def funcstring(params):
            return "subtract(stack[%d], stack[%d])" % params

        @staticmethod
        def derivstring(params):
            return "subtract(deriv[%d], deriv[%d])" % params

        @staticmethod
        def printstring(params):
            return "(%d) - (%d)" % params

        @staticmethod
        def latexstring(params, str_list):
            return "%s - (%s)" % (str_list[params[0]], str_list[params[1]])

    class Multiply(Node):
        """
        multiplication
        derivative calculation needs: add
        """
        arity = 2
        shorthand = "multiply"
        call = np.multiply

        @staticmethod
        def printstring(params):
            return "(%d) * (%d)" % params

        @staticmethod
        def latexstring(params, str_list):
            return "(%s)(%s)" % (str_list[params[0]], str_list[params[1]])

        @staticmethod
        def funcstring(params):
            return "multiply(stack[%d], stack[%d])" % params

        @staticmethod
        def derivstring(params):
            return "add("\
                   "multiply(deriv[%d].transpose(), stack[%d]).transpose(),"\
                   "multiply(deriv[%d].transpose(), stack[%d]).transpose())" %\
                   (params[1], params[0], params[0], params[1])

    class Divide(Node):
        """
        division
        derivative calculation needs: multiply, subtract
        """
        arity = 2
        shorthand = "divide"
        call = np.divide

        @staticmethod
        def printstring(params):
            return "(%d) / (%d)" % params

        @staticmethod
        def latexstring(params, str_list):
            return "\\frac{%s}{%s}" % (str_list[params[0]], str_list[params[1]])

        @staticmethod
        def funcstring(params):
            return "divide(stack[%d], stack[%d])" % params

        @staticmethod
        def derivstring(params):
            return "divide(subtract("\
                   "multiply(deriv[%d].transpose(), stack[%d]), "\
                   "multiply(deriv[%d].transpose(), stack[%d])),"\
                   "multiply(stack[%d], stack[%d])).transpose()" % \
                   (params[0], params[1], params[1], params[0],
                    params[1], params[1])

    class Sin(Node):
        """
        sine
        derivative calculation needs: multiply
        """
        arity = 1
        shorthand = "sin"
        call = np.sin
        shorthand_deriv = "sin_deriv"
        call_deriv = np.cos

        @staticmethod
        def printstring(params):
            return "sin (%d)" % params

        @staticmethod
        def latexstring(params, str_list):
            return "\\sin(%s)" % (str_list[params[0]])

        @staticmethod
        def funcstring(params):
            return "sin(stack[%d])" % params

        @staticmethod
        def derivstring(params):
            return "multiply(deriv[%d].transpose(), "\
                   "sin_deriv(stack[%d])).transpose()" % (params[0], params[0])

    class Cos(Node):
        """
        cosine
        derivative calculation needs: multiply
        """
        arity = 1
        shorthand = "cos"
        call = np.cos
        shorthand_deriv = "cos_deriv"
        call_deriv = np.sin

        @staticmethod
        def printstring(params):
            return "cos (%d)" % params

        @staticmethod
        def latexstring(params, str_list):
            return "\\cos(%s)" % (str_list[params[0]])

        @staticmethod
        def funcstring(params):
            return "cos(stack[%d])" % params

        @staticmethod
        def derivstring(params):
            return "multiply(-deriv[%d].transpose(), "\
                   "cos_deriv(stack[%d])).transpose()" %\
                   (params[0], params[0])

    class Exp(Node):
        """
        e^x
        derivative calculation needs: multiply
        """
        arity = 1
        shorthand = "exp"
        call = np.exp

        @staticmethod
        def printstring(params):
            return "exp (%d)" % params

        @staticmethod
        def latexstring(params, str_list):
            return "\\exp(%s)" % (str_list[params[0]])

        @staticmethod
        def funcstring(params):
            return "exp(stack[%d])" % params

        @staticmethod
        def derivstring(params):
            return "multiply(deriv[%d].transpose(), "\
                   "exp(stack[%d])).transpose()" %\
                   (params[0], params[0])

    class Log(Node):
        """
        log|x|
        derivative calculation needs: divide
        """
        arity = 1
        shorthand = "log"
        call = np.log

        @staticmethod
        def printstring(params):
            return "log|| (%d)" % params

        @staticmethod
        def latexstring(params, str_list):
            return "\\log(|%s|)" % (str_list[params[0]])

        @staticmethod
        def funcstring(params):
            return "log(stack[%d])" % params

        @staticmethod
        def derivstring(params):
            return "divide(deriv[%d].transpose(), stack[%d]).transpose()" % \
                   (params[0], params[0])

    class Abs(Node):
        """
        |x|
        derivative calculation needs: multiply
        """
        arity = 1
        shorthand = "absl"
        shorthand_deriv = "sign"
        call_deriv = np.sign
        call = np.abs

        @staticmethod
        def printstring(params):
            return "abs (%d)" % params

        @staticmethod
        def latexstring(params, str_list):
            return "|%s|" % (str_list[params[0]])

        @staticmethod
        def funcstring(params):
            return "absl(stack[%d])" % params

        @staticmethod
        def derivstring(params):
            return "multiply(deriv[%d].transpose(), "\
                   "sign(stack[%d])).transpose()" %\
                   (params[0], params[0])

    class Sqrt(Node):
        """
        (x)^0.5
        derivative calculation needs: multiply, divide
        """
        arity = 1
        shorthand = "sqroot"
        call = np.sqrt

        @staticmethod
        def printstring(params):
            return "sqrt (%d)" % params

        @staticmethod
        def latexstring(params, str_list):
            return "\\sqrt{%s}" % (str_list[params[0]])

        @staticmethod
        def funcstring(params):
            return "sqroot(stack[%d])" % params

        @staticmethod
        def derivstring(params):
            return "multiply(deriv[%d].transpose(), "\
                   "divide(0.5, sqroot(stack[%d]))).transpose()" %\
                   (params[0], params[0])

    class Pow(Node):
        """
        (x)^y
        derivative calculation needs: add, multiply, divide, log
        """
        arity = 2
        shorthand = "power"
        call = np.power

        @staticmethod
        def printstring(params):
            return "power (%d) (%d)" % params

        @staticmethod
        def latexstring(params, str_list):
            return "(%s)^{(%s)}" % (str_list[params[0]], str_list[params[1]])

        @staticmethod
        def funcstring(params):
            return "power(stack[%d], stack[%d])" % params

        @staticmethod
        def derivstring(params):
            return "multiply(add( multiply(deriv[%d].transpose(), "\
                   "divide(stack[%d],stack[%d])), "\
                   "multiply(deriv[%d].transpose(), log(stack[%d])) ), "\
                   "power(stack[%d], stack[%d])).transpose()" %\
                    (params[0], params[1], params[0], params[0], params[0],
                     params[0], params[1])
