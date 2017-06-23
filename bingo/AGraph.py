"""
This module contains most of the code necessary for the representation of an
acyclic graph (linear stack) in symbolic regression
"""
import abc

import numpy as np
np.seterr(all='raise')


class AGraphManipulator(object):
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
        :param float_lim: max (-min) of floats which are generated
        :param terminal_prob: probability that a new node will be a terminal
        """
        self.nvars = nvars
        self.ag_size = ag_size
        self.nloads = nloads
        self.float_lim = float_lim
        self.terminal_prob = terminal_prob

        self.node_type_list = []
        self.num_node_types = 0

        self.namespace = {}
        self.namespace[AGNodes.Load.shorthand_deriv] = \
            AGNodes.Load.call_deriv


    def add_node_type(self, node_type):
        """
        Add a type of node to the set of allowed types

        :param node_type: acyclic graph node type which will be added to the
                          allowable set
        """
        if node_type not in self.node_type_list:
            self.node_type_list.append(node_type)
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

        # randomly change operation or parameter with equal prob
        if np.random.random() < 0.5 and mut_point > self.nloads: # op change
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

        else:  # parameter change
            new_node_type = orig_node_type
            if orig_node_type.terminal:  # terminals
                new_params = self.rand_terminal_param()
            else: # operators
                new_params = self.rand_operator_params(new_node_type.arity,
                                                       mut_point)

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
        indv_list = []
        for node, params in indv.command_list:
            if node in self.node_type_list: #node
                ind = self.node_type_list.index(node)
            else: # terminal
                ind = -1
            indv_list.append((ind, params))
        return indv_list

    def load(self, indv_list):
        """
        Loads the individual from a pickleable object

        :param indv_list: individual in pickleable form
        :return: individual in normal form
        """
        indv = AGraph(self.namespace)
        for node_num, params in indv_list:
            if node_num in range(len(self.node_type_list)):  # node
                indv.command_list.append((self.node_type_list[node_num],
                                          params))
            elif node_num == -1:  # terminal
                indv.command_list.append((AGNodes.Load, params))
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
        return self.node_type_list[np.random.randint(self.num_node_types)]

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

    def rand_terminal_param(self):
        """
        Produces random terminal value, either input variable or float

        :return: terminal parameter
        """
        i = np.random.randint(self.nvars+1)
        if i < self.nvars:
            return "x[%d]" % i,
        else:
            return str(np.random.random()*self.float_lim),

    def rand_terminal(self):
        """
        Produces random terminal node and value

        :return: Load node, value/input to load
        """
        param = self.rand_terminal_param()
        return AGNodes.Load, param


class AGraph(object):
    """
    Acyclic Graph representation of an equation
    """
    def __init__(self, namespace=None):
        self.command_list = []
        self.compiled = False
        self.compiled_deriv = False
        self.fitness = None
        if namespace is not None:
            self.namespace = namespace.copy()
        else:
            self.namespace = {}

    def copy(self):
        """return a deep copy"""
        dup = AGraph(self.namespace)
        dup.compiled = self.compiled
        dup.fitness = self.fitness
        dup.command_list = list(self.command_list)
        return dup

    def compile(self):
        """compile the stack of commands"""
        util = self.utilized_commands()
        code_str = ("def evaluate(x):\n"
                    "    stack = [None]*%d\n" % len(self.command_list))
        for i, (node, params) in enumerate(self.command_list):
            if util[i]:
                code_str += ("    stack[%d] = " % i +
                             node.funcstring(params) + "\n")

        code_str += "    return stack[-1]\n"
        exec(compile(code_str, '<string>', 'exec'), self.namespace)
        self.compiled = True

    def compile_deriv(self):
        """compile the stack of commands and derivatives"""
        util = self.utilized_commands()
        code_str = "def evaluate_deriv(x):\n"
        code_str += "    n_d = len(x)\n"
        code_str += "    stack = [None]*%d\n" % len(self.command_list)
        code_str += "    deriv = [None]*%d\n" % len(self.command_list)
        for i, (node, params) in enumerate(self.command_list):
            if util[i]:
                code_str += ("    stack[%d] = " % i +
                             node.funcstring(params) + "\n")
                code_str += ("    deriv[%d] = " % i +
                             node.derivstring(params) + "\n")

        code_str += "    return deriv[-1]\n"
        exec(compile(code_str, '<string>', 'exec'), self.namespace)
        self.compiled_deriv = True

    def evaluate(self, params):
        """evaluate the compiled stack"""
        if not self.compiled:
            self.compile()
        return self.namespace['evaluate'](params)

    def evaluate_deriv(self, params):
        """evaluate the compiled stack"""
        if not self.compiled_deriv:
            self.compile_deriv()
        return self.namespace['evaluate_deriv'](params)

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
        str_list = []
        for node, params in self.command_list:
            str_list.append(node.latexstring(params, str_list))
        return str_list[-1].replace('[', '_').replace(']', '')

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

    class Load(Node):
        """load"""
        terminal = True
        shorthand_deriv = "deriv_x"

        @staticmethod
        def call_deriv(index, length):
            """gets derivative array of loaded value"""
            tmp = np.zeros(length)
            if index is not -1:
                tmp[index] = 1
            return tmp

        @staticmethod
        def funcstring(params):
            return params[0]

        @staticmethod
        def derivstring(params):
            if "x" in params[0]:
                index = int(params[0][2:-1])
            else:
                index = -1
            dstring = "deriv_x(%d, n_d)" % index
            return dstring

        @staticmethod
        def printstring(params):
            return params[0]

        @staticmethod
        def latexstring(params, str_list):
            return params[0]

    class Add(Node):
        """addition"""
        arity = 2

        @staticmethod
        def printstring(params):
            return "(%d) + (%d)" % params

        @staticmethod
        def latexstring(params, str_list):
            return "%s + %s" % (str_list[params[0]], str_list[params[1]])

        @staticmethod
        def funcstring(params):
            return "stack[%d] + stack[%d]" % params

        @staticmethod
        def derivstring(params):
            return "deriv[%d] + deriv[%d]" % params

    class Subtract(Node):
        """subtraction"""
        arity = 2

        @staticmethod
        def funcstring(params):
            return "stack[%d] - stack[%d]" % params

        @staticmethod
        def derivstring(params):
            return "deriv[%d] - deriv[%d]" % params

        @staticmethod
        def printstring(params):
            return "(%d) - (%d)" % params

        @staticmethod
        def latexstring(params, str_list):
            return "%s - (%s)" % (str_list[params[0]], str_list[params[1]])

    class Multiply(Node):
        """multiplication"""
        arity = 2

        @staticmethod
        def printstring(params):
            return "(%d) * (%d)" % params

        @staticmethod
        def latexstring(params, str_list):
            return "(%s)(%s)" % (str_list[params[0]], str_list[params[1]])

        @staticmethod
        def funcstring(params):
            return "stack[%d] * stack[%d]" % params

        @staticmethod
        def derivstring(params):
            return "stack[%d] * deriv[%d] + deriv[%d] * stack[%d]" % \
                   (params[0], params[1], params[0], params[1])

    class Divide(Node):
        """division"""
        arity = 2
        shorthand = "div"

        @staticmethod
        def call(num, den):
            """version of divide safe against zero division"""
            if abs(den) > 1e-16:
                return num / den
            else:
                return np.nan

        @staticmethod
        def printstring(params):
            return "(%d) / (%d)" % params

        @staticmethod
        def latexstring(params, str_list):
            return "\\frac{%s}{%s}" % (str_list[params[0]], str_list[params[1]])

        @staticmethod
        def funcstring(params):
            return "div(stack[%d], stack[%d])" % params

        @staticmethod
        def derivstring(params):
            return "(stack[%d] * deriv[%d] - stack[%d] * deriv[%d]) * " \
                   "div(1.0, stack[%d]*stack[%d])" % \
                   (params[1], params[0], params[0], params[1],
                    params[1], params[1])

    class Sin(Node):
        """sine"""
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
            return "deriv[%d] * sin_deriv(stack[%d])" % (params[0], params[0])

    class Cos(Node):
        """cosine"""
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
            return "-deriv[%d] * cos_deriv(stack[%d])" %\
                   (params[0], params[0])

    class Exp(Node):
        """e^x"""
        arity = 1
        shorthand = "exp"

        @staticmethod
        def call(x):
            """safe exponential function"""
            try:
                ans = np.exp(x)
            except (ArithmeticError, OverflowError, FloatingPointError,
                    ValueError):
                ans = np.nan
            return ans

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
            return "deriv[%d] * exp(stack[%d])" % (params[0], params[0])

    class Log(Node):
        """log|x|"""
        arity = 1
        shorthand = "log"

        @staticmethod
        def call(x):
            """safe log function"""
            try:
                ans = np.log(abs(x))
            except (ArithmeticError, OverflowError, FloatingPointError,
                    ValueError):
                ans = np.nan
            return ans

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
            return "deriv[%d] * div(1.0, stack[%d])" % \
                   (params[0], params[0])

    class Abs(Node):
        """|x|"""
        arity = 1
        shorthand = "absl"
        shorthand_deriv = "sign"
        call_deriv = np.sign

        @staticmethod
        def call(x):
            """absolute function which handles nans"""
            try:
                ans = np.abs(x)
            except (ArithmeticError, OverflowError, FloatingPointError,
                    ValueError):
                ans = np.nan
            return ans

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
            return "deriv[%d] * sign(stack[%d])" % (params[0], params[0])

    class Sqrt(Node):
        """(x)^0.5"""
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
            return "deriv[%d] * div(0.5, sqroot(stack[%d]))" % (params[0],
                                                                params[0])
