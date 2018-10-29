from .AGraphGene import AGraphGene
from ..GeneGenerator import GeneGenerator


class AGraphGenerator(GeneGenerator):
    """
    Manipulates AGraph objects for generation, crossover, mutation,
    and distance
    """

    def __init__(self, nvars, ag_size, nloads=1,
                 float_lim=10.0, terminal_prob=0.1):
        """
        Initialization of acyclic graph gene manipulator

        :param nvars: number of independent variables
        :param ag_size: length of command stack
        :param nloads: number of load operation which are required at the start
                       of stack
        :param float_lim: (0, max)  of floats which are generated
        :param terminal_prob: probability that a new node will be a terminal
        """
        self.input_x_dimension = nvars
        self.agraph_size = ag_size
        self.num_inital_load_statements = nloads
        self.max_random_float = float_lim
        self.terminal_probability = terminal_prob

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
        for stack_loc in range(self.agraph_size):
            if np.random.random() < self.terminal_probability \
                    or stack_loc < self.num_inital_load_statements:
                command_list.append(self.rand_terminal())
            else:
                command_list.append(self.rand_operator(stack_loc))
        indv.command_array = np.array(command_list, dtype=int)
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
            param = np.random.randint(self.input_x_dimension)
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
            param = np.random.randint(self.input_x_dimension)
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