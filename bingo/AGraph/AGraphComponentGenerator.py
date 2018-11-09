class AGraphComponentGenerator:
    """
    Manipulates AGraph objects for generation, crossover, mutation,
    and distance
    """

    def __init__(self, input_x_dimension, num_initial_load_statements=1,
                 terminal_probability=0.1,
                 constant_probability=None):
        """
        Initialization of acyclic graph gene manipulator

        :param input_x_dimension: number of independent variables
        :param ag_size: length of command stack
        :param num_initial_load_statements: number of load operation which are required at the start
                       of stack
        :param terminal_probability: probability that a new node will be a terminal
        """
        self.input_x_dimension = input_x_dimension
        self.num_inital_load_statements = num_initial_load_statements
        self.terminal_probability = terminal_probability
        self.constant_probability = constant_probability

        self.node_type_list = []

        self.terminal_inds = []
        self.terminal_frequencies = []
        self.terminal_freq_sum = 0

        self.operator_inds = []
        self.operator_frequencies = []
        self.operator_freq_sum = 0
        self.num_node_types = 0

        self.add_node_type(0)
        for _ in range(input_x_dimension-1):
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