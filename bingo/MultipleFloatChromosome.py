from .Base.ContinuousLocalOptimization import ChromosomeInterface
from .Util.ArgumentValidation import argument_validation
from .MultipleValues import MultipleValueChromosome, MultipleValueGenerator

class MultipleFloatChromosome(MultipleValueChromosome, ChromosomeInterface):

    def __init__(self, list_of_values, needs_opt_list=[]):
        self._check_list_contains_floats(list_of_values)
        super().__init__(list_of_values)
        self._check_list_contains_ints_in_valid_range(needs_opt_list)
        self._needs_opt_list = needs_opt_list

    def needs_local_optimization(self):
        """Does the individual need local optimization

        Returns
        -------
        bool
            Individual needs optimization
        """
        if not self._needs_opt_list:
            return False
        return True

    def get_number_local_optimization_params(self):
        """Get number of parameters in local optimization

        Returns
        -------
        int
            number of paramneters to be optimized
        """
        return len(self._needs_opt_list)

    def set_local_optimization_params(self, params):
        """Set local optimization parameters

        Parameters
        ----------
        params : list-like of numeric
                 Values to set the parameters
        """
        for i, index in enumerate(self._needs_opt_list):
            self.list_of_values[index] = params[i]

    def _check_list_contains_floats(self, list_of_values):
        if not all(isinstance(x, float) for x in list_of_values):
            raise ValueError("The value list must contain only floats.")

    def _check_list_contains_ints_in_valid_range(self, list_of_indices):
        if not list_of_indices:
            return
        if not all(isinstance(x, int) for x in list_of_indices):
            raise ValueError("The list of optimization indices must be \
                              unsigned integers.")
        if min(list_of_indices) < 0 or \
               max(list_of_indices) > len(self.list_of_values):
            raise ValueError("The list of optimization indices must be within \
                              the length of the list of values.")

class MultipleFloatChromosomeGenerator(MultipleValueGenerator):
    """Generation of a population of Multi-Value Chromosomes

    Parameters
    ----------
    random_value_function : user defined function
        A function that returns a list of randomly generated values.
        This list is then passed to the ``MultipleValueChromosome``
        constructor.
    values_per_chromosome : int
        The number of values that each chromosome will hold
    """
    @argument_validation(values_per_chromosome={">=": 0})
    def __init__(self, random_value_function, values_per_chromosome,
                 needs_opt_list=[]):
        self._check_function_produces_float(random_value_function)
        super().__init__(random_value_function, values_per_chromosome)
        self._needs_opt_list = needs_opt_list

    def __call__(self):
        """Generation of a population of size 'population_size'
        of Multi-Value Chromosomes with lists that contain
        'values_per_list' values

        Returns
        -------
        out : MultipleValueChromosome
        """
        random_list = self._generate_list(self._values_per_chromosome)
        return MultipleFloatChromosome(random_list, self._needs_opt_list)

    def _check_function_produces_float(self, random_value_function):
        val = random_value_function()
        if not isinstance(val, float):
            raise ValueError("Random Value Function must generate float values.")
