from .Base.ContinuousLocalOptimization import ChromosomeInterface
from .Util.ArgumentValidation import argument_validation
from .MultipleValues import MultipleValueChromosome, MultipleValueGenerator

class MultipleFloatChromosome(MultipleValueChromosome, ChromosomeInterface):

    def __init__(self, list_of_values, needs_opt_list=[]):
        super().__init__(list_of_values)
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
        super().__init__(random_value_function, values_per_chromosome)
        self._needs_opt_list = needs_opt_list

    def __call__(self):
        """Generation of a population of size 'population_size'
        of Multi-Value Chromosomes with lists that contain
        'values_per_list' values


        Returns
        -------
        out : a MultipleValueChromosome
        """
        random_list = self._generate_list(self._values_per_chromosome)
        return MultipleFloatChromosome(random_list, self._needs_opt_list)
