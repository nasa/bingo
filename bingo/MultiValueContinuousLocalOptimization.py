from .Base.ContinuousLocalOptimization import ChromosomeInterface
from .MultipleValues import MultipleValueChromosome, MultipleValueGenerator

class MultiValueContinuousLocalOptimization(MultipleValueChromosome, ChromosomeInterface):

    def __init__(self, list_of_values, needs_opt_list):
        super().__init__(list_of_values)
        self.needs_opt_list = needs_opt_list

    def needs_local_optimization(self):
        """Does the individual need local optimization

        Returns
        -------
        bool
            Individual needs optimization
        """
        return True


    def get_number_local_optimization_params(self):
        """Get number of parameters in local optimization

        Returns
        -------
        int
            number of paramneters to be optimized
        """
        return len(self.needs_opt_list)

    def set_local_optimization_params(self, params):
        """Set local optimization parameters

        Parameters
        ----------
        params : list-like of numeric
                 Values to set the parameters
        """
        for i, index in enumerate(self.needs_opt_list):
            self.list_of_values[index] = params[i]

class MultiValueContinuousLocalOptimizationGenerator(MultipleValueGenerator):

    def __call__(self):
        """Generation of a population of size 'population_size'
        of Multi-Value Chromosomes with lists that contain
        'values_per_list' values


        Returns
        -------
        out : a MultipleValueChromosome
        """
        random_list = self._generate_list(self._values_per_chromosome)
        needs_op_list = [1, 4, 5]
        return MultiValueContinuousLocalOptimization(random_list, needs_op_list)
