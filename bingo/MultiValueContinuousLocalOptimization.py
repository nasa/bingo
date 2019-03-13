from bingo.Base.ContinuousLocalOptimization import ChromosomeInterface
from bingo.MultipleValues import MultipleValueChromosome

class MultiValueContinuousLocalOptimization(MultipleValueChromosome, ChromosomeInterface):

    

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

    def set_local_optimization_params(self, params):
        """Set local optimization parameters

        Parameters
        ----------
        params : list-like of numeric
                 Values to set the parameters
        """