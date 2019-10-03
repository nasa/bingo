# from SMCpy import ?
from ...evaluation.fitness_function import FitnessFunction


class ModelWrapper: #(SMCModel)
    def __init__(self, equation, fitness_vector_function):
        self._equation = equation
        self._fitness_vector_function = fitness_vector_function

    def evaluate(self, params):
        self._equation.set_local_optimization_params(params)
        return self._fitness_vector_function(self._equation)


class ModelEvidenceFunction(FitnessFunction):

    def __init__(self, fitness_vector_function):
        self._fitness_vector_function = fitness_vector_function

    def __call__(self, individual):
        """ Bulk of SMC stuff goes here

        Parameters
        ----------
        individual: equation/agraph
            individual equation describing the model for which the evidence
            will be calculated

        Returns
        -------
        float :
            the model evidence
        """
        num_params = individual.get_number_local_optimization_params()
        if num_params > 0:
            model = ModelWrapper(individual, self._fitness_vector_function)
            # Do some SMC!
        else:
            # no clue what to do if we have no parameters
            pass

    @property
    def training_data(self):
        """TrainingData : data that can be used in fitness evaluations"""
        return self._fitness_vector_function.training_data

    @training_data.setter
    def training_data(self, value):
        self._fitness_vector_function.training_data = value

    @property
    def eval_count(self):
        """int : the number of evaluations that have been performed"""
        return self._fitness_vector_function.eval_count

    @eval_count.setter
    def eval_count(self, value):
        self._fitness_vector_function.eval_count = value
