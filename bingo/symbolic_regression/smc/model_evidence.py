# from SMCpy import ?
from ...evaluation.fitness_function import FitnessFunction


class ModelWrapper: #(SMCModel)
    def __init__(self, equation, x):
        self._equation = equation
        self._x = x
        self.eval_count = 0

    def evaluate(self, params):
        self.eval_count += 1
        self._equation.set_local_optimization_params(params)
        return self._equation.evaluate_equation_at(self._x)


class ModelEvidenceFunction(FitnessFunction):

    def __init__(self, continuous_local_opt):
        self._continuous_local_opt = continuous_local_opt
        self._eval_count = 0
        # hyper parameters initialized here

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
        _ = self._continuous_local_opt(individual)
        optimal_params = individual.get_local_optimization_params()
        model = ModelWrapper(individual, self.training_data.x)
        observed_data = self.training_data.y
        # Do some SMC!


        self._eval_count += model.eval_count
        return  # evidence

    @property
    def training_data(self):
        """TrainingData : data that can be used in fitness evaluations"""
        return self._continuous_local_opt.training_data

    @training_data.setter
    def training_data(self, value):
        self._continuous_local_opt.training_data = value

    @property
    def eval_count(self):
        """int : the number of evaluations that have been performed"""
        return self._eval_count + self._continuous_local_opt.eval_count

    @eval_count.setter
    def eval_count(self, value):
        self._eval_count = value - self._continuous_local_opt.eval_count
