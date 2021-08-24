import numpy as np
from scipy import stats

from smcpy.mcmc.vector_mcmc import VectorMCMC
from smcpy.mcmc.vector_mcmc_kernel import VectorMCMCKernel
from smcpy import SMCSampler

from bingo.evaluation.fitness_function import FitnessFunction


class MarginalLikelihoodFunction(FitnessFunction):

    def __init__(self, continuous_local_opt,
                 num_particles=150, smc_steps=15, mcmc_steps=12,
                 ess_threshold=0.75, phi_sequence_exponent=0.001,
                 noise_std=5.0, prior_variance=5.0):
        # TODO: What are the default hyper params?

        self._continuous_local_opt = continuous_local_opt
        self._eval_count = 0

        self._num_particles = num_particles
        self._smc_steps = smc_steps
        self._mcmc_steps = mcmc_steps
        self._ess_threshold = ess_threshold
        self._noise_std = noise_std
        self._prior_variance = prior_variance

        self._phi_sequence = \
            self._calculate_phi_sequence(smc_steps, phi_sequence_exponent)

    @staticmethod
    def _calculate_phi_sequence(steps, exponent):
        x = np.linspace(0, 1, steps)
        a = 1/(np.exp(exponent) - 1)
        return a*(np.exp(exponent*x) - 1)

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
        # TODO: how does this work when num model params = 0?
        num_model_params = individual.get_number_local_optimization_params()
        param_names = ['p{}'.format(i) for i in range(num_model_params)]

        # TODO: What should we use for Priors?
        # next two lines used to force revaluation of optimal params
        individual._notify_modification()
        individual._needs_opt = True
        _ = self._continuous_local_opt(individual)
        optimal_params = individual.constants
        priors = [stats.norm(mean, self._prior_variance)
                  for mean in optimal_params]

        # TODO: estimate or assume noise_std?
        vector_mcmc = VectorMCMC(lambda x: self._evaluate_model(x, individual),
                                 self.training_data.y.flatten(),
                                 priors,
                                 self._noise_std,
                                 )
        mcmc_kernel = VectorMCMCKernel(vector_mcmc, param_order=param_names)
        smc = SMCSampler(mcmc_kernel)

        _, evidence = smc.sample(self._num_particles, self._mcmc_steps,
                                 self._phi_sequence, self._ess_threshold)

        return evidence

    def _evaluate_model(self, theta, individual):
        self._eval_count += 1
        theta = theta.T
        individual.set_local_optimization_params(theta)
        return individual.evaluate_equation_at(self.training_data.x).T

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
