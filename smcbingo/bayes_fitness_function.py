import numpy as np
import bingo.util.global_imports as gi
import smcpy.utils.global_imports as smc_gi

from scipy.stats import multivariate_normal as mvn
from scipy.stats import invgamma

from bingo.evaluation.fitness_function import FitnessFunction
from bingo.symbolic_regression import ExplicitTrainingData
from smcpy.mcmc.vector_mcmc import VectorMCMC
from smcpy.mcmc.vector_mcmc_kernel import VectorMCMCKernel
from smcpy import SMCSampler
from smcpy import ImproperUniform


class BayesFitnessFunction(FitnessFunction):

    def __init__(self, continuous_local_opt, num_particles=150, phi_exponent=8,
                 smc_steps=15, mcmc_steps=12, ess_threshold=0.75, std=None,
                 return_nmll_only=True):

        if smc_steps <= 2:
            raise ValueError('smc_steps must be > 2')
        if phi_exponent <= 0:
            raise ValueError('phi_exponent must be > 0')

        self._num_particles = num_particles
        self._smc_steps = smc_steps
        self._mcmc_steps = mcmc_steps
        self._ess_threshold = ess_threshold
        self._std = std
        self._return_nmll_only = return_nmll_only

        self._num_observations = len(continuous_local_opt.training_data.x)
        self._cont_local_opt = continuous_local_opt
        self._fbf_phi_idx, self._phi_seq = self._calc_phi_sequence(phi_exponent)
        self._eval_count = 0

    def __call__(self, individual):
        gi.set_use_gpu(False)
        param_names = self.get_parameter_names(individual)
        individual = self.do_local_opt(individual)
        try:
            proposal = self.generate_proposal_samples(individual,
                                                      self._num_particles)
        except (ValueError, np.linalg.LinAlgError):
            return np.nan

        priors = [ImproperUniform() for _ in range(len(param_names))]
        if self._std is None:
            priors.append(ImproperUniform(0, None))
            param_names.append('std_dev')

        training_data_y = self.training_data.y
        if smc_gi.USING_GPU:
            gi.set_use_gpu(True)
            self.training_data_gpu = ExplicitTrainingData(gi.num_lib.asarray(self.training_data.x),
                                                          gi.num_lib.asarray(self.training_data.y))
            training_data_y = self.training_data_gpu.y

        #    self.training_data.x = gi.num_lib.asarray(self.training_data.x)
        #    self.training_data.y = gi.num_lib.asarray(self.training_data.y)

            """
            new_proposal = [None] * len(proposal)
            for i, prop in enumerate(proposal):
                if type(prop) is dict:
                    new_proposal[i] = {}
                    for key in prop:
                        new_proposal[i][key] = gi.num_lib.asarray(prop[key])
                elif type(prop) is np.ndarray:
                    new_proposal[i] = gi.num_lib.asarray(prop)
            proposal = new_proposal
            """

        evaluate_model = lambda x: self.evaluate_model(x, individual) if not smc_gi.USING_GPU else self.evaluate_model_gpu(x, individual)
        vector_mcmc = VectorMCMC(evaluate_model,
                                 training_data_y.flatten(), priors,
                                 log_like_args=self._std)

        mcmc_kernel = VectorMCMCKernel(vector_mcmc, param_order=param_names)
        smc = SMCSampler(mcmc_kernel)

        try:
            step_list, marginal_log_likes = smc.sample(self._num_particles,
                                                       self._mcmc_steps,
                                                       self._phi_seq,
                                                       self._ess_threshold,
                                                       proposal)

        except (ValueError, np.linalg.LinAlgError, ZeroDivisionError):
            return np.nan

        means = step_list[-1].compute_mean(package=False)
        individual.set_local_optimization_params(means[:-1])


        nmll = -1 * (marginal_log_likes[-1] -
                     marginal_log_likes[self._fbf_phi_idx])


        #if smc_gi.USING_GPU:
        #    self.training_data.x = self.training_data.x.get()
        #    self.training_data.y = self.training_data.y.get()

        if self._return_nmll_only:
            return nmll
        return nmll, step_list, vector_mcmc 

    def get_parameter_names(self, individual):
        num_params = individual.get_number_local_optimization_params()
        return [f'p{i}' for i in range(num_params)]

    def do_local_opt(self, individual):
        individual._notify_modification()
        individual._needs_opt = True
        _ = self._cont_local_opt(individual)
        return individual

    def estimate_covariance(self, individual):
        num_params = individual.get_number_local_optimization_params()
        x = self.training_data.x
        f, f_deriv = individual.evaluate_equation_with_local_opt_gradient_at(x)
        ssqe = np.sum((self.training_data.y - f) ** 2)
        var_ols = ssqe / (len(f) - num_params)
        cov = var_ols * np.linalg.inv(f_deriv.T.dot(f_deriv))
        return cov, var_ols, ssqe

    def generate_proposal_samples(self, individual, num_samples):
        param_names = self.get_parameter_names(individual)
        pdf = np.ones((num_samples, 1))
        samples = np.ones((num_samples, len(param_names)))

        cov, var_ols, ssqe = self.estimate_covariance(individual)
        if param_names != []:
            proposal = mvn(individual.constants, cov, allow_singular=True)
            samples = proposal.rvs(num_samples).reshape(num_samples, -1)
            pdf *= proposal.pdf(samples).reshape(-1, 1)

        if self._std is None:
            len_data = len(self.training_data.x)
            noise_proposal = invgamma((0.01 + len_data) / 2,
                                      scale=(0.01 * var_ols + ssqe) / 2)
            noise_samples = noise_proposal.rvs(num_samples).reshape(-1, 1)
            pdf *= noise_proposal.pdf(noise_samples).reshape(-1, 1)

            param_names.append('std_dev')
            samples = np.concatenate((samples, noise_samples), axis=1)

        samples = dict(zip(param_names, samples.T))
        return samples, pdf

    def evaluate_model(self, params, individual):
        self._eval_count += 1
        individual.set_local_optimization_params(params.T)

        return individual.evaluate_equation_at(self.training_data.x).T

    def evaluate_model_gpu(self, params, individual):
        params = gi.num_lib.asarray(params)

        self._eval_count += 1
        individual.set_local_optimization_params(params.T)

        output = individual.evaluate_equation_at(self.training_data_gpu.x).T
        return output

    def _calc_phi_sequence(self, phi_exponent):
        x = np.linspace(0, 1, self._smc_steps - 1)
        phi_seq = (np.exp(x * phi_exponent) - 1) / (np.exp(phi_exponent) - 1)
        fbf_phi = 1 / np.sqrt(self._num_observations)
        fbf_phi_index = np.searchsorted(phi_seq, [fbf_phi])
        phi_seq = np.insert(phi_seq, fbf_phi_index, fbf_phi)
        return int(fbf_phi_index), phi_seq

    @property
    def eval_count(self):
        return self._eval_count + self._cont_local_opt.eval_count

    @eval_count.setter
    def eval_count(self, value):
        self._eval_count = value - self._cont_local_opt.eval_count

    @property
    def training_data(self):
        return self._cont_local_opt.training_data

    @training_data.setter
    def training_data(self, training_data):
        self._cont_local_opt.training_data = training_data