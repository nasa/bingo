import numpy as np

from scipy.stats import multivariate_normal as mvn
from scipy.stats import invgamma

from bingo.evaluation.fitness_function import FitnessFunction

from smcpy.mcmc.vector_mcmc import VectorMCMC
from smcpy.mcmc.vector_mcmc_kernel import VectorMCMCKernel
from smcpy import AdaptiveSampler
from smcpy import ImproperUniform


class BayesFitnessFunction(FitnessFunction):

    def __init__(self, continuous_local_opt, num_particles=150, mcmc_steps=12,
                 ess_threshold=0.75, std=None,
                 return_nmll_only=True, num_multistarts=1,
                 uniformly_weighted_proposal=True):

        self._num_particles = num_particles
        self._mcmc_steps = mcmc_steps
        self._ess_threshold = ess_threshold
        self._std = std
        self._return_nmll_only = return_nmll_only
        self._num_multistarts = num_multistarts
        self._uniformly_weighted_proposal = uniformly_weighted_proposal

        num_observations = len(continuous_local_opt.training_data.x)
        self._norm_phi = 1 / np.sqrt(num_observations)

        self._cont_local_opt = continuous_local_opt
        self._eval_count = 0

    def __call__(self, individual):
        param_names = self.get_parameter_names(individual)
        individual = self.do_local_opt(individual)
        try:
            proposal = self.generate_proposal_samples(individual,
                                                      self._num_particles)
        except (ValueError, np.linalg.LinAlgError, RuntimeError) as e:
            if self._return_nmll_only:
                return np.nan
            return np.nan, None, None

        priors = [ImproperUniform() for _ in range(len(param_names))]
        if self._std is None:
            priors.append(ImproperUniform(0, None))
            param_names.append('std_dev')

        vector_mcmc = VectorMCMC(lambda x: self.evaluate_model(x, individual),
                                 self.training_data.y.flatten(), priors,
                                 log_like_args=self._std)

        mcmc_kernel = VectorMCMCKernel(vector_mcmc, param_order=param_names)
        smc = AdaptiveSampler(mcmc_kernel)

        try:
            step_list, marginal_log_likes = \
                smc.sample(self._num_particles, self._mcmc_steps,
                           self._ess_threshold,
                           proposal=proposal,
                           required_phi=self._norm_phi)
        except (ValueError, np.linalg.LinAlgError, ZeroDivisionError):
            if self._return_nmll_only:
                return np.nan
            return np.nan, None, None

        max_idx = np.argmax(step_list[-1].log_likes)
        maps = step_list[-1].params[max_idx]
        individual.set_local_optimization_params(maps[:-1])

        nmll = -1 * (marginal_log_likes[-1] -
                     marginal_log_likes[smc.req_phi_index[0]])

        if self._return_nmll_only:
            return nmll
        return nmll, step_list, vector_mcmc

    @staticmethod
    def get_parameter_names(individual):
        num_params = individual.get_number_local_optimization_params()
        return [f'p{i}' for i in range(num_params)]

    def do_local_opt(self, individual):
        individual._notify_modification()
        individual._needs_opt = True
        _ = self._cont_local_opt(individual)
        return individual

    def estimate_covariance(self, individual):
        self.do_local_opt(individual)
        num_params = individual.get_number_local_optimization_params()
        x = self.training_data.x
        f, f_deriv = individual.evaluate_equation_with_local_opt_gradient_at(x)
        ssqe = np.sum((self.training_data.y - f) ** 2)
        var_ols = ssqe / (len(f) - num_params)
        cov = var_ols * np.linalg.inv(f_deriv.T.dot(f_deriv))
        return individual.constants, cov, var_ols, ssqe

    def generate_proposal_samples(self, individual, num_samples):
        param_names = self.get_parameter_names(individual)
        pdf = np.ones((num_samples, 1))
        samples = np.ones((num_samples, len(param_names)))

        num_multistarts = self._num_multistarts
        param_dists = []
        cov_estimates = []
        if not param_names:
            cov_estimates.append(self.estimate_covariance(individual))
        else:
            for _ in range(8*num_multistarts):
                mean, cov, var_ols, ssqe = self.estimate_covariance(individual)
                try:
                    dists = mvn(mean, cov, allow_singular=True)
                except ValueError as e:
                    continue
                cov_estimates.append((mean, cov, var_ols, ssqe))
                param_dists.append(dists)
                if len(param_dists) == num_multistarts:
                    break
            if not param_dists:
                raise RuntimeError('Could not generate any valid proposal '
                                   'distributions')

            pdf, samples = self._get_samples_and_pdf(param_dists, num_samples)

        if self._std is None:
            len_data = len(self.training_data.x)
            noise_dists = [invgamma((0.01 + len_data) / 2,
                                    scale=(0.01 * var_ols + ssqe) / 2)
                           for _, _, var_ols, ssqe in cov_estimates]
            noise_pdf, noise_samples = self._get_samples_and_pdf(noise_dists,
                                                                 num_samples)

            param_names.append('std_dev')
            samples = np.concatenate((samples, noise_samples), axis=1)
            pdf *= noise_pdf

        if self._uniformly_weighted_proposal:
            pdf = np.ones_like(pdf)

        samples = dict(zip(param_names, samples.T))
        return samples, pdf

    @staticmethod
    def _get_samples_and_pdf(distributions, num_samples):
        sub_samples = num_samples // len(distributions)
        samples = np.vstack([proposal.rvs(sub_samples).reshape(sub_samples, -1)
                             for proposal in distributions])
        if samples.shape[0] != num_samples:
            missed_samples = num_samples - samples.shape[0]
            new_samples = np.random.choice(len(distributions)) \
                .rvs(missed_samples).reshape((missed_samples, -1))
            samples = np.vstack([samples, new_samples])
        pdf = np.zeros((samples.shape[0], 1))
        for dist in distributions:
            pdf += dist.pdf(samples).reshape(-1, 1)
        pdf /= len(distributions)
        return pdf, samples

    def evaluate_model(self, params, individual):
        self._eval_count += 1
        individual.set_local_optimization_params(params.T)
        return individual.evaluate_equation_at(self.training_data.x).T

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
