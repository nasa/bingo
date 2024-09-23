"""A module for probabilistic calibration of parameters.

Probabilistic calibration of model parameters can be useful in cases where data
is sparse and/or noisy.  Using a calibration of this type can allow for a better 
estimate of true fitness while being a bit more robust to overfitting.
"""

import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.stats import invgamma

from smcpy.mcmc.vector_mcmc import VectorMCMC
from smcpy.mcmc.vector_mcmc_kernel import VectorMCMCKernel
from smcpy import AdaptiveSampler
from smcpy import ImproperUniform

from .local_optimizer import LocalOptimizer


class SmcpyOptimizer(LocalOptimizer):
    """An optimizer that uses SMCPy for probabilistic parameter calibration

    A class for probabilistic parameter calibration for the parameters of a
    `Chromosome` using SMCPy

    Parameters
    ----------
    objective_fn : VectorBasedFunction, VectorGradientMixin
        A `VectorBasedFunction` with `VectorGradientMixin` (e.g.,
        ExplicitRegression).  It should produce a vector where the target value
        is 0.
    deterministic_optimizer : LocalOptimizer
        A deterministic local optimizer e.g., `ScipyOptimizer`
    num_particles : int
        The number of particles to use in the SMC approximation
    mcmc_steps : int
        The number of MCMC steps to perform with each SMC update
    ess_threshold : float (0-1)
        The effective sample size (ratio) below which SMC particles will be
        resampled
    std : float
        (Optional) The fixed noise level, if it is known
    num_multistarts : int
        (Optional) The number of deterministic optimizations performed when
        developing the SMC proposal
    uniformly_weighted_proposal : bool
        Whether to equally weight particles in the proposal. Default True.

    Attributes
    ----------
    objective_fn
        A function to minimize which can be evaluated by passing in a
        `Chromosome`
    options : dict
        Additional arguments for clo options

    """

    def __init__(
        self,
        objective_fn,
        deterministic_optimizer,
        num_particles=150,
        mcmc_steps=12,
        ess_threshold=0.75,
        std=None,
        num_multistarts=1,
        uniformly_weighted_proposal=True,
    ):

        self._num_particles = num_particles
        self._mcmc_steps = mcmc_steps
        self._ess_threshold = ess_threshold
        self._std = std
        self._num_multistarts = num_multistarts
        self._uniformly_weighted_proposal = uniformly_weighted_proposal
        self._objective_fn = objective_fn
        self._deterministic_optimizer = deterministic_optimizer

        self._norm_phi = self._calculate_norm_phi()

    def _calculate_norm_phi(self):
        num_observations = len(self.training_data)
        return 1 / np.sqrt(num_observations)

    @property
    def objective_fn(self):
        """A `VectorBasedFunction` with `VectorGradientMixin` (e.g.,
        ExplicitRegression). It should produce a vector where the target value
        is 0."""
        return self._objective_fn

    @objective_fn.setter
    def objective_fn(self, value):
        self._objective_fn = value

    @property
    def training_data(self):
        """Training data used in objective function"""
        return self._objective_fn.training_data

    @training_data.setter
    def training_data(self, value):
        self._objective_fn.training_data = value
        self._norm_phi = self._calculate_norm_phi()

    @property
    def eval_count(self):
        """int : the number of evaluations that have been performed"""
        return self._objective_fn.eval_count

    @eval_count.setter
    def eval_count(self, value):
        self._objective_fn.eval_count = value

    @property
    def options(self):
        """dict : optimizer's options"""
        return {
            "num_particles": self._num_particles,
            "mcmc_steps": self._mcmc_steps,
            "ess_threshold": self._ess_threshold,
            "std": self._std,
            "num_multistarts": self._num_multistarts,
            "uniformly_weighted_proposal": self._uniformly_weighted_proposal,
        }

    @options.setter
    def options(self, value):
        if "num_particles" in value:
            self._num_particles = value["num_particles"]
        if "mcmc_steps" in value:
            self._mcmc_steps = value["mcmc_steps"]
        if "ess_threshold" in value:
            self._ess_threshold = value["ess_threshold"]
        if "std" in value:
            self._std = value["std"]
        if "num_multistarts" in value:
            self._num_multistarts = value["num_multistarts"]
        if "uniformly_weighted_proposal" in value:
            self._uniformly_weighted_proposal = value["uniformly_weighted_proposal"]

    def __call__(self, individual):
        try:
            proposal = self._generate_proposal_samples(individual, self._num_particles)
        except (ValueError, np.linalg.LinAlgError, RuntimeError) as e:
            return np.nan, "proposal error", e

        param_names = self._get_parameter_names(individual)
        priors = [ImproperUniform() for _ in range(len(param_names))]
        if self._std is None:
            priors.append(ImproperUniform(0, None))
            param_names.append("std_dev")

        vector_mcmc = VectorMCMC(
            lambda x: self.evaluate_model(x, individual),
            np.zeros(len(self.training_data)),
            priors,
            log_like_args=self._std,
        )
        mcmc_kernel = VectorMCMCKernel(vector_mcmc, param_order=param_names)
        smc = AdaptiveSampler(mcmc_kernel)

        try:
            step_list, marginal_log_likes = smc.sample(
                self._num_particles,
                self._mcmc_steps,
                self._ess_threshold,
                proposal=proposal,
                required_phi=self._norm_phi,
                progress_bar=False,
            )
        except (ValueError, np.linalg.LinAlgError, ZeroDivisionError) as e:
            # print(e)
            return np.nan, "sample error", e

        max_idx = np.argmax(step_list[-1].log_likes)
        maps = step_list[-1].params[max_idx]
        individual.set_local_optimization_params(maps[:-1])

        log_nml = marginal_log_likes[-1] - marginal_log_likes[smc.req_phi_index[0]]

        return log_nml, step_list, vector_mcmc

    def _generate_proposal_samples(self, individual, num_samples):
        param_names = self._get_parameter_names(individual)
        pdf = np.ones((num_samples, 1))
        samples = np.ones((num_samples, len(param_names)))

        num_multistarts = self._num_multistarts
        param_dists = []
        cov_estimates = []
        if not param_names:
            cov_estimates.append(self._estimate_covariance(individual))
        else:
            for _ in range(8 * num_multistarts):
                mean, cov, var_ols, ssqe = self._estimate_covariance(individual)
                try:
                    dists = mvn(mean, cov, allow_singular=True)
                except ValueError as _:
                    continue
                cov_estimates.append((mean, cov, var_ols, ssqe))
                param_dists.append(dists)
                if len(param_dists) == num_multistarts:
                    break
            if not param_dists:
                raise RuntimeError(
                    "Could not generate any valid proposal distributions"
                )

            pdf, samples = self._get_samples_and_pdf(param_dists, num_samples)

        if self._std is None:
            len_data = len(self.training_data)
            scale_data = np.sqrt(
                np.mean(np.square(self.training_data.y))
            )  # TODO can we do this differently without knowing what the training data is?
            noise_dists = []
            for _, _, var_ols, ssqe in cov_estimates:
                shape = (0.01 + len_data) / 2
                scale = max((0.01 * var_ols + ssqe) / 2, 1e-12 * scale_data)
                noise_dists.append(invgamma(shape, scale=scale))
            noise_pdf, noise_samples = self._get_samples_and_pdf(
                noise_dists, num_samples
            )

            param_names.append("std_dev")
            samples = np.concatenate((samples, np.sqrt(noise_samples)), axis=1)
            pdf *= noise_pdf

        if self._uniformly_weighted_proposal:
            pdf = np.ones_like(pdf)

        samples = dict(zip(param_names, samples.T))
        return samples, pdf

    @staticmethod
    def _get_parameter_names(individual):
        num_params = individual.get_number_local_optimization_params()
        return [f"p{i}" for i in range(num_params)]

    def _estimate_covariance(self, individual):
        self._deterministic_optimizer(individual)
        f, f_deriv = self._objective_fn.get_fitness_vector_and_jacobian(individual)
        ssqe = np.sum((f) ** 2)
        var_ols = ssqe / len(f)
        cov = var_ols * np.linalg.inv(f_deriv.T.dot(f_deriv))
        return individual.constants, cov, var_ols, ssqe

    @staticmethod
    def _get_samples_and_pdf(distributions, num_samples):
        sub_samples = num_samples // len(distributions)
        samples = np.vstack(
            [
                proposal.rvs(sub_samples).reshape(sub_samples, -1)
                for proposal in distributions
            ]
        )
        if samples.shape[0] != num_samples:
            missed_samples = num_samples - samples.shape[0]
            new_samples = (
                np.random.choice(distributions)
                .rvs(missed_samples)
                .reshape((missed_samples, -1))
            )
            samples = np.vstack([samples, new_samples])
        pdf = np.zeros((samples.shape[0], 1))
        for dist in distributions:
            pdf += dist.pdf(samples).reshape(-1, 1)
        pdf /= len(distributions)
        return pdf, samples

    def evaluate_model(self, params, individual):
        """Evaluate an individual given a set of parameters

        Parameters
        ----------
        params : numpy array
            parameters for which to evaluate the individual
        individual : Equation
            individual for which to evaluate fitness

        Returns
        -------
        numpy array : fitness vector outputs for the individual w/ the params
        """
        individual.set_local_optimization_params(params.T)
        result = self._objective_fn.evaluate_fitness_vector(individual).T
        if len(result.shape) < 2:
            # TODO, would it be better to remove the flatten in explicit
            # regression and add a flatten to the scipy wrapper?
            result = result.reshape(-1, len(self._objective_fn.training_data))
        return result
