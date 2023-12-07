import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.stats import invgamma

from smcpy import (
    VectorMCMC,
    VectorMCMCKernel,
    AdaptiveSampler,
    ImproperUniform
)
from smcpy.paths import GeometricPath
from smcpy.proposals import MultivarIndependent

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
        reuse_starting_point=True,
    ):

        self._num_particles = num_particles
        self._mcmc_steps = mcmc_steps
        self._ess_threshold = ess_threshold
        self._std = std
        self._num_multistarts = num_multistarts
        self._objective_fn = objective_fn
        self._deterministic_optimizer = deterministic_optimizer
        self._reuse_starting_point = reuse_starting_point

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

    def __call__(self, individual):
        try:
            proposal = self._generate_proposals(individual)
        except (ValueError, np.linalg.LinAlgError, RuntimeError) as e:
            return np.nan, "proposal error", e

        param_names = self._get_parameter_names(individual)
        priors = [ImproperUniform() for _ in range(len(param_names))]
        if self._std is None:
            priors.append(ImproperUniform(0, None))
            param_names.append("std_dev")

        path = GeometricPath(proposal=proposal, required_phi=self._norm_phi)
        vmcmc = VectorMCMC(
            lambda x: self.evaluate_model(x, individual),
            np.zeros(len(self.training_data)),
            priors,
            log_like_args=self._std,
        )
        kernel = VectorMCMCKernel(vmcmc, param_order=param_names, path=path)
        smc = AdaptiveSampler(kernel)

        try:
            step_list, marginal_log_likes = smc.sample(
               self._num_particles,
               self._mcmc_steps,
               self._ess_threshold,
               progress_bar=False,
            )
        except (ValueError, np.linalg.LinAlgError, ZeroDivisionError) as e:
            # print(e)
            return np.nan, "sample error", e

        max_idx = np.argmax(step_list[-1].log_likes)
        maps = step_list[-1].params[max_idx]
        individual.set_local_optimization_params(maps[:-1])

        log_nml = (
            marginal_log_likes[-1] - marginal_log_likes[smc.req_phi_index[0]]
        )

        return log_nml, step_list, vmcmc

    def _generate_proposals(self, individual):
        param_names = self._get_parameter_names(individual)
        num_multistarts = self._num_multistarts

        param_dists = []
        cov_estimates = []
        if not param_names:
            cov_estimates.append(self._estimate_covariance(individual))
        else:
            for i in range(3 * num_multistarts):
                try:
                    do_det_opt = not self._reuse_starting_point or i != 0
                    mean, cov, var_ols, ssqe = self._estimate_covariance(
                        individual, do_det_opt
                    )
                    cov = 0.5*(cov + cov.T) # ensuring symmetry 
                    evals, Q = np.linalg.eig(cov)
                    if np.min(evals) < 0:  # this approximation attempts to correct for cov matrices that are not positive semidefinite
                        D = np.diag(evals)
                        D[D<0] = 0
                        cov = Q.dot(D).dot(Q.T)
                    dists = mvn(mean, cov, allow_singular=True)
                except (ValueError, np.linalg.LinAlgError) as e:
                    continue
                cov_estimates.append((mean, cov, var_ols, ssqe))
                param_dists.append(dists)
                if len(param_dists) == num_multistarts:
                    break
            if not param_dists:
                raise RuntimeError(
                    "Could not generate any valid proposal distributions"
                )

            param_mix_dist = MixtureDist(*param_dists)

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
            param_names.append("std_dev")

            noise_mix_dist = MixtureDist(*noise_dists)

        return MultivarIndependent(param_mix_dist, noise_mix_dist)

    @staticmethod
    def _get_parameter_names(individual):
        num_params = individual.get_number_local_optimization_params()
        return [f"p{i}" for i in range(num_params)]

    def _estimate_covariance(self, individual, do_det_opt=True):
        if do_det_opt:
            self._deterministic_optimizer(individual)
        
        # # RALPH data approx method
        # f, f_deriv = self._objective_fn.get_fitness_vector_and_jacobian(
        #     individual
        # )
        # ssqe = np.sum((f) ** 2)
        # var_ols = ssqe / len(f)
        # cov = var_ols * np.linalg.inv(f_deriv.T.dot(f_deriv))

        # LAPLACE approx
        f, g = self._objective_fn.get_fitness_vector_and_jacobian(
            individual
        )
        h = np.squeeze(individual.evaluate_with_local_opt_hessian_at(self.objective_fn.training_data.x)[1].detach().numpy(),1)
        A = 2*np.sum(np.einsum('...i,...j->...ij', g, g) 
                     + np.expand_dims(f, axis=(1,2))*h, axis=0)
        ssqe = np.sum((f) ** 2)
        var_ols = ssqe / len(f)
        cov = np.linalg.inv(A)
        return individual.constants, cov, var_ols, ssqe

    def evaluate_model(self, params, individual):
        individual.set_local_optimization_params(params.T)
        result = self._objective_fn.evaluate_fitness_vector(individual).T
        if len(result.shape) < 2:
            # TODO, would it be better to remove the flatten in explicit
            # regression and add a flatten to the scipy wrapper?
            result = result.reshape(-1, len(self._objective_fn.training_data))
        return result


class MixtureDist:

    def __init__(self, *args):
        self._dists = args

    def rvs(self, num_samples):
        candidate_samples = np.zeros((
            len(self._dists),
            num_samples,
            self._dists[0].dim if hasattr(self._dists[0], 'dim') else 1
        ))

        for i, d in enumerate(self._dists):
            candidate_samples[i, :, :] = \
                d.rvs(num_samples).reshape(num_samples, -1)

        rng = np.random.default_rng()
        dist_indices = rng.integers(0, len(self._dists), num_samples)
        sample_indices = np.arange(0, num_samples)

        return candidate_samples[dist_indices, sample_indices, :]

    def logpdf(self, x):
        num_samples = x.shape[0]
        pdfs = np.zeros((len(self._dists), num_samples, 1))

        for i, d in enumerate(self._dists):
            pdfs[i, :, :] = d.pdf(x).reshape(num_samples, 1)

        return np.log(pdfs.sum(axis=0) / len(self._dists))