"""Sequential-Monte-Carlo evidence estimation for Expressions."""

# The public estimator intentionally has several sampling controls. SMCPy must
# be imported lazily because it is an optional evidence-estimation dependency.
# pylint: disable=broad-exception-caught,import-outside-toplevel,protected-access,too-many-arguments,too-many-instance-attributes,too-many-locals,too-many-positional-arguments

from dataclasses import dataclass, field
import hashlib

import numpy as np
from scipy import optimize
from scipy.stats import invgamma, multivariate_normal

from .fitting import FitResult, ScipyFitter


class _UserCodeError(Exception):
    """Mark an exception raised by a user-provided fitter or measure."""

    def __init__(self, error):
        super().__init__(str(error))
        self.error = error


@dataclass(frozen=True)
class EvidenceResult:
    """The outcome of an Evidence estimation attempt.

    ``smc_nmll`` is higher-is-better. Failed estimates have negative infinite
    NMLL and do not contain a MAP estimate.

    Parameters
    ----------
    smc_nmll : float
        Higher-is-better normalized marginal log likelihood.
    map_constants : tuple of float or None
        Posterior MAP Expression constants when estimation succeeds.
    success : bool
        Whether sampling and Evidence calculation succeeded.
    message : str, optional
        Failure or status message.
    diagnostics : dict, optional
        Estimator diagnostics.
    posterior : object, optional
        Posterior samples when explicitly requested.
    """

    smc_nmll: float
    map_constants: tuple[float, ...] | None
    success: bool
    message: str | None = None
    diagnostics: dict = field(default_factory=dict)
    posterior: object | None = None


def smc_nmll_loss(result):
    """Adapt an Evidence result to Bingo's lower-is-better Loss convention.

    Parameters
    ----------
    result : EvidenceResult
        Evidence estimation outcome.

    Returns
    -------
    float
        The negated NMLL, or infinity when estimation failed.
    """
    if not result.success or not np.isfinite(result.smc_nmll):
        return np.inf
    return -result.smc_nmll


class SmcEvidenceEstimator:
    """Estimate Expression evidence with an SMC sampler and Laplace proposal.

    Parameters
    ----------
    num_particles, mcmc_steps, ess_threshold : optional
        SMCPy sampling controls.
    std : float, optional
        Known observation noise. When omitted it is inferred by SMC.
    num_multistarts : int, optional
        Number of Laplace proposal components.
    seed : int, optional
        Root seed. The expression's raw structure derives the sampler stream.
    return_posterior : bool, optional
        Include posterior steps in :class:`EvidenceResult` when true.
    fitter : callable, optional
        Fitter used only to construct the Laplace proposal.
    """

    def __init__(
        self,
        *,
        num_particles=150,
        mcmc_steps=12,
        ess_threshold=0.75,
        std=None,
        num_multistarts=1,
        seed=0,
        return_posterior=False,
        fitter=None,
    ):
        self.num_particles = num_particles
        self.mcmc_steps = mcmc_steps
        self.ess_threshold = ess_threshold
        self.std = std
        self.num_multistarts = num_multistarts
        self.seed = seed
        self.return_posterior = return_posterior
        self.fitter = fitter if fitter is not None else ScipyFitter("least_squares")

    def estimate(self, expression, data, residual_measure):
        """Estimate evidence, installing posterior MAP constants on success.

        Parameters
        ----------
        expression : Expression
            Expression whose constants are estimated.
        data : ObjectiveData
            Aligned data consumed by ``residual_measure``.
        residual_measure : callable
            Returns residuals for an Expression, data, and candidate constants.

        Returns
        -------
        EvidenceResult
            The Evidence estimate and optional posterior samples.

        Raises
        ------
        Exception
            Any exception raised by a user-supplied fitter or residual measure.
        """
        try:
            from smcpy import (
                AdaptiveSampler,
                ImproperUniform,
                VectorMCMC,
                VectorMCMCKernel,
            )
            from smcpy.paths import GeometricPath
            from smcpy.proposals import MultivarIndependent
        except ImportError as error:
            return self._failure("SMCPy is required for evidence estimation", error)
        try:
            proposal, diagnostics = self._generate_proposal(
                expression, data, residual_measure, MultivarIndependent
            )
        except _UserCodeError as error:
            raise error.error from error
        except (TypeError, ValueError, np.linalg.LinAlgError, RuntimeError) as error:
            return self._failure("proposal error", error)

        parameter_count = len(expression.constants)
        priors = [ImproperUniform() for _ in range(parameter_count)]
        parameter_names = [f"c{i}" for i in range(parameter_count)]
        if self.std is None:
            priors.append(ImproperUniform(0, None))
            parameter_names.append("std_dev")

        def evaluate_model(params):
            params = np.asarray(params, dtype=float)
            if params.ndim == 1:
                params = params.reshape(1, -1)
            constants = params[:, :parameter_count].T
            residuals = np.asarray(
                [residual_measure(expression, data, candidate) for candidate in constants.T],
                dtype=float,
            )
            return residuals

        try:
            with self._sampling_seed(expression):
                generator = self._generator(expression)
                path = GeometricPath(
                    proposal=proposal, required_phi=1 / np.sqrt(len(data))
                )
                mcmc = VectorMCMC(
                    evaluate_model,
                    np.zeros(len(data)),
                    priors,
                    log_like_args=self.std,
                )
                mcmc.rng = generator
                sampler = AdaptiveSampler(
                    VectorMCMCKernel(
                        mcmc,
                        param_order=parameter_names,
                        path=path,
                        rng=generator,
                    ),
                    show_progress_bar=False,
                )
                steps, marginal_log_likes = sampler.sample(
                    self.num_particles,
                    self.mcmc_steps,
                    self.ess_threshold,
                    resample_rng=lambda _, size: generator.uniform(0, 1, size),
                )
            final_step = steps[-1]
            map_parameters = np.asarray(
                final_step.params[np.argmax(final_step.log_likes)], dtype=float
            )
            map_constants = tuple(map_parameters[:parameter_count])
            phi_index = np.argmin(
                np.abs(np.asarray(sampler._phi_sequence) - 1 / np.sqrt(len(data)))
            )
            smc_nmll = float(marginal_log_likes[-1] - marginal_log_likes[phi_index])
            if not np.isfinite(smc_nmll) or not np.all(np.isfinite(map_constants)):
                raise ValueError("SMC returned non-finite evidence or MAP constants")
        except _UserCodeError as error:
            raise error.error from error
        except Exception as error:  # SMCPy exposes several backend-specific errors.
            return self._failure("sample error", error, diagnostics)

        try:
            expression.commit_fit(map_constants)
        except (TypeError, ValueError) as error:
            return self._failure("MAP commit error", error, diagnostics)
        return EvidenceResult(
            smc_nmll,
            map_constants,
            True,
            diagnostics=diagnostics,
            posterior=steps if self.return_posterior else None,
        )

    def _generate_proposal(self, expression, data, residual_measure, proposal_type):
        components = []
        noise_components = []
        generator = self._generator(expression)
        for index in range(3 * self.num_multistarts):
            trial = expression.copy()
            if index:
                trial.constants = np.asarray(expression.constants) + generator.normal(
                    scale=0.01, size=len(expression.constants)
                )
            try:
                result = self._call_user_code(self.fitter, trial, data, residual_measure)
                if not isinstance(result, FitResult):
                    raise TypeError("evidence proposal fitter must return a FitResult")
                constants = np.asarray(result.constants, dtype=float)
                residuals = self._residuals(expression, data, residual_measure, constants)
                covariance = self._laplace_covariance(
                    expression, data, residual_measure, constants, residuals
                )
                components.append(
                    multivariate_normal(constants, covariance, allow_singular=True)
                )
                ssqe = float(np.dot(residuals, residuals))
                variance = ssqe / len(residuals)
                residual_scale = np.sqrt(np.mean(np.square(residuals)))
                noise_components.append(
                    SqrtInvGamma(
                        (0.01 + len(residuals)) / 2,
                        max(
                            (0.01 * variance + ssqe) / 2,
                            1e-12 * max(residual_scale, 1.0),
                        ),
                    )
                )
            except (TypeError, ValueError, np.linalg.LinAlgError):
                continue
            if len(components) == self.num_multistarts:
                break
        if not components:
            raise RuntimeError("could not generate a proposal distribution")
        distributions = [MixtureDistribution(*components)]
        if self.std is None:
            distributions.append(MixtureDistribution(*noise_components))
        return proposal_type(*distributions), {"proposal_components": len(components)}

    @staticmethod
    def _residuals(expression, data, measure, constants):
        residuals = np.asarray(
            SmcEvidenceEstimator._call_user_code(measure, expression, data, constants),
            dtype=float,
        )
        if residuals.ndim != 1 or not np.all(np.isfinite(residuals)):
            raise ValueError("residual measure must return finite one-dimensional values")
        return residuals

    def _laplace_covariance(self, expression, data, measure, constants, residuals):
        jacobian = getattr(measure, "jacobian", None)
        if jacobian is None:
            jacobian = self._numerical_jacobian(expression, data, measure, constants)
        else:
            try:
                jacobian = np.asarray(
                    self._call_user_code(jacobian, expression, data, constants), dtype=float
                )
            except AttributeError:
                jacobian = self._numerical_jacobian(
                    expression, data, measure, constants
                )
        if jacobian.shape != (len(residuals), len(constants)):
            raise ValueError("residual Jacobian has an invalid shape")
        curvature = 2 * jacobian.T @ jacobian
        residual_hessian = getattr(measure, "residual_hessian", None)
        if residual_hessian is not None:
            try:
                hessians = np.asarray(
                    self._call_user_code(
                        residual_hessian, expression, data, constants
                    ),
                    dtype=float,
                )
            except AttributeError:
                hessians = None
            if hessians is None:
                return self._regularized_covariance(curvature)
            if hessians.shape != (len(residuals), len(constants), len(constants)):
                raise ValueError("residual Hessian has an invalid shape")
            curvature += 2 * np.einsum("i,ijk->jk", residuals, hessians)
        return self._regularized_covariance(curvature)

    @staticmethod
    def _regularized_covariance(curvature):
        curvature = 0.5 * (curvature + curvature.T)
        eigenvalues, eigenvectors = np.linalg.eigh(curvature)
        scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
        eigenvalues = np.maximum(eigenvalues, scale * 1e-12)
        covariance = (eigenvectors / eigenvalues) @ eigenvectors.T
        return 0.5 * (covariance + covariance.T)

    def _numerical_jacobian(self, expression, data, measure, constants):
        return optimize._numdiff.approx_derivative(
            lambda values: self._residuals(expression, data, measure, values), constants
        )

    @staticmethod
    def _call_user_code(callable_, *args):
        try:
            return callable_(*args)
        except Exception as error:
            raise _UserCodeError(error) from error

    def _generator(self, expression):
        if self.seed is None:
            return np.random.default_rng()
        digest = hashlib.blake2b(
            str(self.seed).encode() + self._structure_bytes(expression), digest_size=16
        ).digest()
        return np.random.default_rng(int.from_bytes(digest, "big"))

    def _sampling_seed(self, expression):
        return _SamplingSeed(self._generator(expression))

    @staticmethod
    def _structure_bytes(expression):
        commands = np.asarray(expression.raw_command_array, dtype=np.uint8)
        integers = np.asarray(getattr(expression, "raw_integers", ()), dtype=np.int64)
        return (
            repr(commands.shape).encode()
            + commands.tobytes()
            + integers.tobytes()
        )

    @staticmethod
    def _failure(message, error, diagnostics=None):
        return EvidenceResult(
            -np.inf,
            None,
            False,
            f"{message}: {error}",
            diagnostics or {},
        )


class _SamplingSeed:
    """Temporarily seed libraries that rely on NumPy's legacy global state."""

    def __init__(self, generator):
        self._seed = int(generator.integers(0, 2**32, dtype=np.uint32))
        self._state = None

    def __enter__(self):
        self._state = np.random.get_state()
        np.random.seed(self._seed)

    def __exit__(self, *_):
        np.random.set_state(self._state)


class MixtureDistribution:
    """Equal-weight mixture compatible with SMCPy's independent proposal."""

    def __init__(self, *distributions):
        self._distributions = distributions

    def rvs(self, num_samples, random_state=None):
        """Draw equally from each component distribution."""
        generator = (
            np.random.default_rng(random_state)
            if random_state is not None
            else np.random
        )
        if random_state is None:
            indices = generator.randint(len(self._distributions), size=num_samples)
        else:
            indices = generator.integers(len(self._distributions), size=num_samples)
        samples = [
            np.asarray(distribution.rvs(num_samples, random_state=random_state)).reshape(
                num_samples, -1
            )
            for distribution in self._distributions
        ]
        return np.asarray(samples)[indices, np.arange(num_samples)]

    def logpdf(self, values):
        """Return the equal-weight mixture log density."""
        densities = np.array(
            [distribution.pdf(values) for distribution in self._distributions]
        )
        return np.log(np.mean(densities, axis=0)).reshape(-1, 1)


class SqrtInvGamma:
    """Square-root transformed inverse-gamma distribution for noise proposals."""

    def __init__(self, shape, scale):
        self._distribution = invgamma(shape, scale=scale)

    def rvs(self, *args, **kwargs):
        """Draw standard deviations from the transformed distribution."""
        return np.sqrt(self._distribution.rvs(*args, **kwargs))

    def pdf(self, values):
        """Evaluate the transformed density."""
        values = np.asarray(values)
        return 2 * values * self._distribution.pdf(np.square(values))
