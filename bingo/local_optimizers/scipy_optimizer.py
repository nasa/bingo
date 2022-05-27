import numpy as np
from scipy import optimize

from .optimizer import OptimizerBase
from ..evaluation.gradient_mixin import GradientMixin, VectorGradientMixin

ROOT_SET = {
    # 'hybr',
    'lm'
    # 'broyden1',
    # 'broyden2',
    # 'anderson',
    # 'linearmixing',
    # 'diagbroyden',
    # 'excitingmixing',
    # 'krylov',
    # 'df-sane'
}

MINIMIZE_SET = {
    'Nelder-Mead',
    'Powell',
    'CG',
    'BFGS',
    # 'Newton-CG',
    'L-BFGS-B',
    'TNC',
    # 'COBYLA',
    'SLSQP'
    # 'trust-constr'
    # 'dogleg',
    # 'trust-ncg',
    # 'trust-exact',
    # 'trust-krylov'
}

JACOBIAN_SET = {
    'CG',
    'BFGS',
    # 'Newton-CG',
    'L-BFGS-B',
    'TNC',
    'SLSQP',
    # 'trust-constr'
    # 'dogleg',
    # 'trust-ncg',
    # 'trust-exact',
    # 'trust-krylov',
    # 'hybr',
    'lm'
}


class ScipyOptimizer(OptimizerBase):
    def __init__(self, objective_fn, **options):
        self.options = options
        self.objective_fn = objective_fn

    @property
    def objective_fn(self):
        return self._objective_fn

    @objective_fn.setter
    def objective_fn(self, obj_fn):
        self._gradient_capable = isinstance(obj_fn, VectorGradientMixin)
        self._jacobian_capable = isinstance(obj_fn, GradientMixin)
        self._objective_fn = obj_fn
        self._verify_objective_fn(obj_fn, self.options["method"])

    @staticmethod
    def _verify_objective_fn(objective_fn, method):
        if method in ROOT_SET and not hasattr(objective_fn,
                                              'evaluate_fitness_vector'):
            raise TypeError("{} requires VectorBasedFunction \
                            as a fitness function".format(method))

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, kwargs):
        self._options = kwargs

        # set default param init bounds to [-10000, 10000) if not included
        if "param_init_bounds" not in self._options.keys():
            self._options["param_init_bounds"] = [-10000, 10000]

        # set default method to BFGS if not included
        if "method" not in self._options.keys():
            self._options["method"] = "BFGS"
        self._verify_method(self._options["method"])

        # set default tol to 1e-6 if not included
        if "tol" not in self._options.keys():
            self._options["tol"] = 1e-6

        # scipy_options = normal options w/o param_init_bounds
        self._scipy_options = {k: v for k, v in self._options.items() if
                               k != "param_init_bounds"}

    @staticmethod
    def _verify_method(method):
        if method not in ROOT_SET and method not in MINIMIZE_SET:
            raise KeyError("{} is not a listed method".format(method))

    def __call__(self, individual):
        num_params = individual.get_number_local_optimization_params()
        c_0 = np.random.uniform(*self.options["param_init_bounds"], num_params)
        params = self._run_algorithm_for_optimization(
            self._sub_routine_for_fit_function, individual, c_0)
        individual.set_local_optimization_params(params)

    def _sub_routine_for_fit_function(self, params, individual):
        individual.set_local_optimization_params(params)

        if self.options["method"] in ROOT_SET:
            return self.objective_fn.evaluate_fitness_vector(individual)
        return self.objective_fn(individual)

    def _run_algorithm_for_optimization(self, sub_routine, individual, params):
        backend, jacobian = self._get_backend_and_jacobian()
        try:
            optimize_result = backend(
                sub_routine,
                params,
                args=individual,
                jac=jacobian,
                **self._scipy_options
            )

            return optimize_result.x
        except TypeError:  # issue with too many constants using root method
            old_method = self.options["method"]

            self.options["method"] = "BFGS"  # use minimize method instead
            self._scipy_options["method"] = "BFGS"
            backend, jacobian = self._get_backend_and_jacobian()

            optimize_result = backend(
                sub_routine,
                params,
                args=individual,
                jac=jacobian,
                **self._scipy_options
            )

            self.options["method"] = old_method  # reset to old method
            self._scipy_options["method"] = old_method
            return optimize_result.x

    def _get_backend_and_jacobian(self):
        backend = optimize.minimize
        jacobian = False

        jacobian_method = self.options["method"] in JACOBIAN_SET

        if self.options["method"] in ROOT_SET:
            backend = optimize.root
            if jacobian_method and self._jacobian_capable:
                jacobian = lambda x, indv: \
                    self.objective_fn.get_fitness_vector_and_jacobian(indv)[1]

        else:  # MINIMIZE_SET
            if jacobian_method and self._gradient_capable:
                jacobian = lambda x, indv: \
                    self.objective_fn.get_fitness_and_gradient(indv)[1]

        return backend, jacobian
