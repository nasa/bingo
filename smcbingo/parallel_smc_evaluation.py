from bingo.evaluation.evaluation import Evaluation
import bingo.util.global_imports as gi
import smcpy.utils.global_imports as smc_gi

import numpy as np
import cupy as cp
import math

from scipy.stats import multivariate_normal as mvn
from scipy.stats import invgamma

from bingo.symbolic_regression import ExplicitTrainingData
from smcpy.mcmc.vector_mcmc import VectorMCMC
from smcpy.mcmc.vector_mcmc_kernel import VectorMCMCKernel
from smcpy import SMCSampler

from bingo.util.gpu.gpu_evaluation_kernel import _f_eval_gpu_kernel_parallel

import nvtx


class ParallelSMCEvaluation(Evaluation):

    def __init__(self, continuous_local_opt, num_particles=150, phi_exponent=8,
                 smc_steps=15, mcmc_steps=12, ess_threshold=0.75, std=None):

        if smc_steps <= 2:
            raise ValueError('smc_steps must be > 2')
        if phi_exponent <= 0:
            raise ValueError('phi_exponent must be > 0')

        self._num_particles = num_particles
        self._smc_steps = smc_steps
        self._mcmc_steps = mcmc_steps
        self._ess_threshold = ess_threshold
        self._std = std

        self._num_observations = len(continuous_local_opt.training_data.x)
        self._cont_local_opt = continuous_local_opt
        self._fbf_phi_idx, self._phi_seq = self._calc_phi_sequence(phi_exponent)
        self._eval_count = 0

        if smc_gi.USING_GPU:
            gi.set_use_gpu(True)
            self.training_data_gpu = ExplicitTrainingData(
                    gi.num_lib.asarray(self.training_data.x),
                    gi.num_lib.asarray(self.training_data.y))

        if std is not None:
            raise NotImplementedError

        if not smc_gi.USING_GPU:
            raise NotImplementedError

    def __call__(self, population):
        gi.set_use_gpu(False)
        max_constants = np.max([indv.get_number_local_optimization_params()
                                for indv in population])
        param_names = [f'p{i}' for i in range(max_constants)] + ['std_dev']

        proposals, pop_inds = self._make_proposals(population, max_constants)

        command_arrays = [cp.asarray(population[i]._simplified_command_array)
                          for i in pop_inds]

        gi.set_use_gpu(True)

        evaluate_model = self._get_simplified_eval_call(command_arrays)
        # vector_mcmc = VectorMCMC(evaluate_model,
        #                          self.training_data_gpu.y.flatten(),
        #                          log_like_args=None)
        #
        # mcmc_kernel = VectorMCMCKernel(vector_mcmc, param_order=param_names)
        # smc = SMCSampler(mcmc_kernel)
        #
        # marginal_log_likes = smc.sample(self._num_particles,
        #                                 self._mcmc_steps,
        #                                 self._phi_seq,
        #                                 self._ess_threshold,
        #                                 proposals)
        #
        # nmlls = -1 * (marginal_log_likes[:, -1] -
        #               marginal_log_likes[:, self._fbf_phi_idx])
        #
        # for i, fitness in zip(pop_inds, nmlls):
        #     population[i].fitness = fitness

    def _make_proposals(self, population, max_constants):
        pop_inds = []
        proposal_list = []
        for i, individual in enumerate(population):
            gi.set_use_gpu(False)
            individual = self.do_local_opt(individual)
            # try:
            proposal_list.append(
                    self._generate_proposal_samples(individual,
                                                    max_constants))
            pop_inds.append(i)
            # except (ValueError, np.linalg.LinAlgError) as e:
            #     print(f"excepting proposal for equ {i}")
            #     print(e)
            #     individual.fitness = np.nan
        return np.stack(proposal_list), pop_inds

    def do_local_opt(self, individual):
        individual._notify_modification()
        individual._needs_opt = True
        _ = self._cont_local_opt(individual)
        return individual

    def _generate_proposal_samples(self, individual, max_constants):
        num_constants = individual.get_number_local_optimization_params()
        samples = np.zeros((self._num_particles, max_constants + 1))

        cov, var_ols, ssqe = self._estimate_covariance(individual)
        if num_constants > 0:
            proposal = mvn(individual.constants, cov, allow_singular=True)
            samples[:, :num_constants] = \
                proposal.rvs(self._num_particles).reshape(self._num_particles,
                                                          -1)

        len_data = len(self.training_data.x)
        noise_proposal = invgamma((0.01 + len_data) / 2,
                                  scale=(0.01 * var_ols + ssqe) / 2)
        samples[:, -1] = noise_proposal.rvs(self._num_particles).reshape(-1, 1)
        return samples

    def _estimate_covariance(self, individual):
        num_params = individual.get_number_local_optimization_params()
        x = self.training_data.x
        f, f_deriv = individual.evaluate_equation_with_local_opt_gradient_at(x)
        ssqe = np.sum((self.training_data.y - f) ** 2)
        var_ols = ssqe / (len(f) - num_params)
        cov = var_ols * np.linalg.inv(f_deriv.T.dot(f_deriv))
        return cov, var_ols, ssqe

    def _get_simplified_eval_call(self, stacks):
        THREADS_PER_BLOCK = 256
        data = self.training_data_gpu.x
        data_size = len(data)
        max_stack_size = max([len(c) for c in stacks])
        num_equations = len(stacks)
        stack_sizes = cp.asarray(np.cumsum([0] + [len(s) for s in stacks]))
        combined_stacks = cp.vstack(stacks)
        with nvtx.annotate(message="buffer allocation", color="green"):
            buffer = cp.full((max_stack_size, num_equations,
                              self._num_particles, data_size),
                             np.inf)
        with nvtx.annotate(message="result allocation", color="green"):
            results = cp.full((num_equations, self._num_particles, data_size),
                              np.inf)
        blocks_per_grid = \
            math.ceil(data_size * self._num_particles * num_equations
                      / THREADS_PER_BLOCK)
        return lambda constants: self._evaluate_model_gpu(
                constants, combined_stacks, data, self._num_particles,
                data_size, num_equations, stack_sizes, buffer, results,
                blocks_per_grid, THREADS_PER_BLOCK)

    @staticmethod
    def _evaluate_model_gpu(constants, stacks, data, num_particles,
                            data_size, num_equations, stack_sizes, buffer,
                            results, blocks_per_grid, threads_per_block):
        with nvtx.annotate(message="parallel kernel", color="green"):
            _f_eval_gpu_kernel_parallel[blocks_per_grid, threads_per_block](
                    stacks, data, constants, num_particles,
                    data_size, num_equations, stack_sizes, buffer, results)
        return results

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


