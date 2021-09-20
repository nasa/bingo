import numpy as np
import time
import sys
import cupy as cp

from bingo.local_optimizers.continuous_local_opt \
    import ContinuousLocalOptimization
from bingo.symbolic_regression.explicit_regression \
    import ExplicitRegression, ExplicitTrainingData
from bingo.symbolic_regression \
    import AGraphGenerator, ComponentGenerator, AGraphCrossover, AGraphMutation

from bingo.variation.var_and import VarAnd
from bingo.evaluation.evaluation import Evaluation
from bingo.evolutionary_algorithms.mu_plus_lambda import MuPlusLambda
from bingo.evolutionary_optimizers.island import Island
from bingo.stats.hall_of_fame import HallOfFame

from smcbingo.bayes_fitness_function import BayesFitnessFunction
from smcbingo.nondeterministic_crowding import NondeterministicCrowding

import bingo.util.global_imports as bingo_gi
import smcpy.utils.global_imports as smc_gi


def print_best_individuals(island):
    norm_mlls = np.array([-i.fitness for i in island.population])
    order = np.argsort(-1*norm_mlls)
    for i, o in enumerate(order):
        print(f"{i + 1}\t{norm_mlls[o]}\t{island.population[o]}")
        if i >= 4:
            break


def run_benchmark(mcmc_steps, num_particles, phi_exponent, smc_steps,
                  stack_size):
    # AGRAPH PARAMS
    USE_SIMPLIFICATION = True
    OPERATORS = ['+', '-', '*', 'sin', 'cos']
    # # VARIATION PARAMS
    # CROSSOVER_PROB = 0.4
    # MUTATION_PROB = 0.4
    # # SELECTION PARAMS
    # SELECTION = NondeterministicCrowding()
    # SEED
    np.random.seed(0)
    # DATA
    X_DATA = np.random.uniform(1, 10, size=(150, 3))
    Y_DATA = X_DATA[:, 0] ** 2 * np.sin(X_DATA[:, 1]) + 5 * np.random.normal(5,
                                                                             2)
    TRAINING_DATA = ExplicitTrainingData(X_DATA, Y_DATA)
    # getting things set up
    equ, bff = create_graph_and_bff(mcmc_steps, num_particles, OPERATORS,
                                    phi_exponent, smc_steps, stack_size,
                                    USE_SIMPLIFICATION, TRAINING_DATA)
    # running evolution
    start_time = time.time()
    equ.fitness = bff(equ)
    elapsed_time = time.time() - start_time
    print(equ)
    print(f"\nElapsed time: {elapsed_time}")


def create_graph_and_bff(mcmc_steps, num_particles, operators, phi_exponent,
                         smc_steps, stack_size, use_simplification,
                         train_data):
    # generation
    component_generator = \
        ComponentGenerator(input_x_dimension=train_data.x.shape[1])
    for comp in operators:
        component_generator.add_operator(comp)
    generator = AGraphGenerator(stack_size, component_generator,
                                use_python=True,
                                use_simplification=use_simplification)

    # evaluation
    reg = ExplicitRegression(train_data)
    clo = ContinuousLocalOptimization(reg, algorithm="lm")
    bff = BayesFitnessFunction(clo,
                               num_particles=num_particles,
                               phi_exponent=phi_exponent,
                               smc_steps=smc_steps,
                               mcmc_steps=mcmc_steps)

    agraph = generator()
    while agraph.get_number_local_optimization_params() < 2:
        agraph = generator()

    return agraph, bff


if __name__ == '__main__':

    # BINGO PARAMS
    POPULATION_SIZE = 64
    STACK_SIZE = 64
    NUM_GENERATIONS = 1
    # SMC PARAMS
    NUM_PARTICLES = 800
    SMC_STEPS = 40
    MCMC_STEPS = 10
    PHI_EXPONENT = 2


    # use gpu implementation for bingo
    bingo_gi.set_use_gpu(True)

    # use gpu implementation for smcpy
    smc_gi.set_use_gpu(True)

    run_benchmark(MCMC_STEPS, NUM_PARTICLES, PHI_EXPONENT, SMC_STEPS,
                  STACK_SIZE)





