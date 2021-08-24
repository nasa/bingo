import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import cupy as cp

from tqdm import tqdm

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


def create_island(crossover_prob, mcmc_steps, mutation_prob, num_particles,
                  operators, phi_exponent, population_size, selection,
                  smc_steps, stack_size, use_simplification, train_data):
    # generation
    component_generator = \
        ComponentGenerator(input_x_dimension=train_data.x.shape[1])
    for comp in operators:
        component_generator.add_operator(comp)
    generator = AGraphGenerator(stack_size, component_generator,
                                use_python=True,
                                use_simplification=use_simplification)

    # variation
    crossover = AGraphCrossover()
    mutation = AGraphMutation(component_generator)
    variation_phase = VarAnd(crossover, mutation,
                             crossover_probability=crossover_prob,
                             mutation_probability=mutation_prob)

    # evaluation
    reg = ExplicitRegression(train_data)
    clo = ContinuousLocalOptimization(reg, algorithm="lm")
    bff = BayesFitnessFunction(clo,
                               num_particles=num_particles,
                               phi_exponent=phi_exponent,
                               smc_steps=smc_steps,
                               mcmc_steps=mcmc_steps)
    evaluation_phase = Evaluation(bff)

    # evolutionary algorithm
    ea = MuPlusLambda(evaluation_phase, selection, crossover, mutation,
                      crossover_prob, mutation_prob, population_size)
    ea.variation = variation_phase

    # island
    hof = HallOfFame(10)
    island = Island(ea, generator, population_size=population_size,
                    hall_of_fame=hof)
    return island


if __name__ == '__main__':
    '''
    Pop Size: 80 (12-160)
    Particles: 600 (300-2000)
    MCMC Steps: 22 (12-48) #if below 12, we consistently get an different error when finding LL
    SMC Steps:  22 (16- 40)
    Data points: 100/150 (50-15000) #the data isn't very noisy so 100/150 seems good enough 
    Phi: 6 (4-8)
    Selection: 10% - 20% of population size  is this tournament size?
    Stack Size: 64 (40-80)
    Components: +  -  *  /  ^  e  cos
    Error Metric: rmse and relative=False
    Simplification: False 
    '''

    if len(sys.argv) == 2:
        bingo_gi.set_use_gpu(True)
        smc_gi.set_use_gpu(True)
    elif len(sys.argv) == 3:
        bingo_gi.set_use_parallel_cpu(True)
    elif len(sys.argv) == 4:
        bingo_gi.set_use_gpu(True)
        smc_gi.set_use_gpu(True)
        bingo_gi.set_use_parallel_cpu(True)

    # ISLAND PARAMS
    POPULATION_SIZE = 256

    # AGRAPH PARAMS
    STACK_SIZE = 64
    USE_SIMPLIFICATION = True
    OPERATORS = ['+',  '-',  '*', 'sin', 'cos']

    # VARIATION PARAMS
    CROSSOVER_PROB = 0.4
    MUTATION_PROB = 0.4

    # EVOLUTION PARAMS
    # You can adjust NUM_GENERATIONS to whatever you want.
    # Run time should scale ~linearly wrt to this number.
    # Target values for this would be on the order of 50k, right now this
    # would take us ~20 days, which is why we hope to speed things up.
    NUM_GENERATIONS = 2

    # SELECTION PARAMS
    SELECTION = NondeterministicCrowding()

    # SMC PARAMS
    NUM_PARTICLES = 800
    SMC_STEPS = 40
    MCMC_STEPS = 10
    PHI_EXPONENT = 2

    # SEED
    np.random.seed(0)

    # DATA
    X_DATA = np.random.uniform(1, 10, size=(150, 3))
    Y_DATA = X_DATA[:, 0]**2*np.sin(X_DATA[:, 1]) + 5*np.random.normal(5, 2)
    TRAINING_DATA = ExplicitTrainingData(X_DATA, Y_DATA)


    # getting things set up
    island = create_island(CROSSOVER_PROB, MCMC_STEPS, MUTATION_PROB,
                           NUM_PARTICLES, OPERATORS, PHI_EXPONENT,
                           POPULATION_SIZE, SELECTION, SMC_STEPS, STACK_SIZE,
                           USE_SIMPLIFICATION, TRAINING_DATA)

    # running evolution
    start_time = time.time()
    for _ in tqdm(range(NUM_GENERATIONS)):
        island.evolve(num_generations=1)
    elapsed_time = time.time() - start_time

    print_best_individuals(island)
    print(f"\nElapsed time: {elapsed_time}")


