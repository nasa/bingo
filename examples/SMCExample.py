import numpy as np

from bingo.local_optimizers.scipy_optimizer import ScipyOptimizer
from bingo.local_optimizers.normalized_marginal_likelihood import (
    NormalizedMarginalLikelihood,
)
from bingo.symbolic_regression import (
    ExplicitRegression,
    ExplicitTrainingData,
    AGraph,
    AGraphGenerator,
    ComponentGenerator,
    AGraphCrossover,
    AGraphMutation,
)

from bingo.evolutionary_algorithms.generalized_crowding import (
    GeneralizedCrowdingEA,
)
from bingo.evaluation.evaluation import Evaluation
from bingo.evolutionary_optimizers.island import Island
from bingo.stats.hall_of_fame import HallOfFame

from bingo.selection.probabilistic_crowding import ProbabilisticCrowding

from bingo.util.log import configure_logging

configure_logging("detailed")


def _true_equation():
    true_commands = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 1], [6, 0, 0], [4, 1, 3], [2, 4, 2]]
    )
    true_constants = np.array([2.0, 3.0])

    true_equ = AGraph(use_simplification=False)
    true_equ.command_array = true_commands
    true_equ.set_local_optimization_params(true_constants)
    return true_equ


def get_training_data(equ, minx, maxx, numx, noise_ratio):
    x = np.linspace(minx, maxx, numx).reshape((-1, 1))
    y = equ.evaluate_equation_at(x)
    noise_std = np.mean(np.abs(y)) * noise_ratio
    y += np.random.normal(0, noise_std, y.shape)
    return ExplicitTrainingData(x, y)


def make_fitness_function(
    training_data,
    mcmc_steps,
    num_particles,
    num_multistarts,
):
    fitness = ExplicitRegression(training_data=training_data, metric="mse")

    scipy_opt = ScipyOptimizer(fitness, method="lm")
    nml = NormalizedMarginalLikelihood(
        fitness,
        scipy_opt,
        num_particles=num_particles,
        mcmc_steps=mcmc_steps,
        num_multistarts=num_multistarts,
    )

    return nml


def make_island(
    fitness_function,
    population_size,
    stack_size,
    operators,
    crossover_prob,
    mutation_prob,
):

    # generation
    component_generator = ComponentGenerator(
        input_x_dimension=fitness_function.training_data.x.shape[1]
    )
    for comp in operators:
        component_generator.add_operator(comp)
    generator = AGraphGenerator(
        stack_size,
        component_generator,
        use_simplification=True,
    )

    # variation
    crossover = AGraphCrossover()
    mutation = AGraphMutation(component_generator)

    # evaluation
    evaluation_phase = Evaluation(fitness_function, redundant=True)

    # selection
    selection_phase = ProbabilisticCrowding(log_scale=True)

    # evolutionary algorithm
    ea = GeneralizedCrowdingEA(
        evaluation_phase,
        crossover,
        mutation,
        crossover_prob,
        mutation_prob,
        selection=selection_phase,
    )

    # island
    hof = HallOfFame(
        10,
        similarity_function=lambda x, y: np.array_equal(
            x.command_array, y.command_array
        ),
    )
    island = Island(ea, generator, population_size=population_size, hall_of_fame=hof)
    return island


if __name__ == "__main__":

    np.random.seed(0)

    # DATA PARAMS
    MINX = 0
    MAXX = np.pi * 1.5
    NUMX = 25
    NOISE_RATIO = 0.15
    TRU_EQU = _true_equation()

    # BFF PARAMS
    NUM_PARTICLES = 200
    NUM_MCMC_STEPS = 10
    NUM_MULTISTARTS = 8

    # ISLAND PARAMS
    POPULATION_SIZE = 20

    # AGRAPH PARAMS
    OPERATORS = ["+", "-", "*"]
    STACK_SIZE = 16
    USE_SIMPLIFICATION = True

    # VARIATION PARAMS
    CROSSOVER_PROB = 0.4
    MUTATION_PROB = 0.4

    # EVOLUTION PARAMS
    NUM_GENERATIONS = 1

    np.random.seed(0)
    TRAINING_DATA = get_training_data(TRU_EQU, MINX, MAXX, NUMX, NOISE_RATIO)
    NML = make_fitness_function(
        TRAINING_DATA,
        NUM_MCMC_STEPS,
        NUM_PARTICLES,
        NUM_MULTISTARTS,
    )
    ISLAND = make_island(
        NML,
        POPULATION_SIZE,
        STACK_SIZE,
        OPERATORS,
        CROSSOVER_PROB,
        MUTATION_PROB,
    )

    for _ in range(NUM_GENERATIONS):
        ISLAND.evolve(1)

    print("HOF")
    print(ISLAND.hall_of_fame)
