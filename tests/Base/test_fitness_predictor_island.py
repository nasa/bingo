# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest
import numpy as np

from bingo.Base.MultipleValues import SinglePointCrossover, SinglePointMutation, \
                                 MultipleValueChromosomeGenerator
from bingo.Base.FitnessPredictorIsland import FitnessPredictorIsland as FPI
from bingo.Base import FitnessPredictorIsland
from bingo.Base.MuPlusLambdaEA import MuPlusLambda
from bingo.Base.TournamentSelection import Tournament
from bingo.Base.Evaluation import Evaluation
from bingo.Base.FitnessFunction import FitnessFunction


MAIN_POPULATION_SIZE = 40
PREDICTOR_POPULATION_SIZE = 4
TRAINER_POPULATION_SIZE = 4
SUBSET_TRAINING_DATA_SIZE = 2
FULL_TRAINING_DATA_SIZE = 20


class DistanceToAverage(FitnessFunction):
    def __call__(self, individual):
        self.eval_count += 1
        avg_data = np.mean(self.training_data)
        return np.linalg.norm(individual.values - avg_data)


@pytest.fixture
def ev_alg():
    crossover = SinglePointCrossover()
    mutation = SinglePointMutation(np.random.random)
    selection = Tournament(2)
    training_data = np.linspace(0.1, 1, FULL_TRAINING_DATA_SIZE)
    fitness = DistanceToAverage(training_data)
    evaluator = Evaluation(fitness)
    return MuPlusLambda(evaluator, selection, crossover, mutation,
                        0., 1.0, MAIN_POPULATION_SIZE)


@pytest.fixture
def generator():
    return MultipleValueChromosomeGenerator(np.random.random, 10)


@pytest.fixture
def fitness_predictor_island(ev_alg, generator):
    island = FPI(ev_alg, generator, MAIN_POPULATION_SIZE,
        predictor_population_size=PREDICTOR_POPULATION_SIZE,
        trainer_population_size=TRAINER_POPULATION_SIZE,
        predictor_size_ratio=SUBSET_TRAINING_DATA_SIZE/FULL_TRAINING_DATA_SIZE,
        predictor_computation_ratio=0.4,
        trainer_update_frequency=4,
        predictor_update_frequency=5)
    island._predictor_island._ea.variation._mutation_probability = 1.0
    return island


@pytest.mark.parametrize("param, illegal_value",
                         [("predictor_population_size", -1),
                          ("predictor_update_frequency", 0),
                          ("predictor_size_ratio", 0),
                          ("predictor_size_ratio", 1.2),
                          ("predictor_computation_ratio", -0.2),
                          ("predictor_computation_ratio", 1),
                          ("trainer_population_size", -1),
                          ("trainer_update_frequency", 0)])
def test_raises_error_on_illegal_value_in_init(ev_alg, generator, param,
                                               illegal_value):
    kwargs = {param: illegal_value}
    with pytest.raises(ValueError):
        _ = FPI(ev_alg, generator, 10, **kwargs)


def test_predictor_compute_ratios(fitness_predictor_island):
    # init
    point_evals_predictor = FULL_TRAINING_DATA_SIZE*TRAINER_POPULATION_SIZE
    point_evals_predictor += 2 * point_evals_per_predictor_step()
    point_evals_main = 0
    assert_expected_compute_ratio(fitness_predictor_island,
                                  point_evals_main, point_evals_predictor)

    # main step
    point_evals_main += point_evals_per_main_step()
    fitness_predictor_island.execute_generational_step()
    point_evals_main += point_evals_per_main_step()
    assert_expected_compute_ratio(fitness_predictor_island,
                                  point_evals_main, point_evals_predictor)

    # main + predictor
    fitness_predictor_island.execute_generational_step()
    point_evals_main += point_evals_per_main_step()
    point_evals_predictor += point_evals_per_predictor_step()
    assert_expected_compute_ratio(fitness_predictor_island,
                                  point_evals_main, point_evals_predictor)

    # main + 2 predictor
    fitness_predictor_island.execute_generational_step()
    point_evals_main += point_evals_per_main_step()
    point_evals_predictor += 2 * point_evals_per_predictor_step()
    assert_expected_compute_ratio(fitness_predictor_island,
                                  point_evals_main, point_evals_predictor)

    # main + predictor + trainer update
    fitness_predictor_island.execute_generational_step()
    point_evals_main += point_evals_per_main_step()
    point_evals_predictor += point_evals_per_predictor_step()
    point_evals_predictor += point_evals_per_trainer_update()
    assert_expected_compute_ratio(fitness_predictor_island,
                                  point_evals_main, point_evals_predictor)

    # main + predictor update
    fitness_predictor_island.execute_generational_step()
    point_evals_main += point_evals_per_main_step()
    point_evals_main += point_evals_per_predictor_update()
    assert_expected_compute_ratio(fitness_predictor_island,
                                  point_evals_main, point_evals_predictor)


def test_fitness_predictor_island_ages(fitness_predictor_island):
    predictor_age = 1
    main_age = 0
    assert fitness_predictor_island.generational_age == main_age
    assert fitness_predictor_island._predictor_island.generational_age \
        == predictor_age

    fitness_predictor_island.execute_generational_step()
    main_age += 1
    assert fitness_predictor_island.generational_age == main_age
    assert fitness_predictor_island._predictor_island.generational_age \
        == predictor_age

    fitness_predictor_island.execute_generational_step()
    main_age += 1
    predictor_age += 1
    assert fitness_predictor_island.generational_age == main_age
    assert fitness_predictor_island._predictor_island.generational_age \
        == predictor_age

    fitness_predictor_island.execute_generational_step()
    main_age += 1
    predictor_age += 2
    assert fitness_predictor_island.generational_age == main_age
    assert fitness_predictor_island._predictor_island.generational_age \
        == predictor_age

    fitness_predictor_island.execute_generational_step()
    main_age += 1
    predictor_age += 1
    assert fitness_predictor_island.generational_age == main_age
    assert fitness_predictor_island._predictor_island.generational_age \
        == predictor_age


def test_nan_on_predicted_variance_of_trainer(mocker,
                                              fitness_predictor_island):
    mocker.patch('bingo.Base.FitnessPredictorIsland.np.var')
    FitnessPredictorIsland.np.var.side_effect = OverflowError

    island = fitness_predictor_island
    trainer = island.population[0]
    variance = island._calculate_predictor_variance_of(trainer)
    assert np.isnan(variance)


def assert_expected_compute_ratio(fitness_predictor_island, point_evals_main,
                                  point_evals_predictor):
    ratio_after_init = \
        fitness_predictor_island._get_predictor_computation_ratio()
    np.testing.assert_almost_equal(ratio_after_init,
                                   point_evals_predictor /
                                   (point_evals_predictor + point_evals_main))


def point_evals_per_predictor_step():
    return SUBSET_TRAINING_DATA_SIZE * PREDICTOR_POPULATION_SIZE \
           * TRAINER_POPULATION_SIZE


def point_evals_per_main_step():
    return SUBSET_TRAINING_DATA_SIZE * MAIN_POPULATION_SIZE


def point_evals_per_trainer_update():
    return SUBSET_TRAINING_DATA_SIZE * MAIN_POPULATION_SIZE * \
           PREDICTOR_POPULATION_SIZE + FULL_TRAINING_DATA_SIZE + \
           point_evals_per_predictor_step()


def point_evals_per_predictor_update():
    return point_evals_per_main_step()
