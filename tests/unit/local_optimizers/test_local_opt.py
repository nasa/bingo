from bingo.local_optimizers.local_opt_fitness \
    import LocalOptFitnessFunction
from bingo.chromosomes.chromosome import Chromosome


class DummyLocalOptimizationIndividual(Chromosome):
    def __init__(self):
        super().__init__()
        self._params = [1, 2, 3]
        self._needs_opt = True

    def needs_local_optimization(self):
        return self._needs_opt

    def get_params(self):
        return self._params

    def get_number_local_optimization_params(self):
        return len(self._params)

    def __str__(self):
        pass

    def distance(self, chromosome):
        pass

    def set_local_optimization_params(self, params):
        self._params = params
        self._needs_opt = False


def test_get_eval_count_pass_through(mocker):
    fitness_function = mocker.Mock()
    fitness_function.eval_count = 123
    optimizer = mocker.Mock()
    local_opt_fitness_function = \
        LocalOptFitnessFunction(fitness_function, optimizer)
    assert local_opt_fitness_function.eval_count == 123


def test_set_eval_count_pass_through(mocker):
    fitness_function = mocker.Mock()
    optimizer = mocker.Mock()
    local_opt_fitness_function = \
        LocalOptFitnessFunction(fitness_function, optimizer)
    local_opt_fitness_function.eval_count = 123
    assert fitness_function.eval_count == 123


def test_get_training_data_pass_through(mocker):
    fitness_function = mocker.Mock()
    fitness_function.training_data = 123
    optimizer = mocker.Mock()
    local_opt_fitness_function = \
        LocalOptFitnessFunction(fitness_function, optimizer)
    assert local_opt_fitness_function.training_data == 123


def test_set_training_data_pass_through(mocker):
    fitness_function = mocker.Mock()
    optimizer = mocker.Mock()
    local_opt_fitness_function = \
        LocalOptFitnessFunction(fitness_function, optimizer)
    local_opt_fitness_function.training_data = 123
    assert fitness_function.training_data == 123


def test_get_and_set_optimizer(mocker):
    fitness_function = mocker.Mock()
    opt_1 = mocker.Mock()
    opt_2 = mocker.Mock()
    local_opt_fitness_function = \
        LocalOptFitnessFunction(fitness_function, opt_1)
    assert local_opt_fitness_function.optimizer == opt_1

    local_opt_fitness_function.optimizer = opt_2
    assert local_opt_fitness_function.optimizer == opt_2


def test_call_optimizes_when_necessary(mocker):
    fitness_function = mocker.Mock(
        side_effect=lambda ind: sum(ind.get_params())
    )
    optimizer = mocker.Mock(
        side_effect=lambda ind: ind.set_local_optimization_params([4, 5, 6])
    )

    individual = DummyLocalOptimizationIndividual()

    local_opt_fitness_function = \
        LocalOptFitnessFunction(fitness_function, optimizer)

    returned_fitness = local_opt_fitness_function(individual)

    # make sure optimizer was called with individual
    optimizer.assert_called_once_with(individual)
    assert individual.get_params() == [4, 5, 6]

    # make sure that fitness function was called with individual
    fitness_function.assert_called_once_with(individual)
    assert returned_fitness == fitness_function(individual)


def test_call_doesnt_optimize_when_not_needed(mocker):
    fitness_function = mocker.Mock(
        side_effect=lambda ind: sum(ind.get_params())
    )
    optimizer = mocker.Mock(
        side_effect=lambda ind: ind.set_local_optimization_params([4, 5, 6])
    )

    individual = DummyLocalOptimizationIndividual()
    initial_params = [1, 2, 3]
    individual.set_local_optimization_params(initial_params)

    local_opt_fitness_function = \
        LocalOptFitnessFunction(fitness_function, optimizer)

    returned_fitness = local_opt_fitness_function(individual)

    assert not optimizer.called
    assert individual.get_params() == initial_params

    # make sure that fitness function was called with individual
    fitness_function.assert_called_once_with(individual)
    assert returned_fitness == fitness_function(individual)
