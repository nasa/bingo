# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import inspect

import numpy as np
import pytest

from bingo.evaluation.evaluation import Evaluation
from bingo.symbolic_regression import (
    ExplicitTrainingData,
    ExplicitRegression,
    AGraphCrossover,
    AGraphMutation,
    ComponentGenerator,
)
from bingo.evolutionary_algorithms.age_fitness import AgeFitnessEA
from bingo.evolutionary_optimizers.fitness_predictor_island import (
    FitnessPredictorIsland,
)
from bingo.evolutionary_optimizers.island import Island
from bingo.symbolic_regression.symbolic_regressor import (
    SymbolicRegressor,
    TIME_REDUCTION_FACTOR,
    DEFAULT_OPERATORS,
    SUPPORTED_EA_STRS,
    BEST_POP_MAX,
)
from bingo.symbolic_regression.agraph.agraph import AGraph
from bingo.symbolic_regression.agraph.generator import AGraphGenerator


# NOTE (David Randall): I'm testing a lot of private methods here. Normally
# I would avoid doing this, but to make the tests more manageable, I am doing
# so to break down the larger function tests (e.g., fit).


def get_sym_reg_import(class_or_method_name):
    return "bingo.symbolic_regression.symbolic_regressor." + class_or_method_name


def patch_import(mocker, class_name):
    return mocker.patch(get_sym_reg_import(class_name))


def patch_regr_method(mocker, method_name):
    return mocker.patch(get_sym_reg_import(f"SymbolicRegressor.{method_name}"))


def constructor_attrs():
    params = inspect.signature(SymbolicRegressor.__init__).parameters.keys()
    params = [param for param in params if param != "self"]
    return params


@pytest.mark.parametrize("attribute", constructor_attrs())
def test_init_sets_attributes(mocker, attribute):
    if attribute == "max_time":
        attribute_value = 1
        expected_value = TIME_REDUCTION_FACTOR
    else:
        attribute_value = mocker.Mock()
        expected_value = attribute_value

    regr = SymbolicRegressor(**{attribute: attribute_value})

    assert getattr(regr, attribute) == expected_value


@pytest.fixture
def basic_data():
    X = np.linspace(-10, 10, num=100).reshape((-1, 1))
    y = np.sin(X) + 3.0
    return X, y


@pytest.fixture
def sample_comp_gen(basic_data):
    X, _ = basic_data
    comp_gen = ComponentGenerator(X.shape[1])
    comp_gen.add_operator("+")
    comp_gen.add_operator("sin")
    return comp_gen


@pytest.fixture
def sample_agraph_gen(sample_comp_gen):
    return AGraphGenerator(16, sample_comp_gen)


@pytest.fixture
def sample_ea(basic_data, sample_comp_gen, sample_agraph_gen):
    fitness = ExplicitRegression(ExplicitTrainingData(*basic_data))
    evaluation = Evaluation(fitness)
    crossover = AGraphCrossover()
    mutation = AGraphMutation(sample_comp_gen)

    evo_alg = AgeFitnessEA(
        evaluation,
        sample_agraph_gen,
        crossover,
        mutation,
        crossover_probability=0.4,
        mutation_probability=0.4,
        population_size=100,
    )
    return evo_alg


@pytest.mark.parametrize("clo_alg", ["BFGS", "lm", "SLSQP"])
def test_get_local_opt(mocker, basic_data, clo_alg):
    X, y = basic_data
    tol = mocker.Mock()

    regr = SymbolicRegressor()
    regr.clo_alg = clo_alg
    regr.clo_threshold = tol

    local_opt = regr._get_local_opt(X, y, tol)

    np.testing.assert_array_equal(local_opt._fitness_function.training_data.x, X)
    np.testing.assert_array_equal(local_opt._fitness_function.training_data.y, y)
    assert local_opt.optimizer.options["method"] == clo_alg
    assert local_opt.optimizer.options["tol"] == tol


# TODO test predictor_size_ratio?
@pytest.mark.parametrize(
    "dataset_size, expected_island",
    [
        (100, Island),
        (1199, Island),
        (1200, FitnessPredictorIsland),
        (1500, FitnessPredictorIsland),
    ],
)
def test_make_island(
    mocker, sample_ea, sample_agraph_gen, dataset_size, expected_island
):
    pop_size = 100
    hof = mocker.Mock()

    regr = SymbolicRegressor()
    regr.generator = sample_agraph_gen
    regr.population_size = pop_size

    island = regr._make_island(dataset_size, sample_ea, hof)

    assert isinstance(island, expected_island)
    assert island._ea == sample_ea
    assert island._generator == sample_agraph_gen
    assert island._population_size == pop_size
    assert island.hall_of_fame == hof


def test_force_diversity_in_island(sample_ea, sample_agraph_gen):
    np.random.seed(0)
    pop_size = 100
    island = Island(sample_ea, sample_agraph_gen, pop_size)
    regr = SymbolicRegressor()
    regr.population_size = pop_size
    regr.generator = sample_agraph_gen

    regr._force_diversity_in_island(island)

    unique_indvs = set(str(indv) for indv in island.population)
    assert len(unique_indvs) == pop_size
    assert len(island.population) == pop_size


@pytest.fixture
def sample_agraph():
    equation = AGraph()
    equation.command_array = np.array([[0, 0, 0]], dtype=int)
    return equation


@pytest.mark.timeout(3)
def test_cant_force_diversity_in_island(mocker, sample_ea, sample_agraph):
    np.random.seed(0)
    pop_size = 100
    agraph_gen = mocker.Mock(return_value=sample_agraph)
    island = Island(sample_ea, agraph_gen, pop_size)
    regr = SymbolicRegressor()
    regr.population_size = pop_size
    regr.generator = agraph_gen

    regr._force_diversity_in_island(island)

    unique_indvs = set(str(indv) for indv in island.population)
    assert len(unique_indvs) == 1
    assert len(island.population) == pop_size


def patch_all_get_archipelago(mocker):
    imports = [
        "ComponentGenerator",
        "AGraphCrossover",
        "AGraphMutation",
        "AGraphGenerator",
        "Evaluation",
        "AgeFitnessEA",
        "GeneralizedCrowdingEA",
        "ParetoFront",
    ]
    fns = ["_get_local_opt", "_make_island", "_force_diversity_in_island"]

    patched_imports = {name: patch_import(mocker, name) for name in imports}
    patched_fns = {name: patch_regr_method(mocker, name) for name in fns}

    patched_imports.update(patched_fns)

    return patched_imports


def get_mocked_data(mocker):
    len = 250
    X = mocker.MagicMock()
    X.__len__.return_value = len
    X.shape = (len, mocker.Mock())

    y = mocker.MagicMock()
    y.__len__.return_value = len
    y.shape = (len, mocker.Mock())

    return X, y


@pytest.mark.parametrize("operators", [None, "mock"])
def test_get_archipelago_component_gen(mocker, operators):
    patched_objs = patch_all_get_archipelago(mocker)
    comp_gen = patched_objs["ComponentGenerator"]
    if operators == "mock":
        operators = [mocker.Mock() for _ in range(10)]

    regr = SymbolicRegressor(operators=operators)
    X, y = get_mocked_data(mocker)
    n_proc = mocker.Mock()

    regr._get_archipelago(X, y, n_proc)

    comp_gen.assert_called_once_with(X.shape[1])
    assert regr.component_generator == comp_gen.return_value

    if operators is None:
        operators = DEFAULT_OPERATORS
    for operator in operators:
        comp_gen.return_value.add_operator.assert_any_call(operator)


def test_get_archipelago_agraph_crossover(mocker):
    patched_objs = patch_all_get_archipelago(mocker)
    crossover = patched_objs["AGraphCrossover"]

    regr = SymbolicRegressor()
    X, y = get_mocked_data(mocker)
    n_proc = mocker.Mock()

    regr._get_archipelago(X, y, n_proc)

    crossover.assert_called_once()
    assert regr.crossover == crossover.return_value


def test_get_archipelago_agraph_mutation(mocker):
    patched_objs = patch_all_get_archipelago(mocker)
    mutation = patched_objs["AGraphMutation"]
    comp_gen = patched_objs["ComponentGenerator"]

    regr = SymbolicRegressor()
    X, y = get_mocked_data(mocker)
    n_proc = mocker.Mock()

    regr._get_archipelago(X, y, n_proc)

    mutation.assert_called_once_with(comp_gen.return_value)
    assert regr.mutation == mutation.return_value


def test_get_archipelago_agraph_generator(mocker):
    patched_objs = patch_all_get_archipelago(mocker)
    generator = patched_objs["AGraphGenerator"]
    comp_gen = patched_objs["ComponentGenerator"]

    stack_size = mocker.Mock()
    use_simplification = mocker.Mock()
    regr = SymbolicRegressor(
        stack_size=stack_size, use_simplification=use_simplification
    )
    X, y = get_mocked_data(mocker)
    n_proc = mocker.Mock()

    regr._get_archipelago(X, y, n_proc)

    generator.assert_called_once_with(
        stack_size,
        comp_gen.return_value,
        use_simplification=use_simplification,
        use_python=True,
    )
    assert regr.generator == generator.return_value


def test_get_archipelago_evaluation(mocker):
    patched_objs = patch_all_get_archipelago(mocker)
    age_fitness = mocker.patch("bingo.evolutionary_algorithms.age_fitness")
    age_fitness.AgeFitnessEA = mocker.Mock()
    get_local_opt = patched_objs["_get_local_opt"]
    evaluation = patched_objs["Evaluation"]

    clo_tol = mocker.Mock()
    regr = SymbolicRegressor(clo_threshold=clo_tol)
    X, y = get_mocked_data(mocker)
    n_proc = mocker.Mock()

    regr._get_archipelago(X, y, n_proc)

    get_local_opt.assert_called_once_with(X, y, clo_tol)
    evaluation.assert_called_once_with(get_local_opt.return_value, multiprocess=n_proc)


@pytest.mark.parametrize("evo_alg_str", SUPPORTED_EA_STRS)
def test_get_archipelago_evo_alg_supported(mocker, evo_alg_str):
    patched_objs = patch_all_get_archipelago(mocker)
    evo_alg = patched_objs[evo_alg_str]
    evaluation = patched_objs["Evaluation"].return_value
    generator = patched_objs["AGraphGenerator"].return_value
    crossover = patched_objs["AGraphCrossover"].return_value
    mutation = patched_objs["AGraphMutation"].return_value

    cross_prob = mocker.Mock()
    mutation_prob = mocker.Mock()
    pop_size = mocker.Mock()
    regr = SymbolicRegressor(
        crossover_prob=cross_prob,
        mutation_prob=mutation_prob,
        population_size=pop_size,
        evolutionary_algorithm=evo_alg,
    )
    X, y = get_mocked_data(mocker)
    n_proc = mocker.Mock()

    regr._get_archipelago(X, y, n_proc)

    if evo_alg_str == "AgeFitnessEA":
        evo_alg.assert_called_once_with(
            evaluation,
            generator,
            crossover,
            mutation,
            cross_prob,
            mutation_prob,
            pop_size,
        )
    else:
        evo_alg.assert_called_once_with(
            evaluation, crossover, mutation, cross_prob, mutation_prob
        )


def test_get_archipelago_evo_alg_unsupported(mocker):
    patch_all_get_archipelago(mocker)
    regr = SymbolicRegressor(evolutionary_algorithm=mocker.Mock())
    X, y = get_mocked_data(mocker)
    n_proc = mocker.Mock()

    with pytest.raises(TypeError) as exc_info:
        regr._get_archipelago(X, y, n_proc)
    assert "is an unsupported evolutionary algorithm" in str(exc_info.value)


@pytest.mark.parametrize("evo_alg_str", SUPPORTED_EA_STRS)
def test_get_archipelago_arch_and_return(mocker, evo_alg_str):
    patched_objs = patch_all_get_archipelago(mocker)
    evo_alg = patched_objs[evo_alg_str]
    hof = patched_objs["ParetoFront"]
    make_island = patched_objs["_make_island"]
    force_diversity = patched_objs["_force_diversity_in_island"]

    regr = SymbolicRegressor(evolutionary_algorithm=evo_alg)
    X, y = get_mocked_data(mocker)
    n_proc = mocker.Mock()

    arch = regr._get_archipelago(X, y, n_proc)

    make_island.assert_called_once()
    expected_args = [
        (len(X), evo_alg.return_value, hof.return_value),
        (X.shape[0], evo_alg.return_value, hof.return_value),
    ]
    assert make_island.call_args.args in expected_args

    force_diversity.assert_called_once_with(make_island.return_value)

    assert arch == make_island.return_value


def patch_all_fit(mocker):
    imports = ["np.random.seed", "random.seed", "EquationRegressor"]
    fns = ["_get_archipelago"]

    patched_imports = {name: patch_import(mocker, name) for name in imports}
    patched_fns = {name: patch_regr_method(mocker, name) for name in fns}

    patched_imports.update(patched_fns)

    return patched_imports


@pytest.mark.parametrize("seed", [None, "mock"])
def test_fit_seed(mocker, seed):
    patched_objs = patch_all_fit(mocker)
    np_seed = patched_objs["np.random.seed"]
    random_seed = patched_objs["random.seed"]
    find_best_pop = patch_regr_method(mocker, "_find_best_population")
    find_best_pop.return_value = [mocker.Mock()]

    if seed == "mock":
        seed = mocker.Mock()

    regr = SymbolicRegressor(random_state=seed)
    X, y = get_mocked_data(mocker)

    regr.fit(X, y)

    if seed is None:
        np_seed.assert_not_called()
        random_seed.assert_not_called()
    else:
        np_seed.assert_called_once_with(seed)
        random_seed.assert_called_once_with(seed)


@pytest.mark.parametrize("n_cpus", [None, 0, 1, 2, 3])
def test_fit_archipelago(mocker, n_cpus):
    patched_objs = patch_all_fit(mocker)
    get_archipelago = patched_objs["_get_archipelago"]
    find_best_pop = patch_regr_method(mocker, "_find_best_population")
    find_best_pop.return_value = [mocker.Mock()]
    evolve = get_archipelago.return_value.evolve_until_convergence

    if n_cpus is None:
        n_cpus = 0
        environ_dict = {}
    else:
        environ_dict = {"OMP_NUM_THREADS": str(n_cpus)}
    mocker.patch.dict(get_sym_reg_import("os.environ"), environ_dict, clear=True)

    max_gens = mocker.Mock()
    fit_threshold = mocker.Mock()
    max_time = 1
    regr = SymbolicRegressor(
        generations=max_gens, fitness_threshold=fit_threshold, max_time=max_time
    )
    X, y = get_mocked_data(mocker)

    regr.fit(X, y)

    get_archipelago.assert_called_once_with(X, y, n_cpus)
    assert regr.archipelago == get_archipelago.return_value

    evolve.assert_called_once()
    assert evolve.call_args.kwargs["max_generations"] == max_gens
    assert evolve.call_args.kwargs["fitness_threshold"] == fit_threshold
    assert evolve.call_args.kwargs["max_time"] == TIME_REDUCTION_FACTOR


@pytest.mark.parametrize("hof_size", [0, 2, BEST_POP_MAX + 1])
@pytest.mark.parametrize("pop_size", [1, BEST_POP_MAX, BEST_POP_MAX - 1])
def test_get_best_population(mocker, hof_size, pop_size):
    patched_objs = patch_all_fit(mocker)
    archipelago = patched_objs["_get_archipelago"].return_value
    archipelago.hall_of_fame = [mocker.Mock()] * hof_size
    archipelago.population = [mocker.Mock()] * pop_size
    eq_regressor_constructor = patched_objs["EquationRegressor"]
    eq_regressor = eq_regressor_constructor.return_value
    eq_regressor.fitness = 1

    regr = SymbolicRegressor()
    X, y = get_mocked_data(mocker)

    regr.fit(X, y)

    if hof_size == 0:
        archipelago.update_hall_of_fame.assert_called_once()

    expected_pop_size = min(hof_size + pop_size, BEST_POP_MAX)
    assert eq_regressor_constructor.call_count == expected_pop_size
    assert eq_regressor.fit.call_count == expected_pop_size
    assert regr.get_best_population() == [eq_regressor] * expected_pop_size

    assert regr.get_best_individual() == eq_regressor
    regr.get_best_individual().fit.assert_called()


def test_get_best_population_not_set():
    regr = SymbolicRegressor()
    with pytest.raises(ValueError) as exc_info:
        regr.get_best_population()
    assert "Best population not set" in str(exc_info.value)


def test_fit_return(mocker):
    patched_objs = patch_all_fit(mocker)
    find_best_pop = patch_regr_method(mocker, "_find_best_population")
    find_best_pop.return_value = [mocker.Mock()]

    regr = SymbolicRegressor()
    X, y = get_mocked_data(mocker)

    returned_obj = regr.fit(X, y)

    assert returned_obj is regr


# TODO sample weight testing


def test_get_best_individual_normal(mocker):
    regr = SymbolicRegressor()
    best_ind = mocker.Mock()
    regr.best_ind = best_ind
    assert regr.get_best_individual() == best_ind


def test_get_best_individual_not_set():
    regr = SymbolicRegressor()

    with pytest.raises(ValueError) as exc_info:
        regr.get_best_individual()
    assert "Best individual not set" in str(exc_info.value)


def test_predict_normal(mocker):
    regr = SymbolicRegressor()
    best_ind = mocker.Mock()
    best_ind.predict = mocker.Mock(side_effect=lambda x: x)
    regr.best_ind = best_ind
    X = mocker.Mock()

    assert regr.predict(X) == X


@pytest.mark.parametrize("param", constructor_attrs())
def test_can_set_all_params(mocker, param):
    if param == "max_time":
        param_value = 1
        expected_value = TIME_REDUCTION_FACTOR
    else:
        param_value = mocker.Mock(spec=object)  # spec=object so get_params()
        # doesn't try to treat the mock like a dictionary
        expected_value = param_value

    regr = SymbolicRegressor()
    output = regr.set_params(**{param: param_value})

    assert regr.get_params()[param] == expected_value
    assert output is regr


def test_setting_param_doesnt_reset_others(mocker):
    regr = SymbolicRegressor()
    mocked_ea = mocker.Mock(spec=object)
    regr.set_params(evolutionary_algorithm=mocked_ea)
    regr.set_params(population_size=1)

    # setting population_size shouldn't reset
    # evolutionary_algorithm back to default
    assert regr.evolutionary_algorithm == mocked_ea
    assert regr.get_params()["evolutionary_algorithm"] == mocked_ea

    assert regr.population_size == 1
    assert regr.get_params()["population_size"] == 1


def test_setting_invalid_param_raises_value_error():
    regr = SymbolicRegressor()
    with pytest.raises(ValueError):
        regr.set_params(**{"not_a_valid_param": False})
