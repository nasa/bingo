import numpy as np
import pytest
from bingo.evaluation.evaluation import Evaluation

from bingo.symbolic_regression import ExplicitTrainingData, ExplicitRegression, \
    AGraphCrossover, AGraphMutation, ComponentGenerator

from bingo.evolutionary_algorithms.age_fitness import AgeFitnessEA

from bingo.evolutionary_optimizers.fitness_predictor_island import \
    FitnessPredictorIsland

from bingo.evolutionary_optimizers.island import Island

from bingo.symbolic_regression.symbolic_regressor import SymbolicRegressor
from bingo.symbolic_regression.agraph.operator_definitions import OPERATOR_NAMES
from bingo.evolutionary_optimizers.evolutionary_optimizer import EvolutionaryOptimizer
from bingo.symbolic_regression.agraph.agraph import AGraph
from bingo.symbolic_regression.agraph.generator import AGraphGenerator

INF_REPLACEMENT = 1e100
DEFAULT_OPERATORS = {"+", "-", "*", "/"}
SUPPORTED_EA_STRS = ["AgeFitnessEA", "DeterministicCrowdingEA"]

# NOTE (David Randall): I'm testing a lot of private methods here. Normally
# I would avoid doing this, but to make the tests more manageable, I am doing
# so to break down the larger function tests (e.g., fit).


def get_sym_reg_import(class_or_method_name):
    return "bingo.symbolic_regression.symbolic_regressor." \
           + class_or_method_name

def patch_import(mocker, class_name):
    import_name = "bingo.symbolic_regression.symbolic_regressor." + class_name
    return mocker.patch(import_name)


def patch_regr_method(mocker, method_name):
    import_name = "bingo.symbolic_regression.symbolic_regressor." \
                  "SymbolicRegressor." + method_name
    return mocker.patch(import_name)


@pytest.fixture
def basic_data():
    X = np.linspace(-10, 10, num=100).reshape((-1, 1))
    y = np.sin(X) + 3.0
    return X, y


@pytest.mark.parametrize("invalid_size", [0, -1])
def test_population_size_invalid(invalid_size, basic_data):
    regr = SymbolicRegressor(population_size=invalid_size)
    X, y = basic_data

    with pytest.raises(ValueError):
        regr.fit(X, y)


def test_component_generator_gets_shape_and_ops(mocker, basic_data):
    mocker.patch(get_sym_reg_import("SymbolicRegressor._make_island"))
    mocker.patch(get_sym_reg_import(
        "SymbolicRegressor._force_diversity_in_island"))
    comp_gen = mocker.patch(get_sym_reg_import("ComponentGenerator"))

    mocked_operators = [mocker.Mock() for _ in range(5)]
    regr = SymbolicRegressor(operators=mocked_operators)
    X, y = basic_data

    regr._get_archipelago(X, y, 1)

    comp_gen.assert_called_with(X.shape[1])
    for mocked_operator in mocked_operators:
        comp_gen.return_value.add_operator.assert_any_call(mocked_operator)


def test_agraph_generator_gets_args(mocker, basic_data):
    agraph_gen = mocker.patch(get_sym_reg_import("AGraphGenerator"))
    comp_gen = mocker.patch(get_sym_reg_import("ComponentGenerator"))

    stack_size = mocker.Mock()
    use_simplification = mocker.Mock()

    regr = SymbolicRegressor(stack_size=stack_size,
                             use_simplification=use_simplification)
    X, y = basic_data
    regr._get_archipelago(X, y, 1)

    agraph_gen.assert_called_with(stack_size, comp_gen.return_value,
                                  use_simplification=use_simplification,
                                  use_python=True)


def test_agraph_mutation_gets_comp_gen(mocker, basic_data):
    mocker.patch(get_sym_reg_import("SymbolicRegressor._make_island"))
    mocker.patch(get_sym_reg_import(
        "SymbolicRegressor._force_diversity_in_island"))
    agraph_mutation = mocker.patch(get_sym_reg_import("AGraphMutation"))
    comp_gen = mocker.patch(get_sym_reg_import("ComponentGenerator"))

    regr = SymbolicRegressor()
    X, y = basic_data
    regr._get_archipelago(X, y, 1)

    agraph_mutation.assert_called_with(comp_gen.return_value)


def test_local_opt_and_fitness(mocker, basic_data):
    training_data = mocker.patch(get_sym_reg_import("ExplicitTrainingData"))
    fitness = mocker.patch(get_sym_reg_import("ExplicitRegression"))
    local_opt = mocker.patch(get_sym_reg_import("ContinuousLocalOptimization"))
    clo_alg = mocker.Mock()
    clo_threshold = mocker.Mock()
    metric = mocker.Mock()
    X, y = basic_data

    regr = SymbolicRegressor(metric=metric, clo_alg=clo_alg,
                             clo_threshold=clo_threshold)
    regr._get_archipelago(X, y, 1)

    # TODO can we test that training_data.x == x and fitness.training_data == training_data
    #   testing functionality vs. expected function calls
    training_data.assert_called_with(X, y)
    fitness.assert_called_with(training_data=training_data.return_value,
                               metric=metric)
    local_opt.assert_called_with(fitness.return_value, algorithm=clo_alg,
                                 tol=clo_threshold)


# TODO
def test_evaluation_gets_local_opt(mocker, basic_data):
    mocker.patch(get_sym_reg_import("SymbolicRegressor._make_island"))
    mocker.patch(get_sym_reg_import(
        "SymbolicRegressor._force_diversity_in_island"))
    local_opt = mocker.patch(
        get_sym_reg_import("ContinuousLocalOptimization")).return_value
    mocked_evaluation = mocker.patch(get_sym_reg_import("Evaluation"))
    n_proc = mocker.Mock()
    regr = SymbolicRegressor()
    X, y = basic_data

    regr._get_archipelago(X, y, n_proc)

    mocked_evaluation.assert_called_with(local_opt, multiprocess=n_proc)


@pytest.mark.parametrize("evo_alg_str", ["AgeFitnessEA",
                                         "DeterministicCrowdingEA"])
def test_evolutionary_alg_gets_args(mocker, basic_data, evo_alg_str):
    mocker.patch(get_sym_reg_import("SymbolicRegressor._make_island"))
    mocker.patch(get_sym_reg_import(
        "SymbolicRegressor._force_diversity_in_island"))
    evaluation = mocker.patch(get_sym_reg_import("Evaluation")).return_value
    agraph_gen = mocker.patch(
        get_sym_reg_import("AGraphGenerator")).return_value
    crossover = mocker.patch(get_sym_reg_import("AGraphCrossover")).return_value
    mutation = mocker.patch(get_sym_reg_import("AGraphMutation")).return_value
    crossover_prob = mocker.Mock()
    mutation_prob = mocker.Mock()
    population_size = mocker.Mock()
    mocked_ea = mocker.patch(get_sym_reg_import(evo_alg_str))
    regr = SymbolicRegressor(evolutionary_algorithm=mocked_ea,
                             crossover_prob=crossover_prob,
                             mutation_prob=mutation_prob,
                             population_size=population_size)
    X, y = basic_data

    regr._get_archipelago(X, y, 1)

    if evo_alg_str == "AgeFitnessEA":
        mocked_ea.assert_called_with(evaluation, agraph_gen, crossover,
                                     mutation, crossover_prob, mutation_prob,
                                     population_size)
    elif evo_alg_str == "DeterministicCrowding":
        mocked_ea.assert_called_with(evaluation, crossover, mutation,
                                     crossover_prob, mutation_prob)


# TODO test helper functions, more concrete
#   test _get_clo, _get_archipelago like normal fns then
#   test their combination

@pytest.fixture
def sample_comp_gen(basic_data):
    X, y = basic_data
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

    ea = AgeFitnessEA(evaluation, sample_agraph_gen, crossover, mutation,
                      crossover_probability=0.4, mutation_probability=0.4,
                      population_size=100)
    return ea


@pytest.mark.parametrize("clo_alg", ["BFGS", "lm", "SLSQP"])
def test_get_local_opt(mocker, basic_data, clo_alg):
    X, y = basic_data
    tol = mocker.Mock()

    regr = SymbolicRegressor()
    regr.clo_alg = clo_alg
    regr.clo_threshold = tol

    local_opt = regr._get_clo(X, y, tol)

    np.testing.assert_array_equal(local_opt._fitness_function.training_data.x, X)
    np.testing.assert_array_equal(local_opt._fitness_function.training_data.y, y)
    assert local_opt._algorithm == clo_alg
    assert local_opt.optimization_options["tol"] == tol


# TODO test predictor_size_ratio?
@pytest.mark.parametrize("dataset_size, expected_island",
                         [(100, Island), (1199, Island),
                          (1200, FitnessPredictorIsland),
                          (1500, FitnessPredictorIsland)])
def test_make_island(mocker, sample_ea, sample_agraph_gen, dataset_size,
                     expected_island):
    pop_size = 100
    hof = mocker.Mock()

    regr = SymbolicRegressor()
    regr.generator = sample_agraph_gen
    regr.population_size = pop_size

    island = regr._make_island(dataset_size, sample_ea, hof)

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

    unique_indvs = set([str(indv) for indv in island.population])
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

    unique_indvs = set([str(indv) for indv in island.population])
    assert len(unique_indvs) == 1
    assert len(island.population) == pop_size


def patch_all_get_archipelago(mocker):
    imports = ["ComponentGenerator", "AGraphCrossover", "AGraphMutation",
               "AGraphGenerator", "Evaluation", "AgeFitnessEA",
               "DeterministicCrowdingEA", "HallOfFame"]
    fns = ["_get_clo", "_make_island", "_force_diversity_in_island"]

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
    regr = SymbolicRegressor(stack_size=stack_size,
                             use_simplification=use_simplification)
    X, y = get_mocked_data(mocker)
    n_proc = mocker.Mock()

    regr._get_archipelago(X, y, n_proc)

    generator.assert_called_once_with(stack_size, comp_gen.return_value,
                                      use_simplification=use_simplification,
                                      use_python=True)
    assert regr.generator == generator.return_value


def test_get_archipelago_evaluation(mocker):
    patched_objs = patch_all_get_archipelago(mocker)
    age_fitness = mocker.patch("bingo.evolutionary_algorithms.age_fitness")
    age_fitness.AgeFitnessEA = mocker.Mock()
    get_clo = patched_objs["_get_clo"]
    evaluation = patched_objs["Evaluation"]

    clo_tol = mocker.Mock()
    regr = SymbolicRegressor(clo_threshold=clo_tol)
    X, y = get_mocked_data(mocker)
    n_proc = mocker.Mock()

    regr._get_archipelago(X, y, n_proc)

    get_clo.assert_called_once_with(X, y, clo_tol)
    evaluation.assert_called_once_with(get_clo.return_value,
                                       multiprocess=n_proc)


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
    regr = SymbolicRegressor(crossover_prob=cross_prob,
                             mutation_prob=mutation_prob,
                             population_size=pop_size,
                             evolutionary_algorithm=evo_alg)
    X, y = get_mocked_data(mocker)
    n_proc = mocker.Mock()

    regr._get_archipelago(X, y, n_proc)

    if evo_alg_str == "AgeFitnessEA":
        evo_alg.assert_called_once_with(evaluation, generator, crossover,
                                        mutation, cross_prob, mutation_prob,
                                        pop_size)
    else:
        evo_alg.assert_called_once_with(evaluation, crossover, mutation,
                                        cross_prob, mutation_prob)


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
    hof = patched_objs["HallOfFame"]
    make_island = patched_objs["_make_island"]
    force_diversity = patched_objs["_force_diversity_in_island"]

    regr = SymbolicRegressor(evolutionary_algorithm=evo_alg)
    X, y = get_mocked_data(mocker)
    n_proc = mocker.Mock()

    arch = regr._get_archipelago(X, y, n_proc)

    make_island.assert_called_once()
    expected_args = [(len(X), evo_alg.return_value, hof.return_value),
                     (X.shape[0], evo_alg.return_value, hof.return_value)]
    assert make_island.call_args.args in expected_args

    force_diversity.assert_called_once_with(make_island.return_value)

    assert arch == make_island.return_value


def patch_all_fit(mocker):
    imports = ["np.random.seed", "random.seed"]
    fns = ["_get_archipelago", "_refit_best_individual"]

    patched_imports = {name: patch_import(mocker, name) for name in imports}
    patched_fns = {name: patch_regr_method(mocker, name) for name in fns}

    patched_imports.update(patched_fns)

    return patched_imports


@pytest.mark.parametrize("seed", [None, "mock"])
def test_fit_seed(mocker, seed):
    patched_objs = patch_all_fit(mocker)
    np_seed = patched_objs["np.random.seed"]
    random_seed = patched_objs["random.seed"]

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


@pytest.mark.parametrize("n_cpus", [None, 1, 2, 3])
def test_fit_archipelago(mocker, n_cpus):
    patched_objs = patch_all_fit(mocker)
    get_archipelago = patched_objs["_get_archipelago"]
    evolve = get_archipelago.return_value.evolve_until_convergence

    if n_cpus is None:
        n_cpus = 1
        environ_dict = dict()
    else:
        environ_dict = {"OMP_NUM_THREADS": str(n_cpus)}
    mocker.patch.dict(get_sym_reg_import("os.environ"),
                      environ_dict, clear=True)

    max_gens = mocker.Mock()
    fit_threshold = mocker.Mock()
    max_time = mocker.Mock()
    regr = SymbolicRegressor(generations=max_gens,
                             fitness_threshold=fit_threshold,
                             max_time=max_time)
    X, y = get_mocked_data(mocker)

    regr.fit(X, y)

    get_archipelago.assert_called_once_with(X, y, n_cpus)
    assert regr.archipelago == get_archipelago.return_value

    evolve.assert_called_once()
    assert evolve.call_args.kwargs["max_generations"] == max_gens
    assert evolve.call_args.kwargs["fitness_threshold"] == fit_threshold
    assert evolve.call_args.kwargs["max_time"] == max_time


@pytest.mark.parametrize("hof_len", [0, 1])
def test_fit_best_individual(mocker, hof_len):
    patched_objs = patch_all_fit(mocker)
    refit_best_indv = patched_objs["_refit_best_individual"]
    archipelago = patched_objs["_get_archipelago"].return_value
    hof = [mocker.Mock() for _ in range(hof_len)]
    archipelago.hall_of_fame = hof

    regr = SymbolicRegressor()
    X, y = get_mocked_data(mocker)

    regr.fit(X, y)

    if hof_len == 0:
        assert regr.best_ind == archipelago.get_best_individual.return_value
    else:
        assert regr.best_ind == hof[0]

    refit_best_indv.assert_called_once()
    assert (X, y) in refit_best_indv.call_args


def test_fit_return(mocker):
    patch_all_fit(mocker)

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

    with pytest.raises(ValueError):
        regr.get_best_individual()


def test_predict_normal(mocker):
    regr = SymbolicRegressor()
    best_ind = mocker.Mock()
    best_ind.evaluate_equation_at = mocker.Mock(side_effect=lambda x: x)
    regr.best_ind = best_ind
    X = mocker.Mock()

    assert regr.predict(X) == X


def test_predict_bad_output(mocker):
    regr = SymbolicRegressor()
    best_ind = mocker.Mock()
    best_ind.evaluate_equation_at = mocker.Mock(side_effect=lambda x:
                                                [0, -np.inf, np.inf, np.nan,
                                                 1, 2, 3])
    regr.best_ind = best_ind
    expected_output = [0, -INF_REPLACEMENT, INF_REPLACEMENT, 0, 1, 2, 3]

    np.testing.assert_array_equal(regr.predict(mocker.Mock()), expected_output)


# TODO unit testing of _get_archipelago, mock helper methods and make sure
#   they were called with right arguments

# TODO kind of like the integration of _get_archipelago and _make_island
@pytest.mark.parametrize("evo_alg_str", ["AgeFitnessEA",
                                         "DeterministicCrowdingEA"])
def test_evo_opt_gets_args(mocker, basic_data, evo_alg_str):
    mocker.patch(get_sym_reg_import("SymbolicRegressor._force_diversity_in_island"))
    evo_opt = mocker.patch(get_sym_reg_import("Island"))
    agraph_gen = mocker.patch(get_sym_reg_import("AGraphGenerator"))
    evo_alg = mocker.patch(get_sym_reg_import(evo_alg_str))
    hof = mocker.patch(get_sym_reg_import("HallOfFame"))
    population_size = mocker.Mock()
    X, y = basic_data

    regr = SymbolicRegressor(evolutionary_algorithm=evo_alg,
                             population_size=population_size)
    regr._get_archipelago(X, y, 1)

    evo_opt.assert_called_with(evo_alg.return_value, agraph_gen.return_value,
                               population_size, hall_of_fame=hof.return_value)


def test_evo_opt_adapts_to_dataset_size():
    pass


@pytest.mark.parametrize("invalid_size", [0, -1])
def test_stack_size_invalid(invalid_size, basic_data):
    regr = SymbolicRegressor(stack_size=invalid_size)
    X, y = basic_data

    with pytest.raises(ValueError):
        regr.fit(X, y)


def test_operator_none_default_set():
    regr = SymbolicRegressor(operators=None)
    assert regr.operators == {"+", "-", "*", "/"}


@pytest.mark.parametrize("invalid_operators", [{"abc"}, {"3"}, {""},
                                               "sin"  # needs to be list-like
                                               ])     # of operators
def test_operator_invalid(invalid_operators, basic_data):
    regr = SymbolicRegressor(operators=invalid_operators)
    X, y = basic_data

    with pytest.raises(ValueError):
        regr.fit(X, y)


@pytest.mark.parametrize("invalid_value, expected_err",
                         [(-1, ValueError), (1.1, ValueError),
                          ("a", TypeError), (True, ValueError)])
def test_crossover_prob_invalid(invalid_value, basic_data, expected_err):
    regr = SymbolicRegressor(crossover_prob=invalid_value)
    X, y = basic_data

    with pytest.raises(expected_err):
        regr.fit(X, y)


@pytest.mark.parametrize("invalid_value, expected_err",
                         [(-1, ValueError), (1.1, ValueError),
                          ("a", TypeError), (True, ValueError)])
def test_mutation_prob_invalid(invalid_value, basic_data, expected_err):
    regr = SymbolicRegressor(mutation_prob=invalid_value)
    X, y = basic_data

    with pytest.raises(expected_err):
        regr.fit(X, y)


@pytest.mark.parametrize("invalid_metric", ["aaa", 1, False])
def test_metric_invalid(invalid_metric, basic_data):
    regr = SymbolicRegressor(metric=invalid_metric)
    X, y = basic_data

    with pytest.raises(ValueError):
        regr.fit(X, y)


@pytest.mark.parametrize("invalid_alg", ["not_an_alg", 1, False])
def test_clo_alg_invalid(invalid_alg, basic_data):
    regr = SymbolicRegressor(clo_alg=invalid_alg)
    X, y = basic_data

    with pytest.raises(KeyError):
        regr.fit(X, y)


@pytest.mark.timeout(3)
@pytest.mark.parametrize("invalid_n_gens", [0, -1])
def test_n_gens_invalid(invalid_n_gens, basic_data):
    regr = SymbolicRegressor(generations=invalid_n_gens)
    X, y = basic_data

    with pytest.raises(ValueError):
        regr.fit(X, y)

# TODO make sure constructor args get to respective objects
#      e.g. if metric = "mse", make sure fitness gets "mse"

def constructor_params():
    return ["population_size", "stack_size", "operators",
            "use_simplification", "crossover_prob",
            "mutation_prob", "metric",
            "clo_alg", "generations",
            "fitness_threshold", "max_time", "max_evals",
            "evolutionary_algorithm", "clo_threshold",
            "scale_max_evals", "random_state"]


# TODO need set_param tests? should be same as sklearn
@pytest.mark.parametrize("param", constructor_params())
def test_can_set_all_params(mocker, param):
    mocked_value = mocker.Mock(spec=object)  # spec=object so get_params()
    # doesn't try to treat the mock like a dictionary

    regr = SymbolicRegressor()
    output = regr.set_params(**{param: mocked_value})

    assert regr.get_params()[param] == mocked_value
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


def test_get_best_individual(mocker):
    regr = SymbolicRegressor()
    best_ind = mocker.Mock()
    regr.best_ind = best_ind

    assert regr.get_best_individual() == best_ind


def test_get_best_individual_without_fit_raises_error():
    regr = SymbolicRegressor()
    with pytest.raises(ValueError) as exc_info:
        regr.get_best_individual()
    assert str(exc_info.value) == "Best individual not set"


@pytest.mark.parametrize("data_size", [10, int(1e4), int(1e6)])
def test_fit_calls_evo_opt(mocker, data_size):
    # testing different data sizes in case evo opt changes per data size
    X = np.linspace(-10, 10, data_size).reshape((data_size, 1))
    y = np.linspace(10, -10, data_size).reshape((data_size, 1))

    from bingo.local_optimizers.continuous_local_opt import ContinuousLocalOptimization
    mocker.patch.object(ContinuousLocalOptimization, "__call__", return_value=0.01)

    mocked_evo = mocker.patch.object(EvolutionaryOptimizer, "evolve_until_convergence")

    regr = SymbolicRegressor()
    regr.fit(X, y)

    # TODO mock gens, fitness_threshold, max_evals, max_time?
    # can't use assert_called_with bc there might be extra args
    assert mocked_evo.call_args.kwargs["max_generations"] == regr.generations
    assert mocked_evo.call_args.kwargs["fitness_threshold"] == \
           regr.fitness_threshold
    assert mocked_evo.call_args.kwargs["max_fitness_evaluations"] == \
           regr.max_evals
    assert mocked_evo.call_args.kwargs["max_time"] == regr.max_time


def get_linear_agraph(slope, intercept):
    agraph = AGraph()
    agraph.command_array = np.array([[0, 0, 0],  # X_0
                                     [1, 0, 0],  # C_0
                                     [1, 1, 1],  # C_1
                                     [4, 0, 1],  # X_0 * C_0
                                     [2, 2, 3]  # X_0 * C_0 + C_1
                                     ], dtype=int)
    agraph.set_local_optimization_params([slope, intercept])
    return agraph


def test_fit_finds_sol_in_initial_pop(mocker):
    X = np.linspace(-10, 10).reshape((-1, 1))
    y = -2.3 * X + 3.5

    sol_agraph = get_linear_agraph(-2.3, 3.5)
    mocker.patch.object(AGraphGenerator, "__call__", return_value=sol_agraph)

    regr = SymbolicRegressor()
    regr.fit(X, y)

    np.testing.assert_array_equal(regr.get_best_individual().command_array,
                                  sol_agraph.command_array)
    assert regr.archipelago.generational_age == 0


# TODO test random_state=None
def test_fit_sets_random_seed(mocker):
    X = np.linspace(-10, 10).reshape((-1, 1))
    y = -2.3 * X + 3.5
    np_random_seed = mocker.patch("bingo.symbolic_regression.symbolic_regressor.np.random.seed")
    random_seed = mocker.patch("bingo.symbolic_regression.symbolic_regressor.random.seed")
    random_state = mocker.Mock()

    regr = SymbolicRegressor(random_state=random_state)
    regr.fit(X, y)

    np_random_seed.assert_called_with(random_state)
    random_seed.assert_called_with(random_state)


"""
def test_fit_finds_sol_normal(mocker):
    X = np.linspace(-10, 10).reshape((-1, 1))
    y = 5.0 * X

    regr = SymbolicRegressor()
    regr.fit(X, y)

    assert regr.get_best_individual().fitness <= regr.fitness_threshold
"""

# TODO do rest of validation tests
# TODO validation tests on fit
# TODO check to see which param's don't give descriptive exceptions
# TODO get_best_individual normal test
