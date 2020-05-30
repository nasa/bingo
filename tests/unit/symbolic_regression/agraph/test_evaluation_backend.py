# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import numpy as np
import pytest

from bingo.symbolic_regression.agraph.operator_definitions import *
from bingo.symbolic_regression.agraph.evaluation_backend \
    import evaluation_backend as py_eval_backend

try:
    from bingocpp.build import symbolic_regression as cpp_eval_backend
except ImportError:
    cpp_eval_backend = None

CPP_PARAM = pytest.param("Cpp",
                         marks=pytest.mark.skipif(not cpp_eval_backend,
                                                  reason='BingoCpp import '
                                                         'failure'))

OPERATOR_LIST = [INTEGER, VARIABLE, CONSTANT, ADDITION, SUBTRACTION,
                 MULTIPLICATION, DIVISION, SIN, COS, EXPONENTIAL, LOGARITHM,
                 POWER, ABS, SQRT]


@pytest.fixture(params=["Python", CPP_PARAM])
def engine(request):
    return request.param


@pytest.fixture
def eval_backend(engine):
    if engine == "Python":
        return py_eval_backend
    return cpp_eval_backend


@pytest.fixture
def all_funcs_command_array():
    return np.array([[INTEGER, 5, 5],
                     [VARIABLE, 0, 0],
                     [CONSTANT, 0, 0],
                     [ADDITION, 1, 0],
                     [SUBTRACTION, 2, 3],
                     [MULTIPLICATION, 4, 1],
                     [DIVISION, 5, 1],
                     [SIN, 6, 0],
                     [COS, 7, 0],
                     [EXPONENTIAL, 8, 0],
                     [LOGARITHM, 9, 0],
                     [POWER, 10, 0],
                     [ABS, 11, 0],
                     [SQRT, 12, 0]])


@pytest.fixture
def higher_dim_command_array():
    return np.array([[VARIABLE, 0, 0],
                     [VARIABLE, 1, 1],
                     [CONSTANT, 0, 0],
                     [CONSTANT, 1, 1],
                     [MULTIPLICATION, 0, 2],
                     [MULTIPLICATION, 1, 3],
                     [ADDITION, 4, 5]])


@pytest.fixture
def sample_x():
    return np.vstack((np.linspace(-1.0, 0.0, 11),
               np.linspace(0.0, 1.0, 11))).transpose()


@pytest.fixture
def sample_constants():
    return 10, 3.14


@pytest.fixture
def operator_evals_x0(sample_x, sample_constants):
    x_0 = sample_x[:, 0].reshape((-1, 1))
    c_0 = np.full(x_0.shape, sample_constants[0])
    return {INTEGER: np.zeros_like(x_0),
            VARIABLE: x_0,
            CONSTANT: c_0,
            ADDITION: x_0+x_0,
            SUBTRACTION: x_0-x_0,
            MULTIPLICATION: x_0*x_0,
            DIVISION: x_0/x_0,
            SIN: np.sin(x_0),
            COS: np.cos(x_0),
            EXPONENTIAL: np.exp(x_0),
            LOGARITHM: np.log(np.abs(x_0)),
            POWER: np.power(np.abs(x_0), x_0),
            ABS: np.abs(x_0),
            SQRT: np.sqrt(np.abs(x_0))}


@pytest.fixture
def operator_x_derivs(sample_x):
    def last_nan(array):
        array[-1] = np.nan
        return array
    x_0 = sample_x[:, 0].reshape((-1, 1))
    return {INTEGER: np.zeros_like(x_0),
            VARIABLE: np.ones(x_0.shape),
            CONSTANT: np.zeros(x_0.shape),
            ADDITION: np.full(x_0.shape, 2.0),
            SUBTRACTION: np.zeros(x_0.shape),
            MULTIPLICATION: 2*x_0,
            DIVISION: last_nan(np.zeros(x_0.shape)),
            SIN: np.cos(x_0),
            COS: -np.sin(x_0),
            EXPONENTIAL: np.exp(x_0),
            LOGARITHM: 1.0 / x_0,
            POWER: last_nan(np.power(np.abs(x_0), x_0)*(np.log(np.abs(x_0)) + 1)),
            ABS: np.sign(x_0),
            SQRT: 0.5*np.sign(x_0) / np.sqrt(np.abs(x_0))}


@pytest.fixture
def operator_c_derivs(sample_x, sample_constants):
    c_1 = np.full((sample_x.shape[0], 1), sample_constants[1])
    return {INTEGER: np.zeros_like(c_1),
            VARIABLE: np.zeros(c_1.shape),
            CONSTANT: np.ones(c_1.shape),
            ADDITION: np.full(c_1.shape, 2.0),
            SUBTRACTION: np.zeros(c_1.shape),
            MULTIPLICATION: 2*c_1,
            DIVISION: np.zeros(c_1.shape),
            SIN: np.cos(c_1),
            COS: -np.sin(c_1),
            EXPONENTIAL: np.exp(c_1),
            LOGARITHM: 1.0 / c_1,
            POWER: np.power(np.abs(c_1), c_1)*(np.log(np.abs(c_1)) + 1),
            ABS: np.sign(c_1),
            SQRT: 0.5*np.sign(c_1) / np.sqrt(np.abs(c_1))}


def test_all_funcs_eval(eval_backend, all_funcs_command_array):
    x = np.arange(1, 6).reshape((-1, 1))
    constants = (10, )
    expected_f_of_x = np.array([[0.45070097],
                                [0.9753327],
                                [0.29576841],
                                [0.36247937],
                                [1.0]])
    f_of_x = eval_backend.evaluate(all_funcs_command_array,
                                   x, constants)
    np.testing.assert_array_almost_equal(f_of_x, expected_f_of_x)


def test_higher_dim_func_eval(eval_backend, higher_dim_command_array):
    x = np.arange(8).reshape((-1, 2))
    constants = (10, 100)
    expected_f_of_x = np.sum(x*constants, axis=1).reshape((-1, 1))
    f_of_x = eval_backend.evaluate(higher_dim_command_array,
                                   x, constants)
    np.testing.assert_array_almost_equal(f_of_x, expected_f_of_x)


def test_all_funcs_deriv_x(eval_backend, all_funcs_command_array):
    x = np.arange(1, 6).reshape((-1, 1))
    constants = (10, )
    expected_f_of_x = np.array([[0.45070097],
                                [0.9753327],
                                [0.29576841],
                                [0.36247937],
                                [1.0]])
    expected_df_dx = np.array([[0.69553357],
                               [-0.34293336],
                               [-0.39525239],
                               [0.54785643],
                               [0.0]])
    f_of_x, df_dx = eval_backend.evaluate_with_derivative(
            all_funcs_command_array, x, constants, True)
    np.testing.assert_array_almost_equal(f_of_x, expected_f_of_x)
    np.testing.assert_array_almost_equal(df_dx, expected_df_dx)


def test_all_funcs_deriv_c(eval_backend, all_funcs_command_array):
    x = np.arange(1, 6).reshape((-1, 1))
    constants = (10, )
    expected_f_of_x = np.array([[0.45070097],
                                [0.9753327],
                                [0.29576841],
                                [0.36247937],
                                [1.0]])
    expected_df_dc = np.array([[-0.69553357],
                               [0.34293336],
                               [0.39525239],
                               [-0.54785643],
                               [0.]])
    f_of_x, df_dc = eval_backend.evaluate_with_derivative(
            all_funcs_command_array, x, constants, False)
    np.testing.assert_array_almost_equal(f_of_x, expected_f_of_x)
    np.testing.assert_array_almost_equal(df_dc, expected_df_dc)


def test_higher_dim_func_deriv_x(eval_backend, higher_dim_command_array):
    x = np.arange(8).reshape((4, 2))
    constants = (10, 100)
    expected_f_of_x = np.sum(x*constants, axis=1).reshape((-1, 1))
    expected_df_dx = np.array([constants]*4)

    f_of_x, df_dx = eval_backend.evaluate_with_derivative(
            higher_dim_command_array, x, constants, True)
    np.testing.assert_array_almost_equal(f_of_x, expected_f_of_x)
    np.testing.assert_array_almost_equal(df_dx, expected_df_dx)


def test_higher_dim_func_deriv_c(eval_backend, higher_dim_command_array):
    x = np.arange(8).reshape((4, 2))
    constants = (10, 100)
    expected_f_of_x = np.sum(x*constants, axis=1).reshape((-1, 1))
    expected_df_dc = x

    f_of_x, df_dc = eval_backend.evaluate_with_derivative(
            higher_dim_command_array, x, constants, False)
    np.testing.assert_array_almost_equal(f_of_x, expected_f_of_x)
    np.testing.assert_array_almost_equal(df_dc, expected_df_dc)


@pytest.mark.parametrize("operator", OPERATOR_LIST)
def test_backend_evaluate(eval_backend, sample_x, sample_constants, operator,
                          operator_evals_x0):
    expected_outcome = operator_evals_x0[operator]
    stack = np.array([[0, 0, 0],
                      [0, 1, 1],
                      [operator, 0, 0]])
    f_of_x = eval_backend.evaluate(stack, sample_x, sample_constants)
    np.testing.assert_allclose(expected_outcome, f_of_x)


@pytest.mark.parametrize("operator", OPERATOR_LIST)
def test_backend_evaluate_with_x_derivative(eval_backend, sample_x,
                                            sample_constants, operator,
                                            operator_x_derivs):
    expected_derivative = np.zeros(sample_x.shape)
    expected_derivative[:, 0] = operator_x_derivs[operator].flatten()
    stack = np.array([[0, 0, 0],
                      [0, 0, 0],
                      [0, 1, 1],
                      [operator, 0, 1]])
    _, df_dx = eval_backend.evaluate_with_derivative(stack, sample_x,
                                                     sample_constants, True)
    np.testing.assert_allclose(expected_derivative, df_dx)


@pytest.mark.parametrize("operator", OPERATOR_LIST)
def test_backend_evaluate_with_c_derivative(eval_backend, sample_x,
                                            sample_constants, operator,
                                            operator_c_derivs):
    expected_derivative = np.zeros(sample_x.shape)
    expected_derivative[:, 1] = operator_c_derivs[operator].flatten()
    stack = np.array([[1, 1, 1],
                      [1, 1, 1],
                      [0, 1, 1],
                      [operator, 1, 0]])
    _, df_dx = eval_backend.evaluate_with_derivative(stack, sample_x,
                                                     sample_constants, False)
    np.testing.assert_allclose(expected_derivative, df_dx)
