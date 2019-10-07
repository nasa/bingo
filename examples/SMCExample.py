# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import numpy as np
import matplotlib.pyplot as plt

from bingo.symbolic_regression.explicit_regression \
    import ExplicitRegression, ExplicitTrainingData
from bingo.local_optimizers.continuous_local_opt\
    import ContinuousLocalOptimization
from bingo.symbolic_regression.agraph.agraph import AGraph
from bingo.symbolic_regression.smc.model_evidence import ModelEvidenceFunction


def init_x_vals(start, stop, num_points):
    return np.sort(np.random.uniform(start, stop, (num_points, 1)), axis=0)


def equation_eval(x):
    c_1 = np.random.normal(1, 0.25, x.shape)
    c_2 = np.random.normal(3.5, 0.5, x.shape)
    return c_1*x**2 + c_2*x**3


def execute_generational_steps():
    x = init_x_vals(-10, 10, 100)
    y = equation_eval(x)
    training_data = ExplicitTrainingData(x, y)

    fitness = ExplicitRegression(training_data=training_data)
    local_opt_fitness = ContinuousLocalOptimization(fitness, algorithm='lm')

    test_equation = AGraph()
    test_equation.command_array = np.array([[0, 0, 0],  # c_1*x**2 + c_2*x**3
                                            [1, 0, 0],
                                            [1, 1, 1],
                                            [4, 0, 0],
                                            [4, 3, 0],
                                            [4, 3, 1],
                                            [4, 4, 2],
                                            [2, 5, 6]])
    test_equation.constants = [1, 3.5]

    # PLOT DATA
    # plt.plot(x, y, 'r.', label="input_data")
    # plt.plot(x, test_equation.evaluate_equation_at(x), 'b-', label="equation")
    # plt.legend()
    # plt.show()


    # EVIDENCE CALC
    evidence_function = ModelEvidenceFunction(local_opt_fitness)
    # evidence = evidence_function(test_equation)


def main():
    execute_generational_steps()


if __name__ == '__main__':
    main()
