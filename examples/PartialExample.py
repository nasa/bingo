# to take a partial, call equation.evaluate_equation_with_x_partial_at(x, partial_variables)
# where partial variables is a list of the variables to take derivatives wrt to
# e.g., [1, 0] = take partial wrt to X_1 then X_0

# evaluate_equation_with_x_partial_at will return the equation evaluated at x
# and then a list of the request partials (see examples below)

import numpy as np

from bingo.symbolic_regression.agraph.pytorch_agraph import PytorchAGraph

equation = PytorchAGraph(equation="X_0 ** 2 + 3.5 * X_1 ** 3")

N = 100
num_x = 2
numpy_x = np.random.randn(N, num_x).astype(np.float64)

# demonstrating mixed partials
equation_eval_1, partials_1 = equation.evaluate_equation_with_x_partial_at(numpy_x, [0, 1])
print("mixed partials:")
print("\tpartial 1 (d / dX_0) matches expected?", end=" ")
print(np.allclose(partials_1[0],
                  (2 * numpy_x[:, 0]).reshape((-1, 1))))  # d / dX_0
print("\tpartial 2 (d^2 / (dX_0 dX_1)) matches expected?", end=" ")
print(np.allclose(partials_1[1],
                  np.zeros((N, 1))))  # d^2 / (dX_0 dX_1)

# demonstrating higher order partials
equation_eval_2, partials_2 = equation.evaluate_equation_with_x_partial_at(numpy_x, [1, 1, 1])
print("\nhigher order partials:")
print("\tpartial 1 (d / dX_1) matches expected?", end=" ")
print(np.allclose(partials_2[0],
                  (10.5 * numpy_x[:, 1] ** 2).reshape((-1, 1))))  # d / dX_1
print("\tpartial 2 (d^2 / dX_1^2) matches expected?", end=" ")
print(np.allclose(partials_2[1],
                  (21.0 * numpy_x[:, 1]).reshape((-1, 1))))  # d^2 / dX_1^2
print("\tpartial 3 (d^3 / dX_1^3) matches expected?", end=" ")
print(np.allclose(partials_2[2],
                  21.0 * np.ones((N, 1))))  # d^3 / dX_1^3
