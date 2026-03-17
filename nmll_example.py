import numpy as np

# need to use this for hessian calculation, which is required for NMLL. The standard AGraph does not support this.
from bingo.symbolic_regression.agraph.pytorch_agraph import PytorchAGraph as AGraph

from bingo.symbolic_regression.explicit_regression import (
    ExplicitRegression,
    ExplicitTrainingData,
)
from bingo.local_optimizers.scipy_optimizer import ScipyOptimizer
from bingo.local_optimizers.normalized_marginal_likelihood import (
    NormalizedMarginalLikelihood,
)

# Build agraph for a*x0 + b
# Command array: [operator, operand1, operand2]
#   0: VARIABLE  1: CONSTANT  2: ADDITION  4: MULTIPLICATION
commands = np.array(
    [
        [0, 0, 0],  # x0
        [1, 0, 0],  # C_0 (a)
        [1, 1, 1],  # C_1 (b)
        [4, 1, 0],  # C_0 * x0
        [2, 3, 2],  # (C_0 * x0) + C_1
    ]
)

equ = AGraph(use_simplification=False)
equ.command_array = commands
equ.set_local_optimization_params(np.array([1.0, 1.0]))  # initial guess

# Generate data: y = 2.5*x + 1 + noise
np.random.seed(0)
x = np.linspace(-5, 5, 20).reshape(-1, 1)
y_true = 2.5 * x + 1.0
y = y_true + np.random.normal(0, 0.5, y_true.shape)

training_data = ExplicitTrainingData(x, y)

# Set up fitness and NMLL
fitness = ExplicitRegression(training_data=training_data, metric="mse")
scipy_opt = ScipyOptimizer(fitness, method="lm")
nmll = NormalizedMarginalLikelihood(
    fitness, scipy_opt, num_particles=150, mcmc_steps=12, num_multistarts=1
)

# Evaluate NMLL
nmll_value = nmll(equ)
print(f"Equation: {equ}")
print(f"Optimized constants: {equ.constants}")
print(f"NMLL: {nmll_value}")
