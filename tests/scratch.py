import time
import pickle

from bingo.AGraph import AGraphManipulator as agm
from bingo.AGraph import AGNodes
from bingo.Utils import snake_walk, calculate_partials
from bingo.FitnessPredictor import FPManipulator as fpm

import numpy as np



# make data
#full_data_size = 10
#X = np.linspace(0, 1, full_data_size, False)
#Y = X*X + 0.5
#Y = 1.5*X*X - X*X*X
#Y = np.exp(np.abs(X))*np.sin(X)
#Y = X*X*np.exp(np.sin(X)) + X + np.sin(3.14159/4 - X*X*X)
#X = X.reshape([-1, 2])


X = snake_walk()
Y = (X[:, 0] + X[:, 1])
X = np.hstack((X, Y.reshape([-1, 1])))
X, Y = calculate_partials(X)

# make solution manipulator
a = agm(3, 8, nloads=2)
a.add_node_type(AGNodes.Add)
a.add_node_type(AGNodes.Subtract)
# a.add_node_type(AGNodes.Multiply)
# a.add_node_type(AGNodes.Divide)
# a.add_node_type(AGNodes.Exp)
# a.add_node_type(AGNodes.Log)
# a.add_node_type(AGNodes.Sin)
# a.add_node_type(AGNodes.Cos)
# a.add_node_type(AGNodes.Abs)


# make predictor manipulator
b = fpm(32, X.shape[0])
fp = b.generate()

ag = a.generate()
print ag.latexstring()
for x, y in zip(X[fp.indices, :], Y[fp.indices, :]):
    print x, y, ag.evaluate_deriv(x), ag.evaluate_deriv(x)*y

print fp.fit_func(ag, X, Y, False)
