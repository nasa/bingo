import time
import pickle

from bingo.AGraph import AGraphManipulator as agm
from bingo.AGraph import AGNodes

import numpy as np



# make data
full_data_size = 10
X = np.linspace(0, 1, full_data_size, False)
#Y = X*X + 0.5
#Y = 1.5*X*X - X*X*X
#Y = np.exp(np.abs(X))*np.sin(X)
Y = X*X*np.exp(np.sin(X)) + X + np.sin(3.14159/4 - X*X*X)
X = X.reshape([-1, 2])

# make solution manipulator
a = agm(2, 8, nloads=2)
a.add_node_type(AGNodes.Add)
a.add_node_type(AGNodes.Subtract)
a.add_node_type(AGNodes.Multiply)
a.add_node_type(AGNodes.Divide)
a.add_node_type(AGNodes.Exp)
a.add_node_type(AGNodes.Log)
a.add_node_type(AGNodes.Sin)
a.add_node_type(AGNodes.Cos)
a.add_node_type(AGNodes.Abs)

ag = a.generate()
print ag.latexstring()
for x in X:
    print x, ag.evaluate_deriv(x)
