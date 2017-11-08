import time
import pickle

from bingo.AGraph import AGraphManipulator as agm
from bingo.AGraph import AGNodes
#from bingo.Utils import snake_walk, calculate_partials
#from bingo.FitnessPredictor import FPManipulator as fpm
import AGraphCpp
import numpy as np
from bingo.FitnessMetric import StandardRegression


# make data
#full_data_size = 10
#X = np.linspace(0, 1, full_data_size, False)
#Y = X*X + 0.5
#Y = 1.5*X*X - X*X*X
#Y = np.exp(np.abs(X))*np.sin(X)
#Y = X*X*np.exp(np.sin(X)) + X + np.sin(3.14159/4 - X*X*X)
#X = X.reshape([-1, 2])


# X = snake_walk()
# Y = (X[:, 0] + X[:, 1]*0.5)
# X = np.hstack((X, Y.reshape([-1, 1])))
# X, Y = calculate_partials(X)

# make solution manipulator
# a = agm(2, 16, nloads=2, constant_optimization=True)
# a.add_node_type(AGNodes.Add)
# a.add_node_type(AGNodes.Subtract)
# a.add_node_type(AGNodes.Multiply)
# # a.add_node_type(AGNodes.Divide)
# # a.add_node_type(AGNodes.Exp)
# # a.add_node_type(AGNodes.Log)
# # a.add_node_type(AGNodes.Sin)
# # a.add_node_type(AGNodes.Cos)
# # a.add_node_type(AGNodes.Abs)


# # make predictor manipulator
# b = fpm(32, X.shape[0])
# fp = b.generate()
# print(fp)
#
# ag = a.generate()
#
# ag.command_list[0] = (AGNodes.Load_Data, (0,))
# ag.command_list[1] = (AGNodes.Load_Data, (1,))
# ag.command_list[2] = (AGNodes.Load_Const, (None,))
# ag.command_list[3] = (AGNodes.Multiply, (1, 2))
# ag.command_list[4] = (AGNodes.Add, (0, 3))
# ag.command_list[5] = (AGNodes.Load_Data, (2,))
# ag.command_list[-1] = (AGNodes.Subtract, (5, 4))
#
# print(ag)
# print(ag.latexstring())
# # for x, y in zip(X[fp.indices, :], Y[fp.indices, :]):
# #     print(x, y, ag.evaluate_deriv(x), ag.evaluate_deriv(x)*y)
#
# print(fp.fit_func(ag, X, Y, False))


ac = AGraphCpp.AGraphCppManipulator(2,16,nloads=2)
ac.add_node_type(2)
ac.add_node_type(3)
gc = ac.generate()
gc_list, _ = ac.dump(gc)


a = agm(2, 16, nloads=2, constant_optimization=True)
a.add_node_type(AGNodes.Add)
a.add_node_type(AGNodes.Subtract)

for i, (node, params) in enumerate(gc_list):
    if node is 1:
        if params[0] is -1:
            gc_list[i] = (node, (None,))
g = a.load(gc_list)



#print(g)
#print(gc)



x = np.linspace(1,100,100)
x = x.reshape((-1,2))
y = x[:,0] + 50
y = y.reshape((-1, 1))

kwargs={}
kwargs['x'] = x
kwargs['y'] = y
t0 = time.time()
g.evaluate(StandardRegression, **kwargs)
t1 = time.time()
gc.evaluate(StandardRegression, **kwargs)
t2 = time.time()
print(g.latexstring())
print(gc.latexstring())
print("g evaluation time: ", t1-t0)
print("gc evaluation time: ", t2-t1)
print("speed up: ", (t1-t0)/(t2-t1))