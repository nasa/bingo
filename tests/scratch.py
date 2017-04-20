import time
import pickle

from bingo.AGraph import AGraphManipulator as agm
from bingo.AGraph import AGNodes
from bingo.FitnessPredictor import FPManipulator as fpm
from bingo.FitnessPredictor import FitnessPredictor as fp
from bingo.CoevolutionIsland import CoevolutionIsland as ci
from bingo.Plotting import print_latex, print_pareto

import numpy as np

from bingo.Island import Island

a = agm(1, 16, nloads=2)
a.add_node_type(AGNodes.Add)
a.add_node_type(AGNodes.Subtract)
a.add_node_type(AGNodes.Multiply)
# a.add_node_type(AGNodes.Divide)
# a.add_node_type(AGNodes.Sin)
# a.add_node_type(AGNodes.Cos)

#p1 = a.generate()
#print p1.latexstring()
#d = a.dump(p1)
#pickle.dump(d, open("test.p", "w"))
#p11 = a.load(d)
#print p11.latexstring()

#p2 = a.generate()


#print "p1", p1
#print "p2", p2

#c1, c2 = a.crossover(p1, p2)
#print "----------------------------"
#print "p1", p1.latexstring()
#print "p2", p2.latexstring()
#print "c1", c1.latexstring()
#print "c2", c2

#c2_mut = a.mutation(c2.copy())
#print "c2_mut", c2_mut
#print c2_mut.latexstring()

#print "c2_mut(0,0):", c2_mut.evaluate([1, 1])


#print "c1", c1
#print "c2", c2_mut
#print "dist", a.distance(c2, c2_mut)

"""
def fitness(indv):
    X = np.linspace(0, 1)
    err = 0.0
    for x in X:
        err += abs(indv.evaluate([x]) - (x*x+0.5))
    return err, indv.complexity()

#print fitness(c1)

t0 = time.time()
isle = Island(a, fitness, 200)
d = isle.dump_population()
for i in range(10):
    isle.update_pareto_front()
    best = isle.pareto_front[0]
    print i, fitness(best), best.latexstring()
    isle.deterministic_crowding_step()
isle.load_population(d)
for i in range(2):
    isle.update_pareto_front()
    best = isle.pareto_front[0]
    print i, fitness(best), best.latexstring()
    isle.deterministic_crowding_step()
t1 = time.time()
print "time taken:", t1-t0
"""


#f = fpm(10, 10)

#i1 = fp([1, 2, 2, 7, 8])
#i2 = fp([0, 0, 2, 3, 4])
#print i1
#print i2
#print f.distance(i1, i2)
#print f.distance(i2, i1)


#def fitness_fp(indv):
#    return f.distance(indv, fp([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))

#t0 = time.time()
#isle = Island(f, fitness_fp, 200)
#for i in range(100):
#    best = isle.best_indv()
#    print i, fitness_fp(best), best
#    isle.deterministic_crowding_step()
#t1 = time.time()
#print "time taken:", t1-t0

'''
# make starting data
full_data_size = 200
X = np.linspace(0, 1, full_data_size)
Y = X*X + 0.5
X = X.reshape([-1, 1])

# make fitness predictor
f = fpm(16, full_data_size)
s = f.generate()
fitness = lambda indv: s.fit_func(indv, X, Y)

# do island
t0 = time.time()
isle = Island(a, fitness, 200)
for i in range(100):
    best = isle.best_indv()
    print i, fitness(best), best.latexstring()
    isle.deterministic_crowding_step()
t1 = time.time()
print "time taken:", t1-t0
'''


'''
import copy_reg
from types import MethodType

def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)
'''






# make data
full_data_size = 200
X = np.linspace(-3, 3, full_data_size, False)
#Y = X*X + 0.5
#Y = 1.5*X*X - X*X*X
#Y = np.exp(np.abs(X))*np.sin(X)
Y = X*X*np.exp(np.sin(X)) + X + np.sin(3.14159/4 - X*X*X)
X = X.reshape([-1, 1])

# make solution manipulator
sol_manip = agm(1, 64, nloads=2)
sol_manip.add_node_type(AGNodes.Add)
sol_manip.add_node_type(AGNodes.Subtract)
sol_manip.add_node_type(AGNodes.Multiply)
sol_manip.add_node_type(AGNodes.Divide)
sol_manip.add_node_type(AGNodes.Exp)
sol_manip.add_node_type(AGNodes.Log)
sol_manip.add_node_type(AGNodes.Sin)
sol_manip.add_node_type(AGNodes.Cos)
sol_manip.add_node_type(AGNodes.Abs)

# make predictor manipulator
pred_manip = fpm(16, full_data_size)

# make coevolution island
t0 = time.time()
isle = ci(X, Y,
          2048, 0.7, 0.01, sol_manip,
          128, 0.5, 0.1, pred_manip, 0.1, 100,
          10, 50)
d = isle.dump_populations()
print "----------"
for i in range(5):
    isle.deterministic_crowding_step()
    # print_pareto(isle.solution_island.pareto_front, "front.tif")
t1 = time.time()
print "time taken:", t1-t0
isle.load_populations(d)
for i in range(5):
    isle.deterministic_crowding_step()

for indv in isle.solution_island.pareto_front:
    print "pareto>", indv.fitness, indv.latexstring()
print_latex(isle.solution_island.pareto_front, "eq.tif")
print_pareto(isle.solution_island.pareto_front, "front.tif")




#copy_reg.pickle(MethodType, _pickle_method, _unpickle_method)
#pickle.dump(isle, open("isle.p", "w"))
