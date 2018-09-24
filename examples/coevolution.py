import matplotlib
matplotlib.use('Agg')
import numpy as np
import time

import matplotlib.pyplot as plt

from bingo.AGraph import AGraphManipulator
from bingo.AGraph import AGNodes
from bingo.FitnessPredictor import FPManipulator
from bingo.FitnessMetric import StandardRegression
from bingo.TrainingData import ExplicitTrainingData
from bingo.Island import Island
from bingo.CoevolutionIsland import CoevolutionIsland


import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

data_size = 1000
n_steps = 250

data_x = np.linspace(0,10,data_size)
data_y = 10 - (data_x - 5)*data_x

plt.plot(data_x, data_y, '.')
plt.savefig('raw_data.png')

training_data = ExplicitTrainingData(data_x.reshape((-1,1)), data_y.reshape((-1, 1)))
error_metric = StandardRegression()

def fitness(indv):
    return error_metric.evaluate_fitness(indv, training_data)

sol_manip = AGraphManipulator(1, 128, nloads=1)
sol_manip.add_node_type(AGNodes.Add)
sol_manip.add_node_type(AGNodes.Subtract)
sol_manip.add_node_type(AGNodes.Multiply)

pred_manip = FPManipulator(10, data_size)


i = Island(sol_manip, fitness)
t0 = time.time()
for _ in range(n_steps):
    i.deterministic_crowding_step()
t1 = time.time()
print(n_steps,"island steps took:", t1-t0)


ci = CoevolutionIsland(training_data, sol_manip, pred_manip, error_metric, verbose=False)
t0 = time.time()
for _ in range(n_steps):
    ci.generational_step()
t1 = time.time()
print(n_steps,"coevolution island steps took:", t1-t0)

    

