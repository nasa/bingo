from bingo.SimpleGp import SimpleGp
from bingo.FitnessMetric import StandardRegression
import numpy as np

h = 200  # m
g = 9.8  # m/s^2
t = np.linspace(0,5)  # t = [0 - 5]
y = h - 1.0/2 * g * t * t

y = y.reshape((-1, 1))
t = t.reshape((-1, 1))

building_blocks = [
                    "Add",
                    "Subtract",
                    "Multiply",
                  ]
fitness = StandardRegression()  # sum of squared differences
population_size = 32
gp_pop = SimpleGp(building_blocks, fitness, population_size, data=[t, y],
                  var_names=["t", "y"])  #-**********

print("Initial population:\n", gp_pop)  #------------------------****
gp_pop.plot()

gp_pop.evolve(9)
print("best individual after 10 generations: ",
      gp_pop.get_best_individual()[1])

gp_pop.evolve(90)
print("best individual after 100 generations: ",
      gp_pop.get_best_individual()[1])

gp_pop.evolve(900)
print("best individual after 1000 generations: ",
      gp_pop.get_best_individual()[1])
