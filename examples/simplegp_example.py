from bingo.SimpleGp import SimpleGp
from bingo.FitnessMetric import StandardRegression, ImplicitRegression
import numpy as np

h = 200  # m
g = 9.8  # m/s^2
t = np.linspace(0,5)  # t = [0 - 5]
y = h - 1.0/2 * g * t * t

y = y.reshape((-1,1))
t = t.reshape((-1,1))

building_blocks = [
                    "Add",
                    "Subtract",
                    "Multiply",
                  ]
fitness = StandardRegression()  # sum of squared differences
population_size = 10
gp_pop = SimpleGp(building_blocks, fitness, population_size, data=[t,y],
                  var_names=["t", "y"])  #-**********
print(gp_pop)  #----------------------------------------------------------**********
gp_pop.plot() 
