# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import numpy as np

from bingo.evolutionary_algorithms.age_fitness import AgeFitnessEA
from bingo.evaluation.evaluation import Evaluation
from bingo.evolutionary_optimizers.island import Island
from bingo.local_optimizers.scipy_optimizer import ScipyOptimizer
from bingo.local_optimizers.local_opt_fitness \
    import LocalOptFitnessFunction
from bingo.symbolic_regression import ComponentGenerator, \
                                      AGraphGenerator, \
                                      AGraphCrossover, \
                                      AGraphMutation, \
                                      ExplicitRegression, \
                                      ExplicitTrainingData
POP_SIZE = 100
STACK_SIZE = 20


if __name__ == '__main__':
    def run_result(use_pytorch=True):
        N = 10
        num_x = 2
        x = np.random.randn(N, num_x).astype(np.float64)
        y = (x[:, 0]**2 + 3.5*x[:, 1]**3).reshape((-1, 1))
    
        training_data = ExplicitTrainingData(x, y)

        component_generator = ComponentGenerator(num_x)
        component_generator.add_operator("+")
        component_generator.add_operator("-")
        component_generator.add_operator("*")
    
        crossover = AGraphCrossover()
        mutation = AGraphMutation(component_generator)
    
        # set the use_pytorch flag here also!
        agraph_generator = AGraphGenerator(STACK_SIZE, component_generator,
                                           use_simplification=False,
                                           use_pytorch=use_pytorch)
    
        fitness = ExplicitRegression(training_data=training_data)

        optimizer = ScipyOptimizer(fitness, method='lm')
        local_opt_fitness = LocalOptFitnessFunction(fitness, optimizer)
        evaluator = Evaluation(local_opt_fitness)
    
        ea = AgeFitnessEA(evaluator, agraph_generator, crossover,
                          mutation, 0.4, 0.4, POP_SIZE)
    
        island = Island(ea, agraph_generator, POP_SIZE)
    
        opt_result = island.evolve_until_convergence(max_generations=500,
                                                     fitness_threshold=1e-5)
        if opt_result.success:
            print("Converged with best individual:",
                  island.get_best_individual().get_formatted_string("console"))
        else:
            print("Failed to converge")


    run_result(use_pytorch=True)
