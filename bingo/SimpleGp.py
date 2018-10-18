"""
This module is meant to provide a wrapper to some of the core bingo components
that hides a lot of the complexity.
"""
import logging

from bingo.AGraph import AGNodes, AGraphManipulator
from bingo.FitnessPredictor import FPManipulator
from bingo.IslandManager import SerialIslandManager
from bingo.FitnessMetric import StandardRegression, ImplicitRegression
from bingo.TrainingData import ExplicitTrainingData, ImplicitTrainingData

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sympy import sympify


class SimpleGp(object):
    """
    SimpleGP is a class that wraps some of the functionality of bingo into a
    simple tool that can be used for education or illustration.  Currently it
    is limited to explicit and implicit regression and uses the full-python
    AGraph.
    """

    def __init__(self, building_blocks, fitness_metric, population_size, data,
                 var_names, complexity=16, fp_complexity=32):
        """
        Initialize a simple gp object with some minimum information

        :param building_blocks: list of strings containing the mathematical
                                operators to use
        :param fitness_metric: the fitness metric (implicit or explicit)
        :param population_size: size of the population of evolving equations
        :param data: tuple of 2D numpy arrays representing x and y for explicit
                     or just x for implicit
        :param var_names: list of strings representing variable names
                          associated with the columns of x
        :param complexity: int, maximum complexity of the evolving equations
        :param fp_complexity: int, amount of the data to use for fitness eval
        """

        logging.basicConfig(level=logging.DEBUG, format='%(message)s')

        self.fitness_metric = fitness_metric
        self.var_names = var_names

        # make training data
        if isinstance(self.fitness_metric, StandardRegression):
            training_data = ExplicitTrainingData(data[0], data[1])
            self.X = training_data.x
            self.Y = training_data.y
        elif isinstance(self.fitness_metric, ImplicitRegression):
            training_data = ImplicitTrainingData(data[0])
            self.X = training_data.x
        else:
            print("***ERROR*** only standard and implict regression are " +\
                  "supported by simplegp!\n")
            exit(-1)

        sol_manip = AGraphManipulator(self.X.shape[1], complexity,
                                      self.X.shape[1])
        pred_manip = FPManipulator(fp_complexity, self.X.shape[0])

        node_dict = {'Add':AGNodes.Add,
                     'Subtract':AGNodes.Subtract,
                     'Divide':AGNodes.Divide,
                     'Multiply':AGNodes.Multiply,
                     'Sin':AGNodes.Sin,
                     'Cos':AGNodes.Cos,
                     'Exp':AGNodes.Exp,
                     'Sqrt':AGNodes.Sqrt,
                     'Log':AGNodes.Log,
                     'Abs':AGNodes.Abs,
                    }
        for block in building_blocks:
            sol_manip.add_node_type(node_dict[block])

        self.islmngr = SerialIslandManager(n_islands=1,
                                           solution_training_data=training_data,
                                           solution_manipulator=sol_manip,
                                           predictor_manipulator=pred_manip,
                                           solution_pop_size=population_size,
                                           fitness_metric=fitness_metric
                                          )
        self.evolve()

    def __str__(self):
        """
        Prints the population of equations in the simple gp object
        """
        print_str = ""
        for j, indv in enumerate(self.islmngr.isles[0].solution_island.pop):
            indv_str = self.indv_string(indv)
            print_str += "indv " + str(j) + "   " + indv_str + "\n"
        return print_str

    def plot(self):
        """
        Plots the simple gp equations using matplotlib (explicit only)
        """
        plt.plot(self.X[:, 0], self.Y, '.')
        for indv in self.islmngr.isles[0].solution_island.pop:
            y_est = indv.evaluate_equation_at(x=self.X)
            y_est = y_est*np.ones((self.X.shape[0], 1))
            plt.plot(self.X[:, 0], y_est[:, 0])

    def get_plot_data(self, x_eval=None):
        """
        Gets the data needed to plot (explicit only)

        :param x_eval: X at which to evaluate the functions
        :return: list of n_pop lists of [x, y, fitness, label]
        """
        data = []
        if x_eval is None:
            x_eval = self.X
        for indv in self.islmngr.isles[0].solution_island.pop:
            y_est = indv.evaluate_equation_at(x=x_eval)
            y_est = y_est*np.ones((x_eval.shape[0], 1))
            data.append([x_eval[:, 0], y_est[:, 0], indv.fitness,
                         self.indv_string(indv)])
        return data

    def plotly(self):
        """
        Plots the simple gp equations using plotly (explicit only)
        """
        data = [go.Scatter(x=self.X[:, 0], y=self.Y, mode='markers',
                           name="raw data")]
        for indv in self.islmngr.isles[0].solution_island.pop:
            y_est = indv.evaluate_equation_at(x=self.X)
            y_est = y_est*np.ones((self.X.shape[0], 1))
            data.append(dict(type='scatter',
                             name=self.indv_string(indv),
                             x=self.X[:, 0],
                             y=y_est[:, 0], mode='line',
                             hoverinfo="name",
                            ))
        layout = go.Layout(yaxis=dict(range=[-0.5*self.Y[0], self.Y[0]*1.5],
                                      title='height (m)'),
                           xaxis=dict(title='time (s)'),
                           hovermode="closest")
        return go.Figure(data=data, layout=layout)

    def indv_string(self, indv):
        """
        Converts the individual to a simple string equation that can be parsed
        with sympy then simplified

        :param indv: the individual to be converted to string representation
        :return: string of equation of individual
        """
        indv_str = indv.get_latex_string()
        if indv.constants is not None:
            for i, const in enumerate(indv.constants):
                indv_str = indv_str.replace("c_"+str(i),
                                            "{:.4f}".format(const))
        for i, variable in enumerate(self.var_names):
            indv_str = indv_str.replace("x_"+str(i), variable)
        indv_str = indv_str.replace(")(", ")*(")
        indv_str = str(sympify(indv_str))
        indv_str = indv_str.replace("**", "^")
        indv_str = indv_str.replace("*", "")
        return str(indv_str)

    def evolve(self, n_generations=1):
        """
        Let the simple gp equations evolve a set number of genrations

        :param n_generations: the number of generations to evolve
        """
        self.islmngr.do_steps(n_generations)

    def print_best_individual(self):
        """
        Returns the best equationin the simple gp population in simplified
        string form

        :return: string of best equation
        """
        self.update_true_pareto()
        indv = self.islmngr.pareto_isle.pareto_front[0]
        indv_str = self.indv_string(indv)
        print(indv_str)

    def get_best_individual(self):
        """
        Gets the best equation in the simple gp population

        :return: fitness and string of the best individual
        """
        self.update_true_pareto()
        indv = self.islmngr.pareto_isle.pareto_front[0]
        indv_str = self.indv_string(indv)
        return indv.fitness, indv_str

    def update_true_pareto(self):
        """
        Updates the pareto front using true fitness values
        """
        # get list of all pareto individuals
        par_list = self.islmngr.isles[0].solution_island.dump_pareto()\
                   + self.islmngr.pareto_isle.dump_pareto()

        # load into pareto island
        self.islmngr.pareto_isle.load_population(par_list)
        self.islmngr.pareto_isle.update_pareto_front()
