"""
This module contains the functions for general plotting purposes
"""
import matplotlib.pyplot as plt
import pylab


def print_latex(pop, file_name):
    """
    Makes pyplot figure with latex table on it: the table consists of equations
     describing the populations

    :param pop: population containing individuals with fitnesses and a
                latexstring() member function
    :param file_name: desired output file name
    """

    plt.figure(figsize=(10, len(pop)))
    plt.plot()

    for i, indv in enumerate(pop):
        plt.text(0.32, i, "$%s$" % indv.latexstring())
        plt.text(0.02, i,
                 "${0:.3e}$        ${1:.2f}$".format(*indv.fitness))

    plt.text(0.01, len(pop), "Fitness   Complexity")
    plt.text(0.31, len(pop), "Equation")
    plt.axis([0, 1, -1, 1 + len(pop)])
    plt.tick_params(axis='x', which='both', bottom='off', top='off',
                    labelbottom='off')
    plt.tick_params(axis='y', which='both', left='off', right='off',
                    labelleft='off')
    pylab.savefig(file_name)
    plt.close()


def print_age_fitness(pop, file_name):
    """
    Prints the pareto front as a stair-step plot

    :param pop: pareto population containing individuals with fitnesses
    :param file_name: desired output file name
    """
    age_x = []
    fitness_y = []
    for indv in pop:
        age_x.append(indv.genetic_age)
        fitness_y.append(indv.fitness[0])

    ax1 = plt.subplot(111)
    ax1.scatter(age_x, fitness_y)
    ax1.set_xlim(left=0)
    # plt.gca().set_ylim(top=50)
    # plt.gca().set_ylim(bottom=0)
    ax1.set_yscale("log")
    pylab.savefig(file_name)
    plt.close()


def print_pareto(pop, file_name):
    """
    Prints the pareto front as a stair-step plot

    :param pop: pareto population containing individuals with fitnesses
    :param file_name: desired output file name
    """
    pareto_x = []
    pareto_y = []
    for indv in pop:
        pareto_x.append(indv.fitness[1])
        pareto_y.append(indv.fitness[0])
    plt.step(pareto_x, pareto_y)
    plt.scatter(pareto_x, pareto_y)
    pylab.savefig(file_name)
    plt.close()


def print_1d_best_soln(X, Y, eval_func, fitness_metric, file_name):
    """
    Prints the solution of a 1-d problem y=f(x)

    :param X: independent data
    :param Y: true solution
    :param eval_func: evaluation function to get y_estimated as function of X
    :param file_name: desired output filename
    """
    y_est = eval_func(eval_x=X, eval_y=Y, fitness_metric=fitness_metric,
                      x=X, y=Y)
    plt.scatter(X.flatten(), Y.flatten(), c='blue', label='data')
    plt.plot(X.flatten(), y_est.flatten(), c='red', label='best_fit')
    plt.legend()
    pylab.savefig(file_name)
    plt.close()
