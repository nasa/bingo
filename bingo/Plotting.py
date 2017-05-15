"""
This module contains the functions for general plotting purposes
"""
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pylab


def print_latex(pop, file_name):
    """
    makes pyplot figure with latex table on it: the table consists of equations
     describing the populations
    :param pop: population containing individuals with fitnesses and
                latexstring()
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


def print_pareto(pop, file_name):
    """
    prints the pareto front as a stair-step plot
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


def print_1d_best_soln(X, Y, eval_func, file_name):
    """
    Prints the solution of a 1-d problem y=f(x)
    :param X: independent data
    :param Y: true solution
    :param eval_func: evaluation function to get y_estimated as function of X
    :param file_name: desired output filename
    """
    y_est = []
    for x in X:
        y_est.append(eval_func(x))
    plt.scatter(X, Y)
    plt.plot(X, y_est)
    pylab.savefig(file_name)
    plt.close()
