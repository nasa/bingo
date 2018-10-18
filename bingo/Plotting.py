"""
This module contains the functions for general plotting purposes
"""
import matplotlib.pyplot as plt
import pylab
import numpy as np


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
        plt.text(0.32, i, "$%s$" % indv.get_latex_string())
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


def print_1d_best_soln(X, Y, eval_func, file_name):
    """
    Prints the solution of a 1-d problem y=f(x)

    :param X: independent data
    :param Y: true solution
    :param eval_func: evaluation function to get y_estimated as function of X
    :param file_name: desired output filename
    """
    y_est = eval_func(X)
    plt.scatter(X.flatten(), Y.flatten(), c='blue', label='data')
    plt.plot(X.flatten(), y_est.flatten(), c='red', label='best_fit')
    plt.legend()
    pylab.savefig(file_name)
    plt.close()


def print_convergence_stats(ages, fitnesses, file_name, title=None,
                            convergence=None):
    """
    Plots the convergence stats for a set of bingo simulations.  the input data
    is in the format that is obtained by post processing a log.txt file.

    :param ages: ages of simulations, 0's indicate converegence
    :param fitnesses: fitnesses of each simulation at the corresponding age
    :param file_name: name of image file to be saved
    :param title: title to be shown at the top of the plot
    :return:
    """
    _, ax1 = plt.subplots(figsize=(11, 8.5))
    if title is not None:
        ax1.set_title(title)
    ax1.set_yscale("log", nonposx='clip')
    width = 0.7*np.max(ages)/ages.shape[0]
    ax1.boxplot(fitnesses.transpose(),
                positions=np.max(ages, axis=1),
                widths=width,
                showmeans=True,
                medianprops=dict(color='red', linewidth=2))
    # ax1.violinplot(fitnesses.transpose(),
    #                positions=np.max(ages, axis=1),
    #                widths=width,
    #                showmeans=True)
    if convergence is not None:
        ax1.plot([np.min(ages), np.max(ages)],
                 [convergence, convergence],
                 'm--')
        ax1.set_ylabel('fitness (convergence = %e)' % convergence)
    else:
        ax1.set_ylabel('fitness')
    ax1.set_xlabel('age')
    nskip = int((ages.shape[0]+1)/10)
    ax1.set_xticks(np.max(ages, axis=1)[nskip-1::nskip])
    ax1.set_xticklabels(np.max(ages, axis=1)[nskip-1::nskip])

    success = 1 - (np.count_nonzero(ages, axis=1))/ages.shape[1]
    ax2 = ax1.twinx()
    ax2.plot(np.max(ages, axis=1), 100*success, 'b')
    ax2.set_ylabel('convergence (%%), %d repeats' % ages.shape[1], color='b')
    ax2.tick_params('y', colors='b')
    ax2.set_ylim(0, 100)
    ax2.set_xlim(np.min(ages), np.max(ages))

    plt.tight_layout()
    pylab.savefig(file_name)
    plt.close()
