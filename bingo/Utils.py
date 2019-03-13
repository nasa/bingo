"""
Utils contains useful utility functions for doing and testing symbolic
regression problems in the bingo package
"""
import math

import numpy as np


def snake_walk():
    """
    Generates 2-d dataset which looks like a snake wiggling back and forth

    :returns: 2d numpy array for data
    """
    n_samps = 200
    step_size = 0.2
    x_true = np.zeros([n_samps, 2])
    for i in range(n_samps):
        if i is 0:
            x_true[i, 0] = step_size * 0.5
            x_true[i, 1] = step_size / 2
            direction = step_size
        else:
            if i % (n_samps//10) == 0:
                direction *= -1
                x_true[i, 0] = -direction
                x_true[i, 1] += step_size
            x_true[i, 0] += x_true[i - 1, 0] + direction
            x_true[i, 1] += x_true[i - 1, 1]

    x_true[:, 0] = savitzky_golay(x_true[:, 0], 21, 2, 0)
    x_true[:, 1] = savitzky_golay(x_true[:, 1], 21, 2, 0)
    return x_true


def post_process_logfile(filename):
    """ does same basic post processing of log files """
    with open(filename) as log_file:
        # get info from each sim_run
        prev_line = ""
        sim_runs = []
        sim_age = []
        sim_fitness = []
        n_sims = 0
        longest_sim = 0
        for i, line in enumerate(log_file):
            # parse previous line
            tokens = prev_line.split()
            if i > 0 and len(tokens) > 0:
                age = int(tokens[0])
                sim_age.append(age)
                fitness = float(tokens[2])
                sim_fitness.append(fitness)

                # check for ending run
                run_ending = False
                if len(line.split()) > 0:
                    if age >= int(line.split()[0]):
                        run_ending = True
                else:
                    run_ending = True

                # save run info if run is ending
                if run_ending:
                    n_sims += 1
                    if len(sim_age) > longest_sim:
                        longest_sim = len(sim_age)
                    sim_runs.append((np.array(sim_age), np.array(sim_fitness)))
                    sim_age = []
                    sim_fitness = []

            # save line
            prev_line = line

        # do work on last line
        tokens = prev_line.split()
        if len(tokens) > 0:
            age = int(tokens[0])
            sim_age.append(age)
            fitness = float(tokens[2])
            sim_fitness.append(fitness)
        if len(sim_age) >= longest_sim:  # NOTE this assumes at least 1 failure
            n_sims += 1
            sim_runs.append((np.array(sim_age), np.array(sim_fitness)))

    # pack all sims into set arrays
    ages = np.empty((longest_sim, n_sims), np.int)
    fitnesses = np.empty((longest_sim, n_sims))
    for i, (age, fit) in enumerate(sim_runs):
        ages[:len(age), i] = age
        ages[len(age):, i] = 0
        fitnesses[:len(fit), i] = fit
        fitnesses[len(fit):, i] = fit[-1]

    return ages, fitnesses
