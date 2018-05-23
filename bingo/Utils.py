"""
Utils contains useful utility functions for doing and testing symbolic
regression problems in the bingo package
"""
import math

import numpy as np


def calculate_partials(X):
    """
    Calculate derivatves with respect to time (first dimension).

    :param X: 2d numpy array in for which derivatives will be calculated in the
              first diminsion. Distinct trajectories can be specified by
              separating the datasets within X by rows of np.nan
    :return: updated X array and corresponding time derivatives
    """
    # find splits
    break_points = np.where(np.any(np.isnan(X), 1))[0].tolist()
    break_points.append(X.shape[0])

    start = 0
    for end in break_points:
        x_seg = np.copy(X[start:end, :])
        # calculate time derivs using filter
        time_deriv = np.empty(x_seg.shape)
        for i in range(x_seg.shape[1]):
            time_deriv[:, i] = savitzky_golay_gram(x_seg[:, i], 7, 3, 1)
        # remove edge effects
        time_deriv = time_deriv[3:-4, :]
        x_seg = x_seg[3:-4, :]

        if start is 0:
            x_all = np.copy(x_seg)
            time_deriv_all = np.copy(time_deriv)
            inds_all = np.arange(start+3, end-4)
        else:
            x_all = np.vstack((x_all, np.copy(x_seg)))
            time_deriv_all = np.vstack((time_deriv_all,
                                        np.copy(time_deriv)))

            inds_all = np.hstack((inds_all,
                                  np.arange(start+3, end-4)))
        start = end + 1

    return x_all, time_deriv_all, inds_all


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """
    Smooth (and optionally differentiate) data with a Savitzky-Golay filter
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.

    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.

    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688

    :param y: array_like, shape (N,)
        the values of the time history of the signal.
    :param window_size: int
        the length of the window. Must be an odd integer number.
    :param order: int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    :param deriv: int
        the order of the derivative to compute (default = 0 means only
        smoothing)
    :return: ys : ndarray, shape (N)
            the smoothed signal (or it's n-th derivative).
    """
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError as __:
        raise ValueError(
            "window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError(
            "window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError(
            "window_size is too small for the polynomials order")
    order_range = list(range(order + 1))
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in
                range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * math.factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


def savitzky_golay_gram(y, window_size, order, deriv=0):
    """
    Smooth (and optionally differentiate) data with a Savitzky-Golay filter
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.

    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.

    A Gram polynomial version of this is used to have better estimates near the
    boundaries of the data.

    .. [3] P.A. Gorry, General Least-Squares Smoothing and Differentiation by
       the Convolution (Savitzky-Golay) Method. Analytical Chemistry, 1990, 62,
       pp 570-573

    :param y: array_like, shape (N,)
        the values of the time history of the signal.
    :param window_size: int
        the length of the window. Must be an odd integer number.
    :param order: int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    :param deriv: int
        the order of the derivative to compute (default = 0 means only
        smoothing)
    :return: ys : ndarray, shape (N)
            the smoothed signal (or it's n-th derivative).
    """
    n = order  # order
    m = np.int((window_size - 1)/2)  # 2m + 1 = size of filter
    s = deriv  # derivative order

    def GenFact(a, b):
        """Generalized factorial"""
        g_f = 1
        for j in range(a-b+1, a+1):
            g_f *= j
        return g_f

    def GramPoly(gp_i, gp_m, gp_k, gp_s):
        """
        Calculates the Gram Polynomial (gp_s=0) or its gp_s'th derivative
        evaluated at gp_i, order gp_k, over 2gp_m+1 points
        """
        if gp_k > 0:
            gram_poly = (4. * gp_k - 2.) / (gp_k * (2. * gp_m - gp_k + 1.)) * \
                        (gp_i * GramPoly(gp_i, gp_m, gp_k - 1, gp_s) + \
                         gp_s * GramPoly(gp_i, gp_m, gp_k - 1, gp_s - 1)) - \
                        ((gp_k - 1.) * (2. * gp_m + gp_k)) / \
                        (gp_k * (2. * gp_m - gp_k + 1.)) * \
                        GramPoly(gp_i, gp_m, gp_k - 2, gp_s)

        else:
            if gp_k == 0 and gp_s == 0:
                gram_poly = 1.
            else:
                gram_poly = 0.
        return gram_poly

    def GramWeight(gw_i, gw_t, gw_m, gw_n, gw_s):
        """
        Calculate the weight og the gw_i'th data point for the gw_t'th
        Least-Square point of the gw_s'th derivative over 2gw_m+1 points,
        order gw_n
        """
        weight = 0
        for k in range(gw_n + 1):
            weight += (2. * k + 1.) * GenFact(2 * gw_m, k) / \
                      GenFact(2 * gw_m + k + 1, k + 1) * \
                      GramPoly(gw_i, gw_m, k, 0) * \
                      GramPoly(gw_t, gw_m, k, gw_s)
        return weight

    # fill weights
    weights = np.empty((2*m+1, 2*m+1))
    for i in range(-m, m+1):
        for t in range(-m, m+1):
            weights[i + m, t + m] = GramWeight(i, t, m, n, s)

    # do convolution
    y_len = len(y)
    f = np.empty(y_len)
    for i in range(y_len):
        if i < m:
            y_center = m
            w_ind = i
        elif y_len - i <= m:
            y_center = y_len - m - 1
            w_ind = 2*m+1 - (y_len-i)
        else:
            y_center = i
            w_ind = m
        f[i] = 0
        for k in range(-m, m+1):
            f[i] += y[y_center + k] * weights[k + m, w_ind]

    return f


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
