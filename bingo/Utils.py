"""
Utils contains useful utility functions for doing and testing symbolic
regression problems in the bingo package
"""
import math

import numpy as np


def calculate_partials(X):
    """ calculate partial derivatves with respect to time (index) """
    # find splits
    break_points = np.where(np.any(np.isnan(X), 1))[0].tolist()
    break_points.append(X.shape[0])

    start = 0
    for end in break_points:
        x_seg = np.copy(X[start:end, :])
        # calculate time derivs using filter
        time_deriv = np.empty(x_seg.shape)
        for i in range(x_seg.shape[1]):
            time_deriv[:, i] = savitzky_golay(x_seg[:, i], 7, 3, 1)
        # remove edge effects
        time_deriv = time_deriv[3:-4, :]
        x_seg = x_seg[3:-4, :]

        if start is 0:
            x_all = np.copy(x_seg)
            time_deriv_all = np.copy(time_deriv)
        else:
            x_all = np.vstack((x_all, np.copy(x_seg)))
            time_deriv_all = np.vstack((time_deriv_all,
                                        np.copy(time_deriv)))
        start = end + 1

    return x_all, time_deriv_all


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """
    Smooth (and optionally differentiate) data with a Savitzky-Golay filter
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only
        smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
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


def snake_walk():
    """snake walk for independent data"""
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
