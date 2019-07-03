"""Implicit Symbolic Regression

Explicit symbolic regression is the search for a function, f, such that
f(x) = constant.  One of the most difficult part of this task is avoiding
trivial solutions like f(x) = 0*x.

The classes in this module encapsulate the parts of bingo evolutionary analysis
that are unique to implicit symbolic regression. Namely, these classes are
appropriate fitness evaluators, a corresponding training data container, and
two helper functions.
"""
import warnings
import logging

import numpy as np

from ..Base.FitnessFunction import VectorBasedFunction
from ..Base.TrainingData import TrainingData

LOGGER = logging.getLogger(__name__)


class ImplicitRegression(VectorBasedFunction):
    """ Implicit Regression, version 2

    Fitness of this metric is related to the cos of angle between between
    df_dx(x) and dx_dt. df_dx(x) is calculated through derivatives of the input
    Equation individual at training_data.x. dx_dt is from training_data.dx_dt.

    Different normalization and error checking are available.

    Parameters
    ----------
    training_data :
                   data that is used in fitness evaluation.  Must have
                   attributes x and dx_dt.
    required_params : int
                      (optional) minimum number of nonzero components of dot
    normalize_dot : bool
                    normalize the terms in the dot product (default = False)
    """
    def __init__(self, training_data, required_params=None,
                 normalize_dot=False):
        super().__init__(training_data)
        self._required_params = required_params
        self._normalize_dot = normalize_dot

    def evaluate_fitness_vector(self, individual):
        self.eval_count += 1
        _, df_dx = individual.evaluate_equation_with_x_gradient_at(
            x=self.training_data.x)

        dot_product = self._do_dfdx_dot_dxdt(df_dx)

        if self._required_params is not None:
            if not self._enough_parameters_used(dot_product):
                return np.full((self.training_data.x.shape[0],), np.inf)

        denominator = np.sum(np.abs(dot_product), axis=1)
        normalized_fitness = np.sum(dot_product, axis=1) / denominator
        normalized_fitness[~np.isfinite(denominator)] = np.inf
        return normalized_fitness

    def _enough_parameters_used(self, dot_product):
        n_params_used = (abs(dot_product) > 1e-16).sum(1)
        enough_params_used = np.any(n_params_used >= self._required_params)
        return enough_params_used

    def _do_dfdx_dot_dxdt(self, df_dx):
        left_dot = df_dx
        right_dot = self.training_data.dx_dt
        if self._normalize_dot:
            left_dot = self._normalize_by_row(left_dot)
            right_dot = self._normalize_by_row(right_dot)
        return left_dot * right_dot

    @staticmethod
    def _normalize_by_row(array):
        return array / np.linalg.norm(array, axis=1).reshape((-1, 1))


class ImplicitRegressionSchmidt(VectorBasedFunction):
    """ Implicit Regression, Adapted from Schmidt and Lipson papers

    Fitness in this method is the difference of partial derivatives pairs
    calculated with the data and the input Equation individual.

    Parameters
    ----------
    training_data :
                   data that is used in fitness evaluation.  Must have
                   attributes x and dx_dt.

    Notes
    -----
    This may not be a correct implementation of this algorithm.  Importantly,
    it couldn't reproduce the  results in the papers.
    """
    def evaluate_fitness_vector(self, individual):
        _, df_dx = individual.evaluate_equation_with_x_gradient_at(
            x=self.training_data.x)

        num_parameters = self.training_data.x.shape[1]
        worst_fitness = 0
        diff_worst = np.full((num_parameters, ), np.inf)
        for i in range(num_parameters):
            for j in range(num_parameters):
                if i != j:
                    df_dxi = np.copy(df_dx[:, i])
                    df_dxj = np.copy(df_dx[:, j])
                    dxi_dxj_2 = (self.training_data.dx_dt[:, i] /
                                 self.training_data.dx_dt[:, j])
                    for k in range(num_parameters):
                        if k not in (i, j):
                            df_dxj += df_dx[:, k] * \
                                      self.training_data.dx_dt[:, k] / \
                                      self.training_data.dx_dt[:, j]

                    dxi_dxj_1 = df_dxj / df_dxi
                    diff = np.log(1. + np.abs(dxi_dxj_1 + dxi_dxj_2))
                    fit = np.mean(diff)
                    if np.isfinite(fit) and fit > worst_fitness:
                        diff_worst = np.copy(diff)
                        worst_fitness = fit
        return diff_worst


class ImplicitTrainingData(TrainingData):
    """
    ImplicitTrainingData: Training data of this type contains an input array of
    data (x)  and its time derivative (dx_dt).  Both must be 2 dimensional
    numpy arrays

    Parameters
    ----------
     x : 2D numpy array
         independent variable
     dx_dt : 2D numpy array
             (optional) time derivative of x.  If not is provided dx_dt is
             calculated from x.
    """
    def __init__(self, x, dx_dt=None):
        if x.ndim == 1:
            warnings.warn("Explicit training x should be 2 dim array, " +
                          "reshaping array")
            x = x.reshape([-1, 1])
        if x.ndim > 2:
            raise ValueError('Explicit training x should be 2 dim array')

        if dx_dt is None:
            x, dx_dt, _ = calculate_partials(x)
        else:
            if dx_dt.ndim != 2:
                raise ValueError('Implicit training dx_dt must be 2 dim array')

        self.x = x
        self.dx_dt = dx_dt

    def __getitem__(self, items):
        """gets a subset of the ExplicitTrainingData

        Parameters
        ----------
         items : list or int
                 index (or indices) of the subset

        Returns
        -------
         ExplicitTrainingData :
                                a subset
        """
        temp = ImplicitTrainingData(self.x[items, :], self.dx_dt[items, :])
        return temp

    def __len__(self):
        """gets the length of the first dimension of the data

        Returns
        -------
         int :
                index-able size
        """
        return self.x.shape[0]


def calculate_partials(X):
    """Calculate derivatves with respect to time (first dimension).

    Parameters
    ----------
     X : 2d numpy array
         array for which derivatives will be calculated in the first diminsion.
         Distinct trajectories can be specified by separating the datasets
         within X by rows of np.nan

    Returns
    -------
    2d numpy array :
        updated X array and corresponding time derivatives
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

        if start == 0:
            x_all = np.copy(x_seg)
            time_deriv_all = np.copy(time_deriv)
            inds_all = np.arange(start + 3, end - 4)
        else:
            x_all = np.vstack((x_all, np.copy(x_seg)))
            time_deriv_all = np.vstack((time_deriv_all,
                                        np.copy(time_deriv)))

            inds_all = np.hstack((inds_all,
                                  np.arange(start + 3, end - 4)))
        start = end + 1

    return x_all, time_deriv_all, inds_all


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
    polynomial of high order over a odd-sized window centered a
    the point.

    A Gram polynomial version of this is used to have better estimates near the
    boundaries of the data.

    Parameters
    ----------
     y : array_like, shape (N,)
         the values of the time history of the signal.
     window_size : int
                   the length of the window. Must be an odd integer number.
     order : int
             the order of the polynomial used in the filtering.
             Must be less then `window_size` - 1.
     deriv : int
             the order of the derivative to compute (default = 0 means only
             smoothing)

    Returns
    -------
     ys : ndarray, shape (N)
          the smoothed signal (or it's n-th derivative).

    References
    ----------
    .. [3] P.A. Gorry, General Least-Squares Smoothing and Differentiation by
       the Convolution (Savitzky-Golay) Method. Analytical Chemistry, 1990, 62,
       pp 570-573
    """
    n_order = order
    m_half_filter_size = np.int((window_size - 1) / 2)  # 2m + 1 = filter size
    s_derivative_order = deriv

    def generalized_factorial(a, b):
        """Generalized factorial"""
        g_f = 1
        for j in range(a - b + 1, a + 1):
            g_f *= j
        return g_f

    def gram_polynomial(gp_i, gp_m, gp_k, gp_s):
        """
        Calculates the Gram Polynomial (gp_s=0) or its gp_s'th derivative
        evaluated at gp_i, order gp_k, over 2gp_m+1 points
        """
        if gp_k > 0:
            gram_poly = (4. * gp_k - 2.) / (gp_k * (2. * gp_m - gp_k + 1.)) * \
                        (gp_i * gram_polynomial(gp_i, gp_m, gp_k - 1, gp_s) +
                         gp_s * gram_polynomial(gp_i, gp_m, gp_k - 1,
                                                gp_s - 1)) - \
                        ((gp_k - 1.) * (2. * gp_m + gp_k)) / \
                        (gp_k * (2. * gp_m - gp_k + 1.)) * \
                        gram_polynomial(gp_i, gp_m, gp_k - 2, gp_s)

        else:
            if gp_k == 0 and gp_s == 0:
                gram_poly = 1.
            else:
                gram_poly = 0.
        return gram_poly

    def gram_weight(gw_i, gw_t, gw_m, gw_n, gw_s):
        """
        Calculate the weight og the gw_i'th data point for the gw_t'th
        Least-Square point of the gw_s'th derivative over 2gw_m+1 points,
        order gw_n
        """
        weight = 0
        for k in range(gw_n + 1):
            weight += (2. * k + 1.) * generalized_factorial(2 * gw_m, k) / \
                      generalized_factorial(2 * gw_m + k + 1, k + 1) * \
                      gram_polynomial(gw_i, gw_m, k, 0) * \
                      gram_polynomial(gw_t, gw_m, k, gw_s)
        return weight

    # fill weights
    weights = np.empty((2 * m_half_filter_size + 1, 2 * m_half_filter_size + 1))
    for i in range(-m_half_filter_size, m_half_filter_size + 1):
        for t in range(-m_half_filter_size, m_half_filter_size + 1):
            weights[i + m_half_filter_size, t + m_half_filter_size] = \
                gram_weight(i, t, m_half_filter_size, n_order,
                            s_derivative_order)

    # do convolution
    y_len = len(y)
    f = np.empty(y_len)
    for i in range(y_len):
        if i < m_half_filter_size:
            y_center = m_half_filter_size
            w_ind = i
        elif y_len - i <= m_half_filter_size:
            y_center = y_len - m_half_filter_size - 1
            w_ind = 2 * m_half_filter_size + 1 - (y_len - i)
        else:
            y_center = i
            w_ind = m_half_filter_size
        f[i] = 0
        for k in range(-m_half_filter_size, m_half_filter_size + 1):
            f[i] += y[y_center + k] * weights[k + m_half_filter_size, w_ind]

    return f
