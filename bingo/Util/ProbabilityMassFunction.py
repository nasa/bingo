"""
This module implements a probability mass function from which single samples
can be drawn
"""
import logging
import numpy as np

LOGGER = logging.getLogger(__name__)


class ProbabilityMassFunction(object):
    """
    The ProbabilityMassFunction (PMF) class is designed to allow for easy
    creation and use of a probability mass function.  Items and associated
    probability weights are given. Samples (items) can then be drawn from the
    pmf according to their relative weights.
    """

    def __init__(self, items=None, weights=None):
        """
        Initialize a PMF with its starting items and their associated weights.

        :param items: The items in the PMF.
        :type items: list
        :param weights: The relative weights of the items. The default is even
                        weighting.
        :type weights: list-like, numeric
        """
        if items is None:
            items = []

        self._is_init_param_listlike(items)
        self.items = items

        if weights is None:
            weights = self._get_default_weights()

        self._is_init_param_listlike(weights)
        self._is_weights_same_size_as_items(weights)
        self.total_weight, self.normalized_weights = \
            self._normalize_weights(weights)

    @staticmethod
    def _is_init_param_listlike(init_param):
        if not hasattr(init_param, "__len__") or isinstance(init_param, str):
            LOGGER.error("Initialization of ProbabilityMassFunction with "
                         "non-listlike parameters")
            LOGGER.error(str(init_param))
            raise ValueError

    def _get_default_weights(self):
        n_items = len(self.items)
        if n_items > 0:
            weights = [1.0 / n_items] * n_items
        else:
            weights = []
        return weights

    def _is_weights_same_size_as_items(self, weights):
        if len(weights) != len(self.items):
            LOGGER.error("Initialization of ProbabilityMassFunction with "
                         "items and weights of different dimensions")
            LOGGER.error("items = " + str(self.items))
            LOGGER.error("weights = " + str(weights))
            raise ValueError

    @staticmethod
    def _normalize_weights(weights):
        total_weight = ProbabilityMassFunction._calculate_total_weight(weights)
        normalized_weights = np.array(weights) / total_weight
        ProbabilityMassFunction._check_valid_weights(normalized_weights,
                                                     weights)
        return total_weight, normalized_weights

    @staticmethod
    def _check_valid_weights(normalized_weights, weights):
        if len(normalized_weights) > 0:
            if not np.isclose(np.sum(normalized_weights), 1.0) or \
                            np.min(normalized_weights) < 0.0:
                LOGGER.error("Invalid weights encountered in "
                             "ProbabilityMassFunction")
                LOGGER.error("weights = " + str(weights))
                raise ValueError

    @staticmethod
    def _calculate_total_weight(weights):
        try:
            total_weight = np.sum(weights)
        except TypeError:
            LOGGER.error("Initialization of ProbabilityMassFunction with "
                         "non-numeric weights")
            LOGGER.error("weights = " + str(weights))
            raise TypeError
        return total_weight

    def add_item(self, new_item, new_weight=None):
        """
        Adds a single item to the PMF.

        :param new_item: The item to be added.
        :type new_item: any
        :param new_weight: The weight associated with the item. The default is
                           the average weight of the other items.
        :type new_weight: numeric
        """
        self.items.append(new_item)

        if new_weight is None:
            new_weight = self._get_mean_current_weight()

        weights = self.total_weight*self.normalized_weights
        weights = np.append(weights, new_weight)

        self.total_weight, self.normalized_weights = \
            self._normalize_weights(weights)

    def _get_mean_current_weight(self):
        if len(self.normalized_weights) is 0:
            return 1.0
        else:
            return self.total_weight/len(self.normalized_weights)

    def draw_sample(self):
        """
        Draw a random sample from the PMF according to the probabilities
        associated with weighting of items.
        :return: a single item
        :rtype: any
        """
        return np.random.choice(self.items, 1, p=self.normalized_weights)[0]
