"""Base classes for genetic individuals

This module has the base classes for genetic individuals in bingo evolutionary
analyses.  Extension of bingo can be performed by developing subclasses of the
classes herein.
"""

import copy
from abc import ABCMeta, abstractmethod


class GeneticIndividual(metaclass=ABCMeta):
    """A genetic individual

    This class is the base of a genetic individual in bingo evolutionary
    analyses.

    Attributes
    ----------
    fitness
    genetic_age : int
                  age of the oldest component of the genetic material in the
                  individual
    fit_set : bool
              Whether the fitness has been calculated for the individual
    """
    def __init__(self):
        self.genetic_age = 0
        self._fitness = None
        self.fit_set = False

    @property
    def fitness(self):
        """numeric or tuple of numeric: The fitness of the individual"""
        return self._fitness

    @fitness.setter
    def fitness(self, fitness):
        self._fitness = fitness
        self.fit_set = True

    def copy(self):
        """copy

        Returns
        -------
            A deep copy of self
        """
        return copy.deepcopy(self)

    @abstractmethod
    def __str__(self):
        """String conversion of individual

        Returns
        -------
        str
            individual string form
        """
        raise NotImplementedError

    @abstractmethod
    def needs_local_optimization(self):
        """Does the individual need local optimization

        Returns
        -------
        bool
            Individual needs optimization
        """
        raise NotImplementedError

    @abstractmethod
    def get_number_local_optimization_params(self):
        """Get number of parameters in local optimization

        Returns
        -------
        int
            number of paramneters to be optimized
        """
        raise NotImplementedError

    @abstractmethod
    def set_local_optimization_params(self, params):
        """Set local optimization parameters

        Parameters
        ----------
        params : list-like of numeric
                 Values to set the parameters
        """
        raise NotImplementedError


class EquationIndividual(GeneticIndividual, metaclass=ABCMeta):
    """Base representation of an equation

    This class is the base of a equations used in symbolic regression alayses
    in bingo.
    """

    @abstractmethod
    def evaluate_equation_at(self, x):
        """Evaluate the equation.

        Get value of the equation at points x.

        Parameters
        ----------
        x : MxD array of numeric.
            Values at which to evaluate the equations. D is the number of
            dimensions in x and M is the number of data points in x.

        Returns
        -------
        Mx1 array of numeric
            :math:`f(x)`
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_equation_with_x_gradient_at(self, x):
        """Evaluate equation and get its derivatives.

        Get value the equation at x and its gradient with respect to x.

        Parameters
        ----------
        x : MxD array of numeric.
            Values at which to evaluate the equations. D is the number of
            dimensions in x and M is the number of data points in x.

        Returns
        -------
        tuple(Mx1 array of numeric, MxD array of numeric)
            :math:`f(x)` and :math:`df(x)/dx_i`
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_equation_with_local_opt_gradient_at(self, x):
        """Evaluate equation and get its derivatives.

        Get value the equation at x and its gradient with respect to
        optimization parameters.

        Parameters
        ----------
        x : MxD array of numeric.
            Values at which to evaluate the equations. D is the number of
            dimensions in x and M is the number of data points in x.

        Returns
        -------
        tuple(Mx1 array of numeric, MxL array of numeric)
            :math:`f(x)` and :math:`df(x)/dc_i`. L is the number of
            optimization paremeters.
        """
        raise NotImplementedError

    @abstractmethod
    def get_latex_string(self):
        """Latex conversion of individual

        Returns
        -------
        str
            equation in latex interpretable form
        """
        raise NotImplementedError

    @abstractmethod
    def get_console_string(self):
        """Simplified string conversion of equation

        Returns
        -------
        str
            compact form of equation
        """
        raise NotImplementedError

    @abstractmethod
    def get_complexity(self):
        """Calculate complexity of equation.

        Returns
        -------
        numeric
            complexity measure of equation
        """
        raise NotImplementedError
