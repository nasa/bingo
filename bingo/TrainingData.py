"""
This module contains the definition of various data containers that store
training data for bingo.
"""

import abc
import warnings
import logging

import numpy as np

from .Utils import calculate_partials

LOGGER = logging.getLogger(__name__)


class TrainingData(object, metaclass=abc.ABCMeta):
    """
    Training Data superclass which defines the methods needed for derived
    classes
    """

    @abc.abstractmethod
    def __getitem__(self, items):
        """
        This function allows for the sub-indexing of the training data
        :param items: list of indices for the subset
        :return: must return a TrainingData object
        """
        pass

    @abc.abstractmethod
    def size(self):
        """
        gets the number of indexable points in the training data
        :return: size of the training dataset
        """
        pass


class ExplicitTrainingData(TrainingData):
    """
    ExplicitTrainingData: Training data of this type contains an input array of
    data (x)  and an output array of data (y).  Both must be 2 dimensional
    numpy arrays
    """

    def __init__(self, x, y):
        """
        Initialization of explicit training data
        :param x: numpy array, dependent variable
        :param y: numpy array, independent variable
        """
        if x.ndim == 1:
            warnings.warn("Explicit training x should be 2 dim array, " +
                          "reshaping array")
            x = x.reshape([-1, 1])
        if x.ndim > 2:
            raise ValueError('Explicit training x should be 2 dim array')

        if y.ndim == 1:
            warnings.warn("Explicit training y should be 2 dim array, " +
                          "reshaping array")
            y = y.reshape([-1, 1])
        if y.ndim > 2:
            raise ValueError('Explicit training y should be 2 dim array')

        self.x = x
        self.y = y

    def __getitem__(self, items):
        """
        gets a subset of the ExplicitTrainingData
        :param items: list or int, index (or indices) of the subset
        :return: an ExplicitTrainingData
        """
        temp = ExplicitTrainingData(self.x[items, :], self.y[items, :])
        return temp

    def size(self):
        """
        gets the length of the first dimension of the data
        :return: indexable size
        """
        return self.x.shape[0]


class ImplicitTrainingData(TrainingData):
    """
    ImplicitTrainingData: Training data of this type contains an input array of
    data (x)  and its time derivative (dx_dt).  Both must be 2 dimensional
    numpy arrays
    """

    def __init__(self, x, dx_dt=None):
        """
        Initialization of implicit training data
        :param x: numpy array, dependent variable
        :param dx_dt: numpy array,  time derivative of x
        """
        if x.ndim == 1:
            warnings.warn("Explicit training x should be 2 dim array, " +
                          "reshaping array")
            x = x.reshape([-1, 1])
        if x.ndim > 2:
            raise ValueError('Explicit training x should be 2 dim array')

        # dx_dt not provided
        if dx_dt is None:
            x, dx_dt, _ = calculate_partials(x)

        # dx_dt is provided
        else:
            if dx_dt.ndim != 2:
                raise ValueError('Implicit training dx_dt must be 2 dim array')

        self.x = x
        self.dx_dt = dx_dt

    def __getitem__(self, items):
        """
        gets a subset of the ExplicitTrainingData
        :param items: list or int, index (or indices) of the subset
        :return: an ExplicitTrainingData
        """
        temp = ImplicitTrainingData(self.x[items, :], self.dx_dt[items, :])
        return temp

    def size(self):
        """
        gets the length of the first dimension of the data
        :return: indexable size
        """
        return self.x.shape[0]


class PairwiseAtomicTrainingData(TrainingData):
    """
    PairwiseAtomicTrainingData: Training data of this type contains distances
    (r) between ataoms in several configurations. each configuration has an
    associated potential energy.  The r values beloning to each configuration
    are bounded by configuration limits (config_lims_r)
    """

    def __init__(self, potential_energy, configurations=None, r_list=None,
                 config_lims_r=None):
        """
        Initializes PairwiseAtomicTrainingData
        :param potential_energy: 1d array, potential energy for each
                                 configuration
        :param configurations: list of tuples (structure, period, r_cutoff),
                               where the structure is an array of x,y,z
                               locations of atoms. period is the periodic size
                               of the configuration. rcutoff is the cutoff
                               distance after which the pairwise interaction
                               does not effect the potential energy.
        :param r_list: 2d array, list of all pairwise distances
        :param config_lims_r: 1d array, bounds of all of the r_indices
                              corresponding to each configuration
        """

        if potential_energy.ndim > 1:
            warnings.warn("Pairwise atomic training data: potential energy " +\
                          "should be1 dim array, flattening array")
            potential_energy = potential_energy.flatten()

        if configurations is not None:
            if potential_energy.shape[0] != len(configurations):
                raise ValueError("Pairwise atomic training data: potential " +\
                                 "energy and configurations are different " +\
                                 "sizes")
            r_list = []
            config_lims_r = [0]
            for (structure, periodic_size, r_cutoff) in configurations:
                # make radius list
                natoms = structure.shape[0]
                rcutsq = r_cutoff**2
                for atomi in range(0, natoms):
                    xtmp = structure[atomi, 0]
                    ytmp = structure[atomi, 1]
                    ztmp = structure[atomi, 2]
                    for atomj in range(atomi + 1, natoms):
                        delx = structure[atomj, 0] - xtmp
                        while delx > 0.5 * periodic_size:
                            delx -= periodic_size
                        while delx < -0.5 * periodic_size:
                            delx += periodic_size
                        dely = structure[atomj, 1] - ytmp
                        while dely > 0.5 * periodic_size:
                            dely -= periodic_size
                        while dely < -0.5 * periodic_size:
                            dely += periodic_size
                        delz = structure[atomj, 2] - ztmp
                        while delz > 0.5 * periodic_size:
                            delz -= periodic_size
                        while delz < -0.5 * periodic_size:
                            delz += periodic_size

                        rsq = delx * delx + dely * dely + delz * delz
                        if rsq <= rcutsq:
                            r_list.append(np.sqrt(rsq))
                config_lims_r.append(len(r_list))
            r_list = np.array(r_list).reshape([-1, 1])
            config_lims_r = np.array(config_lims_r)

        elif potential_energy is None or \
                r_list is None or config_lims_r is None:
            raise RuntimeError('Invalid construction of PairwiseAtomicData')

        self.r = r_list
        self.config_lims_r = config_lims_r
        self.potential_energy = potential_energy

    def __getitem__(self, items):
        """
        gets a subset of the ParwiseAtomicTrainingData
        :param items: list or int, index (or indices) of the subset
        :return: an ExplicitTrainingData
        """

        r_inds = []
        new_config_lims_r = [0]
        for i in items:
            r_inds += range(self.config_lims_r[i], self.config_lims_r[i+1])
            new_config_lims_r.append(len(r_inds))
        new_config_lims_r = np.array(new_config_lims_r)

        new_potential_energy = self.potential_energy[items]
        temp = PairwiseAtomicTrainingData(
            potential_energy=new_potential_energy,
            r_list=self.r[r_inds, :],
            config_lims_r=new_config_lims_r)
        return temp

    def size(self):
        """
        gets the number of configurations
        :return: indexable size
        """
        return self.potential_energy.shape[0]
