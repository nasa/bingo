"""
test_sym_reg tests the standard symbolic regression nodes
"""

import random
import numpy as np


from bingo.AGraph import AGraphManipulator as agm
from bingo.AGraphCpp import AGraphCppManipulator as agcm
from bingo.AGraph import AGNodes
from bingo.FitnessPredictor import FPManipulator as fpm
from bingo.IslandManager import SerialIslandManager
from bingo.FitnessMetric import AtomicPotential


N_ISLANDS = 2
MAX_STEPS = 500
EPSILON = 1.0e-8
N_STEPS = 25


def test_atomic_potential_linear():

    a = 4.0  # units of sigma
    natoms = 4
    rcut = 2.5  # arb units
    n = 75

    def energy_func(r):
        eps = 1.5  # arb units
        return eps*r

    def force_func(r):
        eps = 1.5
        return eps

    X, Y, _ = generate_configurations(n, natoms, a, rcut,
                                      energy_func, force_func)
    compare_agraph_potential(X, Y)
    compare_agraphcpp_potential(X, Y)


def test_atomic_potential_quadratic():

    a = 4.0  # units of sigma
    natoms = 5
    rcut = 2.5  # arb units
    n = 100

    def energy_func(r):
        eps = 1.5  # arb units
        return eps*r*r

    def force_func(r):
        eps = 1.5
        return 2.0*eps*r

    X, Y, _ = generate_configurations(n, natoms, a, rcut,
                                      energy_func, force_func)
    compare_agraph_potential(X, Y)
    compare_agraphcpp_potential(X, Y)


def generate_configurations(n, natoms, a, rcut, energy_func, force_func):
    """generate sets of atomic potential configurations"""
    # Setup
    rcutsq = rcut * rcut

    # Loop through n configurations
    configs = []
    energies = []
    forces = []
    for i in range(n):
        # generate structure
        rclose = 0.770
        rclosesq = rclose * rclose
        structure = []
        for atomi in range(0, natoms):
            flag = 0
            while flag == 0:
                xtmp = random.uniform(0.0, a)
                ytmp = random.uniform(0.0, a)
                ztmp = random.uniform(0.0, a)
                flag = 1
                for atomj in range(0, len(structure)):
                    delx = structure[atomj][0] - xtmp
                    while delx > 0.5 * a: delx -= a
                    while delx < -0.5 * a: delx += a
                    dely = structure[atomj][1] - ytmp
                    while dely > 0.5 * a: dely -= a
                    while dely < -0.5 * a: dely += a
                    delz = structure[atomj][2] - ztmp
                    while delz > 0.5 * a: delz -= a
                    while delz < -0.5 * a: delz += a

                    rsq = delx * delx + dely * dely + delz * delz
                    if rsq <= rclosesq:
                        flag = 0
                        break

            structure.append([xtmp, ytmp, ztmp])

        # determine energy and force on first atom
        energy = 0.0
        force = [0.0, 0.0, 0.0]

        for atomi in range(0, natoms):
            xtmp = structure[atomi][0]
            ytmp = structure[atomi][1]
            ztmp = structure[atomi][2]
            for atomj in range(atomi + 1, natoms):
                delx = structure[atomj][0] - xtmp
                while delx > 0.5 * a: delx -= a
                while delx < -0.5 * a: delx += a
                dely = structure[atomj][1] - ytmp
                while dely > 0.5 * a: dely -= a
                while dely < -0.5 * a: dely += a
                delz = structure[atomj][2] - ztmp
                while delz > 0.5 * a: delz -= a
                while delz < -0.5 * a: delz += a

                rsq = delx * delx + dely * dely + delz * delz
                if rsq <= rcutsq:
                    energy += energy_func(np.sqrt(rsq))
                    if atomi == 0:
                        pair_f = force_func(np.sqrt(rsq))
                        force[0] -= pair_f * delx
                        force[1] -= pair_f * dely
                        force[2] -= pair_f * delz

        configs.append((np.array(structure), a, rcut))
        energies.append(energy)
        forces.append(force)

    return np.array(configs), \
           np.array(energies).reshape([-1, 1]), \
           np.array(forces).reshape([-1, 1])


def compare_agraphcpp_potential(X, Y):
    """does the comparison using agraphcpp"""

    # make solution manipulator
    sol_manip = agcm(1, 16, nloads=2)
    sol_manip.add_node_type(2)
    sol_manip.add_node_type(4)

    # make predictor manipulator
    pred_manip = fpm(32, X.shape[0])

    # make and run island manager
    islmngr = SerialIslandManager(N_ISLANDS,
                                  data_x=X,
                                  data_y=Y,
                                  solution_manipulator=sol_manip,
                                  predictor_manipulator=pred_manip,
                                  fitness_metric=AtomicPotential)
    success = islmngr.run_islands(MAX_STEPS, EPSILON, step_increment=N_STEPS)
    assert success


def compare_agraph_potential(X, Y):
    """does the comparison using agraph"""

    # make solution manipulator
    sol_manip = agm(1, 16, nloads=2)
    sol_manip.add_node_type(AGNodes.Add)
    sol_manip.add_node_type(AGNodes.Multiply)
    # sol_manip.add_node_type(AGNodes.Subtract)
    # sol_manip.add_node_type(AGNodes.Divide)
    # sol_manip.add_node_type(AGNodes.Exp)
    # sol_manip.add_node_type(AGNodes.Sin)
    # sol_manip.add_node_type(AGNodes.Cos)

    # make predictor manipulator
    pred_manip = fpm(32, X.shape[0])

    # make and run island manager
    islmngr = SerialIslandManager(N_ISLANDS,
                                  data_x=X,
                                  data_y=Y.flatten(),
                                  solution_manipulator=sol_manip,
                                  predictor_manipulator=pred_manip,
                                  fitness_metric=AtomicPotential)
    success = islmngr.run_islands(MAX_STEPS, EPSILON, step_increment=N_STEPS)
    assert success
