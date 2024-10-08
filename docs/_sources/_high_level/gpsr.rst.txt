GPSR
====

An important concept for understanding Bingo is GPSR. GPSR stands 
for genetic programming with symbolic regression. Symbolic regression involves
finding an equation that models data and genetic programming is a method to do
symbolic regression by modeling evolution.

Equations
---------

GPSR uses the process of evolution to find equations that model data well.
To accomodate the genetic-like process of GPSR, we encode equations as
directed acyclic graphs: ``AGraphs``. ``AGraphs`` have nodes and connections
between nodes. These nodes are either terminals, which load data, or operators,
which perform operations (e.g., addition, multiplication, etc.).

.. image:: ../_static/agraph.svg
    :width: 300
    :align: center

Consider the ``AGraph`` above which represents the equation
:math:`C_0 X_0 + X_0 + X_1`. Notice, how there are terminal nodes at the bottom
of the ``AGraph`` :math:`C_0`, :math:`X_0`, and :math:`X_1`. Node :math:`C_0`
loads constant 0, which is a free-form numeric value that can change based on
data (i.e. :math:`C_0` could be -1.2, 5.0, etc. depending on its setting).
Nodes :math:`X_0` and :math:`X_1` load the first and second variables of the
dataset respectively. There are also operators that turn the terminals
into a full-scale equation. For example, the multiplication node results
in :math:`C_0 * X_0 = C_0 X_0` and the root node results in the entire
equation :math:`C_0 X_0 + X_0 + X_1`.

Evolution
---------
GPSR works by evolving a population of equations. It evolves equations in
stages: variation, evaluation, and selection. It continues to do this until
a good enough equation is found or another criteria is met (see the picture
below).

.. figure:: ../_static/gpsr.svg
    :width: 630
    :align: center

Variation
^^^^^^^^^
In order to get better individuals, we have to change those that are already in
the population. This is usually done through two possible operators: mutation
and crossover. Mutation takes an equation and slightly changes it.

.. figure:: ../_static/mutation.svg
    :width: 630
    :align: center

    Mutation of an equation.

Crossover takes two equations and mixes their parts
together. Some of the individuals from our population undergo one, both, or
neither of these operators to form a child population.

.. figure:: ../_static/crossover.svg
    :width: 630
    :align: center

    Crossover between two equations.

Evaluation
^^^^^^^^^^
In order to make decisions about which individuals to keep, we want to be
able to gauge how good/fit each individual is. Evaluation is the process of
assigning fitness values to individuals which mark how good/bad they are
at the particular task that we care about.

Selection
^^^^^^^^^
We now have a population of parents and children with scores associated with
each individual, so we can decide which ones should move onto the next
generation. Selection is the process of deciding which individuals will
continue in the evolutionary process. This is usually done by looking at
individual's fitness and mixing in some randomness or other factors.

.. figure:: ../_static/selection.svg
    :width: 630
    :align: center

    Selection process on parents + children (individuals from variation).

Finishing
^^^^^^^^^
Once we have reached a termination criteria (e.g., we evolved for a certain
number of generations or we found a good enough individual), we stop evolving
and get the final population. We can then select some individual from that
population (or the entire run) to use to do our task.