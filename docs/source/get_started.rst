Get Started
===========

If you haven't already, check out the `installation guide <installation.html>`_
to get Bingo setup.

..
    TODO: GPSR explanation should probably be a high-level story, not in the get started page

GPSR
----

Before you can understand what Bingo does, it's important to understand what
GPSR is. GPSR stands for genetic programming with symbolic regression.
Symbolic regression involves finding an equation that models data and
genetic programming is a method to do symbolic regression by modeling evolution.

Evolution
^^^^^^^^^
GPSR works by evolving a population of equations. It evolves equations in
stages: variation, evaluation, and selection. It continues to do this until
a good enough equation is found or another criteria is met (see the picture
below).

Variation
^^^^^^^^^
In order to get better individuals, we have to change those that are already in
the population. This is usually done through two possible operators: mutation
and crossover. Mutation takes an equation and slightly changes it (see the
picture below). While crossover takes two equations and mixes their parts
together. Some of the individuals from our population undergo one, both, or
neither of these operators to form a child population.

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

Finishing
^^^^^^^^^
Once we have reached a termination criteria (e.g., we evolved for a certain
number of generations or we found a good enough individual), we stop evolving
and get the final population. We can then select some individual from that
population (or the entire run) to use to do our task.

Using the SkLearn Wrapper
-------------------------

It is recommended to use the scikit-learn wrapper: ``SymbolicRegressor`` when
first learning Bingo. Let's setup a test case to show how it works. You can
learn more about the sklearn wrapper in the `high-level guide <high_level.html>`_.

Creating training data
^^^^^^^^^^^^^^^^^^^^^^
Let's make some dummy data to train on.

Input
"""""
Bingo expects that the input is formatted with each variable as a column and
each datapoint as a row. So, if we had 2 variables and 10 samples,
we would have an array with 10 rows and 2 columns:

.. code-block:: python

    import numpy as np
    X_0 = np.linspace(1, 10, num=10).reshape((-1, 1))
    X_1 = np.linspace(-10, 1, num=10).reshape((-1, 1))
    X = np.hstack((X_0, X_1))

Output
""""""
Bingo expects output data to be formatted as a
of the same number of samples as the input. Using the previous setup, let's
create output data by using the equation :math:`5.0 * X_0 + X_1`:

.. code-block:: python

    y = 5.0 * X_0 + X_1

.. note::
    Bingo starts counting at 0, so :math:`X_0` is the first variable,
    :math:`X_1` is the second, and so on.

Training
""""""""

We can then easily fit a model on this data:

..
    TODO verify this works with the API

.. code-block:: python

    from bingo.symbolic_regression.symbolic_regressor import SymbolicRegressor
    regressor = SymbolicRegressor()
    regressor.fit(X, y)

Results
"""""""

We can easily get the best equation after the model is fit:

.. code-block:: python

    print("best individual is:", regressor.best_individual)
.. code-block:: console

    > best individual is: 5.0 * X_0 + X_1
..
    TODO selection methods, predict(), and evaluating individual
