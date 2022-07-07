Formatting Data
===============

All Bingo equations expect data to be formatted based on the number
of variables and datapoints in the dataset.

Input
"""""
Bingo expects that input data is formatted with each variable as a column and
each datapoint as a row.

Layout of inputs:

=============== ============== ============== ============== ==============
:math:`i`       :math:`X_0`    :math:`X_1`    :math:`\ldots` :math:`X_n`
=============== ============== ============== ============== ==============
0               0.1            1.2            :math:`\ldots` 1.2
1               0.1            2.3            :math:`\ldots` 3.5
2               0.1            1.2            :math:`\ldots` 6.0
:math:`\vdots`  :math:`\vdots` :math:`\vdots` :math:`\vdots` :math:`\vdots`
=============== ============== ============== ============== ==============

.. note::
    Bingo starts counting at 0, so :math:`X_0` is the first variable,
    :math:`X_1` is the second, and so on.

So, if we had 2 variables and 10 samples, we would have an array with
10 rows and 2 columns:

.. code-block:: python

    import numpy as np
    X_0 = np.linspace(1, 10, num=10).reshape((-1, 1))
    X_1 = np.linspace(-10, 1, num=10).reshape((-1, 1))
    X = np.hstack((X_0, X_1))

Output
""""""
Bingo expects output data to be formatted as a
of the same number of samples as the input.

Layout of output:

=========== =========== =========== ============== ===========
:math:`i`   :math:`0`   :math:`1`   :math:`\ldots` :math:`n`
=========== =========== =========== ============== ===========
:math:`y_i` 0.0         -1.1        :math:`\ldots` 5.0
=========== =========== =========== ============== ===========

Using the previous setup, let's
create output data by using the equation :math:`5.0 * X_0 + X_1`:

.. code-block:: python

    y = 5.0 * X_0 + X_1
