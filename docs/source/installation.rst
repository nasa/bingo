Installation Guide
==================

To install Bingo, simply use pip:

.. code-block:: console

    pip install bingo-nasa

Evidence Estimation
-------------------

Install the optional Evidence-estimation dependency when using
``SmcEvidenceEstimator``:

.. code-block:: console

    pip install "bingo-nasa[evidence]"

Source Code
-----------

For those looking to develop their own features in Bingo.

First clone the repo and move into the directory:

.. code-block:: console

    git clone https://github.com/nasa/bingo.git
    cd bingo

Then make sure you have the requirements necessary to use Bingo:

.. code-block:: console

    pip install -r requirements.txt

Optionally build the C++ expression backend:

.. code-block:: console

    ./.build_cppagraph.sh

Now you should be good to go! You can run Bingo's test suite to make sure that
the installation process worked properly:

.. code-block:: console

    pytest tests
