Installation Guide
==================

To install Bingo, simply use pip:

.. code-block:: console

    pip install bingo-nasa

To use parallel island evolution, install the MPI extra:

.. code-block:: console

    pip install "bingo-nasa[MPI]"

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

The source-checkout requirements include MPI support for development and the
full test suite. For a package installation with only parallel evolution added,
use ``pip install "bingo-nasa[MPI]"``.

Optionally build the C++ expression backend:

.. code-block:: console

    ./.build_cppagraph.sh

Now you should be good to go! You can run Bingo's test suite to make sure that
the installation process worked properly:

.. code-block:: console

    pytest tests
