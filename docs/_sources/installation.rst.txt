Installation Guide
==================

To install Bingo, simply use pip:

.. code-block:: console

    pip install bingo-nasa

Source Code
-----------

For those looking to develop their own features in Bingo.

First clone the repo and move into the directory:

.. code-block:: console

    git clone --recurse-submodules https://github.com/nasa/bingo.git
    cd bingo

Then make sure you have the requirements necessary to use Bingo:

.. code-block:: console

    pip install -r requirements.txt

Then build BingoCpp:

.. code-block:: console

    ./.build_bingocpp.sh

Now you should be good to go! You can run Bingo's test suite to make sure that
the installation process worked properly:

.. code-block:: console

    pytest tests
