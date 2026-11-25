Installation Guide
==================

To install Bingo, simply use pip:

.. code-block:: console

    pip install bingo-nasa

Experimental Windows Wheels
---------------------------

Windows x64 wheels can be built using the **Windows Wheel Trial** GitHub Actions
workflow. It builds wheels for Python 3.11 through 3.14,
tests the installed C++ backend and Python/C++ parity, and verifies a clean
installation with uv. This workflow does not publish to PyPI.

Until the workflow is available on the default branch, a temporary trigger
also runs the trial when ``.github/workflows/windows-wheels.yml`` changes in a
push to ``develop``. Open the resulting run under **Actions**; no merge to the
default branch is needed for this trial. Remove this temporary push trigger
once the workflow is available for manual dispatch on the default branch.

After the workflow changes are pushed to GitHub and available on the default
branch, select **Actions > Windows Wheel Trial > Run workflow** and choose the
branch to build. Download and unzip the successful run's
``windows-wheel-cp312`` artifact for Python 3.12, or the corresponding artifact
for another Python version. Artifacts are retained for 14 days.

In a new directory on Windows, install the downloaded wheel without compiling:

.. code-block:: powershell

    uv venv --python 3.12
    # Replace this path with the actual downloaded wheel filename.
    uv pip install --only-binary=:all: "C:\path\to\downloaded.whl"
    .\.venv\Scripts\python.exe -c "from bingo.expressions.agraph.cppagraph import AGraphExpression; print('C++ backend loaded')"

Use a wheel matching your Python version: ``cp311`` for Python 3.11,
``cp312`` for 3.12, ``cp313`` for 3.13, or ``cp314`` for 3.14. These wheels
target 64-bit x86 Windows (``win_amd64``), not ARM64 or 32-bit Windows.
They are experimental artifacts of the selected branch, not necessarily the
same version as a published release. Installing a wheel does not require
Visual Studio C++ Build Tools.

Evidence Estimation
-------------------

Install the optional Evidence-estimation dependency when using
``SmcEvidenceEstimator``:

.. code-block:: console

    pip install "bingo-nasa[evidence]"

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
