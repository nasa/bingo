Installation Guide
==================

To install Bingo, simply use pip:

.. code-block:: console

    pip install bingo-nasa

Windows Wheels
--------------

The release workflow builds Windows x64 wheels for Python 3.11 through 3.14
alongside Linux wheels and the source distribution. Each Windows wheel passes
C++ backend and Python/C++ parity tests, serial installation checks, metadata
validation, and a clean binary-only installation with uv before publication.
The Windows wheels are included in the same tag-gated PyPI publication and
signed GitHub Release as the Linux distributions.

Once a release containing Windows wheels is published, install it without
Visual Studio C++ Build Tools:

.. code-block:: console

    uv add bingo-nasa

The reusable **Windows Wheels** GitHub Actions workflow also supports manual
builds without publication. Once it is available on the default branch, select
**Actions > Windows Wheels > Run workflow** and choose the branch to build.
Alternatively, run **Distribution Validation and Release** manually to build
and validate distributions for both platforms without publishing them.
Download and unzip the successful run's
``windows-wheel-cp312`` artifact for Python 3.12, or the corresponding artifact
for another Python version. Artifacts are retained for 14 days.

In a new directory on Windows, install the downloaded wheel without compiling:

.. code-block:: console

    uv venv --python 3.12
    uv pip install --python .\.venv\Scripts\python.exe --only-binary=:all: "C:\path\to\downloaded.whl"
    .\.venv\Scripts\python.exe -c "from bingo.expressions.agraph.cppagraph import AGraphExpression; print('C++ backend loaded')"

Replace the wheel path above with the actual downloaded filename. The explicit
``--python`` selects the new environment rather than an active Conda environment.
Use a wheel matching your Python version: ``cp311`` for Python 3.11,
``cp312`` for 3.12, ``cp313`` for 3.13, or ``cp314`` for 3.14. These wheels
target 64-bit x86 Windows (``win_amd64``), not ARM64 or 32-bit Windows.
Downloaded workflow wheels are artifacts of the selected branch, not necessarily the
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
