#!/usr/bin/env bash

set -e

MPI_EXEC=$(python -c "import mpi4py;import os;filename = next(iter(mpi4py.get_config().items()))[1];print(os.path.dirname(filename)+'/mpiexec');")

for i in tests/integration/mpitests/*.py
do
  if [ $i != "tests/integration/mpitests/mpitest_util.py" ]
  then
    echo "Running mpitest: $i"
    $MPI_EXEC -np 3 coverage run --parallel-mode --source=bingo $i
  fi
done
coverage combine

pytest tests --cov=bingo --cov-report=term-missing --cov-append
