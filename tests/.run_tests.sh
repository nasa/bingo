#!/usr/bin/env bash

set -e

MPI_EXEC=$(python -c "import mpi4py;import os;filename = next(iter(mpi4py.get_config().items()))[1];print(os.path.dirname(filename)+'/mpiexec');")
RUN_MODE=${1-"coverage"}

echo "Running tests in $RUN_MODE mode"

# run mpi tests
for i in tests/integration/mpitests/*.py
do
  if [ $i != "tests/integration/mpitests/mpitest_util.py" ]
  then
    echo "Running mpitest: $i in $RUN_MODE mode"
    if [ $RUN_MODE == "coverage" ]
    then
      $MPI_EXEC -np 3 coverage run --parallel-mode --source=bingo $i
    elif [ $RUN_MODE == "normal" ]
    then
      $MPI_EXEC -np 3 python $i
    fi
  fi
done

# run pytest tests
if [ $RUN_MODE == "coverage" ]
then
  coverage combine
  pytest tests --cov=bingo --cov-report=term-missing --cov-append
elif [ $RUN_MODE == "normal" ]
then
  pytest tests
fi
