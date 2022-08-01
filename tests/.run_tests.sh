#!/usr/bin/env bash

set -e

MPI_EXEC=$(python -c "import mpi4py;import os;filename = next(iter(mpi4py.get_config().items()))[1];print(os.path.dirname(filename)+'/mpiexec');")

run_mpi_tests() {  # $1 = run mode
  for i in tests/integration/mpitests/*.py
  do
    if [ $i != "tests/integration/mpitests/mpitest_util.py" ]
    then
      echo "Running mpitest: $i in $1 mode"
      if [ $1 == "coverage" ]
      then
        $MPI_EXEC -np 3 coverage run --parallel-mode --source=bingo $i
      elif [ $1 == "normal" ]
      then
        $MPI_EXEC -np 3 python $i
      fi
    fi
  done
}


RUN_MODE=${1-"coverage"}

echo "Running tests in $RUN_MODE mode"

if [ $RUN_MODE == "coverage" ]
then
  run_mpi_tests $RUN_MODE
  coverage combine
  pytest tests --cov=bingo --cov-report=term-missing --cov-append
elif [ $RUN_MODE == "normal" ]
then
  run_mpi_tests $RUN_MODE
  pytest tests
fi
