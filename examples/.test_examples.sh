#!/usr/bin/env bash

set -e

python -c "from bingo import symbolic_regression; print('Using %s Backend' % ('c++' if symbolic_regression.ISCPP else 'Python'))"

for i in examples/*.ipynb
do
  echo "Running Notebook: $i"
  jupyter nbconvert --stdout --execute --to python $i > /dev/null
  echo "Success"
  echo ""
done

MPI_EXEC=`which mpiexec`

for i in examples/*.py
do
  echo "Running Script: $i"
  if [ $i == "examples/SRParallelArchipelagoExample.py" ]
  then
    $MPI_EXEC -np 3 python $i > /dev/null
  else
    python $i > /dev/null
  fi
  echo "Success"
  echo ""
done
