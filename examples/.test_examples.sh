#!/usr/bin/env bash

set -e

# python -c "from bingo import symbolic_regression; print('Using %s Backend' % ('c++' if symbolic_regression.ISCPP else 'Python'))"

for i in examples/*.ipynb
do
  echo "Running Notebook: $i"
  for attempt in 1 2 3
  do
    if jupyter nbconvert --stdout --execute --to python $i > /dev/null
    then
      break
    fi
    if [ "$attempt" -eq 3 ]
    then
      echo "Notebook failed after $attempt attempts" >&2
      exit 1
    fi
    echo "Notebook execution failed; retrying ($attempt/3)"
  done
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
