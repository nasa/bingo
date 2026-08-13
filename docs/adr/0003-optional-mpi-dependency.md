# Optional MPI Dependency

Bingo's default installation provides serial functionality without MPI. Parallel evolution is installed with the `MPI` optional dependency, so `ParallelArchipelago` remains separately MPI-validated without imposing `mpi4py` installation on every user. This is an immediate documented upgrade change; source-checkout dependencies remain MPI-capable for development and full-suite validation.
