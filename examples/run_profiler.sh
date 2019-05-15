#!/bin/bash

PROFILER="cProfile"
FILE_NAME="AgraphExample.py"
OUTPUT="agraph_profile.cprof"

# Run FILE_NAME with PROFILER and write to OUTPUT
python -m ${PROFILER} -o ${OUTPUT} ${FILE_NAME} 