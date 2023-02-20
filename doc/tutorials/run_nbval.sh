#!/bin/bash

# This script must be run from the kikuchipy/ top directory:
#   $ chmod u+x ./doc/tutorials/run_nbval.sh
#   $ ./doc/tutorials/run_nbval.sh

# List notebooks that nbval should run
declare -a NOTEBOOKS=(\
  "hough_indexing.ipynb"\
  "hybrid_indexing.ipynb"\
  "mandm2021_sunday_short_course.ipynb"\
  "pattern_matching.ipynb"\
  "pc_extrapolate_plane.ipynb"\
  "pc_fit_plane.ipynb"\
)

# Append relative path to notebook names
for i in "${!NOTEBOOKS[@]}"; do
  NOTEBOOKS[i]=doc/tutorials/"${NOTEBOOKS[i]}"
done

# Test with nbval
pytest -v --nbval "${NOTEBOOKS[@]}" --nbval-sanitize-with doc/tutorials/tutorials_sanitize.cfg
