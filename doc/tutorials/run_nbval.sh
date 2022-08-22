#!/bin/bash

# This script must be run from the kikuchipy/ top directory:
#   $ chmod u+x ./doc/tutorials/run_nbval.sh
#   $ ./doc/tutorials/run_nbval.sh

# List notebooks that nbval should skip
declare -a SKIP=(\
  "esteem2022_diffraction_workshop.ipynb"\
)

# Find all notebooks in the doc/tutorials directory
readarray -t ALL_NOTEBOOKS < <(find doc/tutorials -maxdepth 1 -type f -name "*.ipynb")
declare -a ALL_NOTEBOOKS

# Remove notebooks to skip
for i in "${!ALL_NOTEBOOKS[@]}"; do
  NOTEBOOK="${ALL_NOTEBOOKS[i]}"
  SKIP_THIS=0
  for j in "${!SKIP[@]}"; do
    if [[ "${NOTEBOOK}" = doc/tutorials/"${SKIP[j]}" ]]; then
      SKIP_THIS=1
    fi
  done
  if [[ "${SKIP_THIS}" = 0 ]]; then
    NOTEBOOKS[$i]="${NOTEBOOK}"
  fi
done

# Test with nbval
pytest -v --nbval "${NOTEBOOKS[@]}" --sanitize-with doc/tutorials/tutorials_sanitize.cfg
