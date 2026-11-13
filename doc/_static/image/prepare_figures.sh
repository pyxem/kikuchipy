#!/bin/bash
##
## Copyright 2019-2026 the kikuchipy developers
##
## This file is part of kikuchipy.
##
## kikuchipy is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## kikuchipy is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with kikuchipy. If not, see <http://www.gnu.org/licenses/>.
##

DIR=${PWD}/

# Convert .svg to .png
declare -a NAMES=(\
    "reference_frames/sample_detector_geometry" \
    "reference_frames/sample_detector_geometry_bruker" \
    "reference_frames/sample_detector_geometry_edax" \
    "reference_frames/sample_detector_geometry_emsoft" \
    "reference_frames/sample_detector_geometry_oxford" \
)
for NAME in "${NAMES[@]}"; do
    inkscape ${DIR}/${NAME}.svg -d 150 --export-filename=${DIR}/${NAME}.png
done

# Create figures
python reference_frames/doc_reference_frames.py

# Concatenate some figures
magick \
    ${DIR}/reference_frames/gnomonic_coordinates.png \
    ${DIR}/reference_frames/detector_coordinates.png \
    -background white -splice  10x0+0+0 \
    -gravity center +append \
    -chop 10x0+0+0 \
    ${DIR}/reference_frames/gnomonic_detector_coordinates.png

# Remove some figures
declare -a NAMES=(\
    "reference_frames/gnomonic_coordinates.png" \
    "reference_frames/detector_coordinates.png" \
)
for NAME in "${NAMES[@]}"; do
    rm ${DIR}/${NAME}
done
