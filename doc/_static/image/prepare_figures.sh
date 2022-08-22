#!/bin/bash

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
convert \
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
