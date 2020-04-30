#!/bin/bash

# Requires ImageMagick

# Directories
PATPROC=pattern_processing
CHANGES=change_scan_pattern_size
VIS=visualizing_patterns

# Pattern processing
convert ${PATPROC}/pattern_raw.jpg ${PATPROC}/pattern_static.jpg +append ${PATPROC}/static_correction.jpg
convert ${PATPROC}/pattern_static.jpg ${PATPROC}/pattern_dynamic.jpg +append ${PATPROC}/dynamic_correction.jpg
convert ${PATPROC}/pattern_dynamic.jpg ${PATPROC}/pattern_adapthist.jpg +append ${PATPROC}/adapthist.jpg
convert ${PATPROC}/pattern_scan_static.png ${PATPROC}/pattern_scan_averaged.png +append ${PATPROC}/average_neighbour_pattern.jpg
convert ${PATPROC}/contrast_stretching_before.png ${PATPROC}/contrast_stretching_after.png +append ${PATPROC}/contrast_stretching.jpg
convert ${PATPROC}/rescale_intensities_before.png ${PATPROC}/rescale_intensities_after.png +append ${PATPROC}/rescale_intensities.jpg

# Change image size
convert ${PATPROC}/pattern_dynamic.jpg ${CHANGES}/pattern_cropped.jpg +append ${CHANGES}/change_pattern_size.jpg

# Virtual image
convert ${VIS}/vbse_navigator.jpg -resize 860x581 ${VIS}/vbse_navigator_rescaled.jpg
convert ${VIS}/vbse_navigator_rescaled.jpg ${VIS}/pattern_roi.jpg -gravity center +append ${VIS}/roi_vbse_navigator.jpg
