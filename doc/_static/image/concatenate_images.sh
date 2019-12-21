#!/bin/bash

# Requires ImageMagick

# Directories
BGCORR=background_correction
CHANGES=change_scan_pattern_size
VIS=visualizing_patterns

# Background correction
#convert ${BGCORR}/pattern_raw.jpg ${BGCORR}/pattern_static.jpg +append ${BGCORR}/static_correction.jpg
#convert ${BGCORR}/pattern_static.jpg ${BGCORR}/pattern_dynamic.jpg +append ${BGCORR}/dynamic_correction.jpg
#convert ${BGCORR}/pattern_dynamic.jpg ${BGCORR}/pattern_adapthist.jpg +append ${BGCORR}/adapthist.jpg

# Change pattern size
#convert ${BGCORR}/pattern_dynamic.jpg ${CHANGES}/pattern_cropped.jpg +append ${CHANGES}/change_pattern_size.jpg

# Virtual image
convert ${VIS}/vbse_navigator.jpg -resize 860x581 ${VIS}/vbse_navigator_rescaled.jpg
convert ${VIS}/vbse_navigator_rescaled.jpg ${VIS}/pattern_roi.jpg -gravity center +append ${VIS}/roi_vbse_navigator.jpg
