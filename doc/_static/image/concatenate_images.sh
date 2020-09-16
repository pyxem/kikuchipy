#!/bin/bash

# Requires ImageMagick

# Directories
PATPROC=pattern_processing
CHANGES=change_scan_pattern_size
VIS=visualizing_patterns
FEATMAP=feature_maps
VIRTUAL=virtual_backscatter_electron_imaging
REF_FRAMES=reference_frames

# Pattern processing
convert ${PATPROC}/pattern_raw.png ${PATPROC}/pattern_static.png +append ${PATPROC}/static_correction.jpg
convert ${PATPROC}/pattern_raw.png ${PATPROC}/dynamic_background.png +append ${PATPROC}/get_dynamic_background.jpg
convert ${PATPROC}/pattern_static.png ${PATPROC}/pattern_dynamic.png +append ${PATPROC}/dynamic_correction.jpg
convert ${PATPROC}/pattern_dynamic.jpg ${PATPROC}/pattern_adapthist.jpg +append ${PATPROC}/adapthist.jpg
convert ${PATPROC}/pattern_scan_static.png ${PATPROC}/pattern_scan_averaged.png +append ${PATPROC}/average_neighbour_pattern.jpg
convert ${PATPROC}/contrast_stretching_before.png ${PATPROC}/contrast_stretching_after.png +append ${PATPROC}/contrast_stretching.jpg
convert ${PATPROC}/rescale_intensities_before.png ${PATPROC}/rescale_intensities_after.png +append ${PATPROC}/rescale_intensities.jpg
convert ${PATPROC}/normalize_intensity_before.png ${PATPROC}/normalize_intensity_after.png +append ${PATPROC}/normalize_intensity.jpg
convert ${PATPROC}/pattern_dynamic.png ${PATPROC}/fft_filter_highlowpass_after.png +append ${PATPROC}/fft_filter_highlowpass_result.jpg
convert ${PATPROC}/fft_filter_highlowpass2d.png ${PATPROC}/fft_filter_highlowpass1d.png +append ${PATPROC}/fft_filter_highlowpass.jpg
convert ${PATPROC}/fft_filter_laplacian_correlate.png ${PATPROC}/fft_filter_laplacian_spatial.png -gravity center +append ${PATPROC}/fft_filter_laplacian.jpg

# Change image size
convert ${PATPROC}/pattern_dynamic.jpg ${CHANGES}/pattern_cropped.png +append ${CHANGES}/change_pattern_size.jpg

# Visualizing patterns
# Rescale navigators
convert ${VIS}/vbse_navigator.png -resize 860x581 ${VIS}/vbse_navigator_rescaled.jpg
convert ${VIS}/orientation_similarity_map_navigator.jpg -resize 860x581 ${VIS}/osm_navigator_rescaled.jpg
convert ${VIS}/orientation_map_navigator.jpg -resize 860x581 ${VIS}/om_navigator_rescaled.jpg
# Concatenate images
convert ${VIS}/vbse_signal.png ${VIS}/vbse_navigator_rescaled.jpg -gravity center +append ${VIS}/vbse_navigation.jpg
convert ${VIS}/vbse_signal.png ${VIS}/osm_navigator_rescaled.jpg -gravity center +append ${VIS}/osm_navigation.jpg
convert ${VIS}/vbse_signal.png ${VIS}/om_navigator_rescaled.jpg -gravity center +append ${VIS}/om_navigation.jpg

# Feature maps
convert ${FEATMAP}/image_quality_pattern.png ${FEATMAP}/fft_spectrum.png ${FEATMAP}/fft_frequency_vectors.png +append ${FEATMAP}/image_quality_pattern.jpg

# Virtual imaging
convert ${VIRTUAL}/images_nav.jpg -resize 477x433 ${VIRTUAL}/images_nav_rescaled.jpg
convert ${VIRTUAL}/images_nav_rescaled.jpg ${VIRTUAL}/images_sig.jpg -gravity center +append ${VIRTUAL}/images.jpg

# Reference frames
convert ${REF_FRAMES}/detector_coordinates.png ${REF_FRAMES}/detector_coordinates.jpg
convert ${REF_FRAMES}/gnomonic_coordinates.png ${REF_FRAMES}/gnomonic_coordinates.jpg
convert ${REF_FRAMES}/detector_coordinates.jpg ${REF_FRAMES}/gnomonic_coordinates.jpg +append ${REF_FRAMES}/detector_gnomonic_coordinates.jpg
