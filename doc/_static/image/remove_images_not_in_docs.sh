#!/bin/bash

# Images to remove after running concatenate_images.sh. These images are not
# used in the documentation.

# TODO: Use lists and loops

# change_scan_pattern_size
CHANGES=change_scan_pattern_size
rm ${CHANGES}/pattern_cropped.png

# pattern_processing
PATPROC=pattern_processing
rm ${PATPROC}/dynamic_background.png
rm ${PATPROC}/pattern_raw.png
rm ${PATPROC}/pattern_static.png
rm ${PATPROC}/pattern_dynamic.png
rm ${PATPROC}/pattern_scan_static.png
rm ${PATPROC}/pattern_scan_averaged.png
rm ${PATPROC}/rescale_intensities_before.png
rm ${PATPROC}/rescale_intensities_after.png
rm ${PATPROC}/contrast_stretching_before.png
rm ${PATPROC}/contrast_stretching_after.png
rm ${PATPROC}/normalize_intensity_before.png
rm ${PATPROC}/normalize_intensity_after.png
rm ${PATPROC}/fft_filter_highlowpass2d.png
rm ${PATPROC}/fft_filter_highlowpass1d.png
rm ${PATPROC}/fft_filter_highlowpass_after.png
rm ${PATPROC}/fft_filter_laplacian_correlate.png
rm ${PATPROC}/fft_filter_laplacian_spatial.png

# feature_maps
FEATMAP=feature_maps
rm ${FEATMAP}/fft_frequency_vectors.png
rm ${FEATMAP}/fft_spectrum.png
rm ${FEATMAP}/image_quality_pattern.png

# virtual_backscatter_electron_imaging
VIRTUAL=virtual_backscatter_electron_imaging
rm ${VIRTUAL}/images_nav.jpg
rm ${VIRTUAL}/images_nav_rescaled.jpg
rm ${VIRTUAL}/images_sig.jpg

# visualizing_patterns
VIS=visualizing_patterns
rm ${VIS}/om_navigator_rescaled.jpg
rm ${VIS}/orientation_map_navigator.jpg
rm ${VIS}/orientation_similarity_map_navigator.jpg
rm ${VIS}/osm_navigator_rescaled.jpg
rm ${VIS}/pattern.jpg
rm ${VIS}/pattern_roi.jpg
rm ${VIS}/simulated_pattern.jpg
rm ${VIS}/standard_navigator.jpg
rm ${VIS}/vbse_navigator.png
rm ${VIS}/vbse_navigator_rescaled.jpg
rm ${VIS}/vbse_signal.png

# reference frames
REF_FRAMES=reference_frames
rm ${REF_FRAMES}/gnomonic_coordinates.jpg
rm ${REF_FRAMES}/gnomonic_coordinates.png
rm ${REF_FRAMES}/detector_coordinates.jpg
rm ${REF_FRAMES}/detector_coordinates.png
