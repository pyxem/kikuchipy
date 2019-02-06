# -*- coding: utf-8 -*-
# Copyright 2019 The KikuchiPy developers
#
# This file is part of KikuchiPy.
#
# KikuchiPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# KikuchiPy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with KikuchiPy. If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import dask.array as da
from skimage.feature import greycomatrix, greycoprops
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.stats import binned_statistic
from kikuchipy.utils.expt_utils import rescale_pattern_intensity


def texture_features_glcm(components, features):
    """Determine texture features in decomposition components using the
    grey level co-occurrence matrix (GLCM).

    Parameters
    ----------
    components : three-dimensional array_like
        Decomposition factors or loadings.
    features : array_like of str
        Features of the GLCM to compute.

    Returns
    -------
    feature_values : two-dimensional array_like
        Feature values in an array with shape (# components,
        # features).

    Notes
    -----
    This is an experimental function and should be used with care.
    """
    if isinstance(components, da.Array):
        components = components.compute(progressbar=False)

    if not (isinstance(features, list) or isinstance(features, np.ndarray)):
        features = [features]

    # Rescale intensities to the full unsigned 8-bit integer range, keeping
    # relative intensities between components
    components = components.astype(np.float64)
    dtype = np.uint8
    iinfo_dtype = np.iinfo(dtype)  # Yields a GLCM shape of (255, 255)
    imin, imax = components.min(), components.max()
    scale = float(iinfo_dtype.max/(imax - imin))
    for i, comp in enumerate(components):
        components[i] = rescale_pattern_intensity(comp, imin, scale, dtype)
    components = components.astype(dtype)

    # Determine GLCM features
    feature_values = np.zeros((components.shape[0], len(features)))
    for i, comp in enumerate(components):
        glcm = greycomatrix(comp, distances=[1], angles=[0], symmetric=True,
                            normed=True)
        for j, feat in enumerate(features):
            feature_values[i, j] = greycoprops(glcm, feat)[0, 0]

    return feature_values


def cluster_texture_features(feature_values):
    """Cluster texture features determined from grey level co-occurrence
    matrices (GLCMs) of decomposition components into two classes,
    signal or noise.

    Clustering is done based upon the assumption that the more noisy the
    decomposition component is, the more random the pixel intensities
    are.

    Parameters
    ----------
    feature_values : two-dimensional array_like
        GLCM feature values in an array with shape (# components,
        # features). The function assumes the following features in each
        column in order: factor contrast, correlation and dissimilarity
        and loading correlation.

    Returns
    -------
    signal : array_like of bool
        Array with values True/False (signal/noise) with indices
        corresponding to the components' indices in the learning
        results.

    Notes
    -----
    This is an experimental function and should be used with care.
    """
    # Reduce high-dimensional feature data to two dimensions before clustering
    # using t-distributed stochastic neighbour embedding (t-SNE)
    tsne = TSNE(perplexity=5, random_state=1)
    Y = tsne.fit_transform(feature_values)

    # Cluster feature values into eight clusters (default) using k-means
    # clustering, because there is too much overlap between the signal and noise
    # components
    clusters1 = KMeans(random_state=0).fit(Y)

    # Calculate average GLCM values within the eight clusters
    unique_clusters = np.unique(clusters1.labels_)
    num_features = feature_values.shape[1]
    cluster_values = np.zeros((len(unique_clusters), num_features))
    for i in unique_clusters:
        cluster_values[i] = [np.mean(feature_values[clusters1.labels_ == i, j])
                             for j in range(num_features)]

    # Random images have virtually no correlation between neighbour pixels, and
    # the more random an image is, the higher the contrast. We exploit these
    # texture features for the final clustering, computing factor contrast
    # divided by loading correlation.
    final_feature = cluster_values[:, 0] / cluster_values[:, 3]
    final_feature = final_feature.reshape(-1, 1)

    # Cluster those clusters into two clusters (signal = 0 and noise = 1)
    clusters2 = KMeans(n_clusters=2, random_state=1).fit(final_feature)

    # Classify components
    signal = np.ones_like(clusters1.labels_, dtype=bool)
    for i in unique_clusters:
        signal[clusters1.labels_ == i] = not clusters2.labels_[i]

    return signal


def detect_noisy_components(factors):
    """Detect decomposition component factors with periodic lines, i.e.
    detector noise, with sufficiently higher intensity than any relevant
    signal.

    Parameters
    -----------
    factors : three-dimensional array_like
        Decomposition factors.

    Returns
    -------
    too_noisy : array_like of bool
        Array with values True/False (too noisy/OK) with indices
        corresponding to the components' indices in the learning
        results.

    Notes
    -----
    This is an experimental function and should be used with care.
    """
    # Prepare for computation of power spectra of factors
    spectra = np.zeros_like(factors)
    num_factors, sx, sy = spectra.shape
    sum_x = np.zeros((num_factors, sx))
    sum_y = np.zeros((num_factors, sy))

    # Spectrum masks: a centre cross and a centre circle
    x, y = np.meshgrid(np.arange(sx), np.arange(sy))
    centre = (int(sx / 2), int(sy / 2))
    radius = int(((sx + sy) / 2) / 10)
    R2 = ((x - centre[0]) ** 2 + (y - centre[1]) ** 2)

    # Compute power spectra and apply masks
    for i, factor in enumerate(factors):
        spec = np.fft.fftshift(np.fft.fft2(factor))
        spec = np.log(abs(spec))  # Real part of spectrum
        spec *= (R2 >= radius**2)  # Circle mask
        spec *= ((x <= centre[0] - 1) + (x >= centre[0] + 1))  # Line mask x
        spec *= ((y <= centre[1] - 1) + (y >= centre[1] + 1))  # Line mask y
        sum_x[i] = np.sum(spec, axis=0)  # Sum intensities in x
        sum_y[i] = np.sum(spec, axis=1)  # Sum intensities in y
        spectra[i] = spec

    # Sum spectra in both directions
    if sx != sy:
        if sx > sy:  # Bin in y direction to get same size
            sum_y_new = np.zeros_like(sum_x)
            for i, sum in enumerate(sum_y):
                sum = sum_y[i]
                sum_y_new[i] = binned_statistic(sum, sum, bins=sx).statistic
            sum_y = sum_y_new
        else:  # Bin in x direction
            sum_x_new = np.zeros_like(sum_y)
            for i, sum in enumerate(sum_x):
                sum = sum_x[i]
                sum_x_new[i] = binned_statistic(sum, sum, bins=sy).statistic
            sum_x = sum_x_new
    sum_z = sum_x + sum_y
    half_spectrum = int(sum_z.shape[1] / 2)

    # Distance between median position of max. intensity and actual position of
    # max. intensity in each spectrum
    indices_max = sum_z[:, :half_spectrum].argmax(axis=1)
    median_index_max = int(np.median(indices_max))
    diff_max_index = abs(indices_max - median_index_max)

    # Difference in max. intensity in a window where max. intensity is expected
    # and actual max. intensity in each spectrum
    sum_z_max = np.max(sum_z, axis=1)
    window_width = int(sum_z.shape[1] / 12)
    window_start = half_spectrum - int(radius / 2) - window_width
    window_end = window_start + window_width
    window = range(window_start, window_end)
    sum_z_win_max = np.max(sum_z[:, window], axis=1)
    diff_max_intensity = abs(sum_z_max - sum_z_win_max) / sum_z_max + 1

    # Classify factors
    too_noisy = np.where((diff_max_index > window_width) &
                         (diff_max_intensity > 1.01))

    return too_noisy
