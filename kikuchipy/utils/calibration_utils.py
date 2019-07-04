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


def edax2emsoft(xstar, ystar, zstar, pattern_width, pattern_height,
                detector_pixel_size):
    """Convert pattern centre in EDAX/TSL format to EMsoft format, as
    described in [1]_.

    Parameters
    ----------
    xstar : float
        Horizontal (left-to-right) coordinate of the point on the
        scintillator closest to the generation point of the pattern on
        the surface, in fraction of screen pixels.
    ystar : float
        Vertical (bottom-to-top) coordinate of the point on the
        scintillator closest to the generation point of the pattern on
        the surface, in fraction of screen pixels.
    zstar : float
        Ratio of detector-surface distance to the detector width.
    pattern_width : float
        Detector width in pixels.
    pattern_height : float
        Detector height in pixels.
    detector_pixel_size : float
        Detector pixel size in µm.

    Returns
    -------
    xpc : float
        Horizontal coordinate of the point on the scintillator closest
        to the generation point of the pattern on the surface, in
        fraction of screen pixels from the centre.
    ypc : float
        Vertical coordinate of the point on the scintillator closest
        to the generation point of the pattern on the surface, in
        fraction of screen pixels from the centre.
    L : float
        Distance from the generation point of the pattern on the surface
        to the scintillator, in µm.

    References
    ----------
    .. [1] Jackson et al.: Dictionary Indexing of Electron Back-Scatter
           Diffraction Patterns: A Hands-On Tutorial. Integrating
           Materials and Manufacturing Innovation 2019 8:226–246, doi:
           https://doi.org/10.1007/s40192-019-00137-4.
    """
    xpc = pattern_width * (xstar - 0.5)
    ypc = pattern_width * ystar - pattern_height * 0.5
    L = pattern_width * detector_pixel_size * zstar
    return xpc, ypc, L


def oxford2emsoft(xstar, ystar, zstar, pattern_width, pattern_height,
                  detector_pixel_size):
    """Convert pattern centre in Oxford format to EMsoft format, as
    described in [1]_.

    Parameters
    ----------
    xstar : float
        Horizontal (left-to-right) coordinate of the point on the
        scintillator closest to the generation point of the pattern on
        the surface, in fraction of detector pixels.
    ystar : float
        Vertical (bottom-to-top) coordinate of the point on the
        scintillator closest to the generation point of the pattern on
        the surface, in fraction of detector pixels.
    zstar : float
        Ratio of detector-surface distance to the detector width.
    pattern_width : float
        Detector width in pixels.
    pattern_height : float
        Detector height in pixels.
    detector_pixel_size : float
        Detector pixel size in µm.

    Returns
    -------
    xpc : float
        Horizontal coordinate of the point on the scintillator closest
        to the generation point of the pattern on the surface, in
        fraction of screen pixels from the centre.
    ypc : float
        Vertical coordinate of the point on the scintillator closest
        to the generation point of the pattern on the surface, in
        fraction of screen pixels from the centre.
    L : float
        Distance from the generation point of the pattern on the surface
        to the scintillator, in µm.

    References
    ----------
    .. [1] Jackson et al.: Dictionary Indexing of Electron Back-Scatter
           Diffraction Patterns: A Hands-On Tutorial. Integrating
           Materials and Manufacturing Innovation 2019 8:226–246, doi:
           https://doi.org/10.1007/s40192-019-00137-4.
    """
    xpc = pattern_width * (xstar - 0.5)
    ypc = pattern_height * (ystar - 0.5)
    L = pattern_width * detector_pixel_size * zstar
    return xpc, ypc, L


def bruker2emsoft(xstar, ystar, zstar, pattern_width, pattern_height,
                  detector_pixel_size):
    """Convert pattern centre in Bruker format to EMsoft format, as
    described in [1]_.

    Parameters
    ----------
    xstar : float
        Horizontal (left-to-right) coordinate of the point on the
        scintillator closest to the generation point of the pattern on
        the surface, in fraction of screen pixels.
    ystar : float
        Vertical (bottom-to-top) coordinate of the point on the
        scintillator closest to the generation point of the pattern on
        the surface, in fraction of screen pixels.
    zstar : float
        Ratio of detector-surface distance to the detector height.
    pattern_width : float
        Detector width in pixels.
    pattern_height : float
        Detector height in pixels.
    detector_pixel_size : float
        Detector pixel size in µm.

    Returns
    -------
    xpc : float
        Horizontal coordinate of the point on the scintillator closest
        to the generation point of the pattern on the surface, in
        fraction of screen pixels from the centre.
    ypc : float
        Vertical coordinate of the point on the scintillator closest
        to the generation point of the pattern on the surface, in
        fraction of screen pixels from the centre.
    L : float
        Distance from the generation point of the pattern on the surface
        to the scintillator, in µm.

    References
    ----------
    .. [1] Jackson et al.: Dictionary Indexing of Electron Back-Scatter
           Diffraction Patterns: A Hands-On Tutorial. Integrating
           Materials and Manufacturing Innovation 2019 8:226–246, doi:
           https://doi.org/10.1007/s40192-019-00137-4.
    """
    xpc = pattern_width * (xstar - 0.5)
    ypc = pattern_height * (0.5 - ystar)
    L = pattern_height * detector_pixel_size * zstar
    return xpc, ypc, L
