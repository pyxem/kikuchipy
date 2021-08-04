# -*- coding: utf-8 -*-
# Copyright 2019-2021 The kikuchipy developers
#
# This file is part of kikuchipy.
#
# kikuchipy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# kikuchipy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with kikuchipy. If not, see <http://www.gnu.org/licenses/>.

"""Utilities for attaching an EBSD detector,
:class:`kikuchipy.detectors.EBSDDetector` to an EBSD signal,
:class:`kikuchipy.signals.EBSD`.
"""

from kikuchipy.detectors import EBSDDetector


def _detector_is_compatible_with_signal(
    detector: EBSDDetector,
    navigation_shape: tuple,
    signal_shape: tuple,
    raise_if_not: bool = False,
) -> bool:
    """Check whether a signal's navigation and signal shape is
    compatible with a detector and return a bool or raise a ValueError
    if it is not.
    """
    # TODO: Check pixel scale
    compatible = True
    error_msg = None

    if signal_shape != detector.shape:
        compatible = False
        error_msg = "Detector and signal must have the same signal shape"

    detector_nav_shape = detector.navigation_shape
    if detector_nav_shape != (1,) and detector_nav_shape != navigation_shape:
        compatible = False
        error_msg = (
            "Detector must have exactly one projection center (PC), or one PC per "
            "pattern in an array of shape signal's navigation shape + (3,)"
        )

    if raise_if_not and not compatible:
        raise ValueError(error_msg)
    else:
        return compatible
