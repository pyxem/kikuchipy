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
from hyperspy.misc.utils import DictionaryTreeBrowser


def _ebsd_metadata():
    """Create an empty metadata node for an EBSD dataset."""

    md = DictionaryTreeBrowser()
    ebsd_str = 'Acquisition_instrument.SEM.Detector.EBSD'
    md.add_node(ebsd_str)
    params = {'detector': None,
              'azimuth_angle': None,
              'elevation_angle': None,
              'sample_tilt': None,
              'binning': None,
              'detector_pixel_size': None,
              'exposure_time': None,
              'frame_number': None,
              'frame_rate': None,
              'scan_time': None,
              'gain': None,
              'grid_type': None,
              'n_columns': None,
              'n_rows': None,
              'pattern_centre': None,
              'pattern_height': None,
              'pattern_width': None,
              'step_x': None,
              'step_y': None,
              'scan_reference_directions': None,
              'crystal_reference_directions': None,
              }
    md.Acquisition_instrument.SEM.Detector.EBSD.add_dictionary(params)

    return md
