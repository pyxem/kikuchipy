# Copyright 2019-2022 The kikuchipy developers
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

import numpy as np
from orix.sampling import get_sample_fundamental

import kikuchipy as kp


def test_dictionary_indexing(benchmark):
    """Benchmark dictionary indexing of nine (60, 60) EBSD patterns to
    a dictionary of about 3600 patterns with an orientation space
    resolution of about 6 degrees.
    """
    # Load patterns
    s = kp.data.nickel_ebsd_small()
    s.remove_static_background()
    s.remove_dynamic_background()

    # Load master pattern
    mp = kp.data.nickel_ebsd_master_pattern_small(projection="lambert")

    # Sample orientation space
    rot = get_sample_fundamental(resolution=6, point_group=mp.phase.point_group)

    # Define detector
    sig_shape = s.axes_manager.signal_shape[::-1]
    detector = kp.detectors.EBSDDetector(
        shape=sig_shape,
        pc=(0.42, 0.22, 0.50),
        sample_tilt=70,
    )

    # Sample dictionary
    s_dict = mp.get_patterns(rot, detector, mp, compute=True)

    # Signal mask
    signal_mask = ~kp.filters.Window("circular", sig_shape).astype(bool)

    xmap = benchmark(
        s.dictionary_indexing,
        dictionary=s_dict,
        signal_mask=signal_mask,
        keep_n=1,
    )

    # Relaxed check of results, just to make sure results are not way
    # off
    assert np.isclose(xmap.scores.mean(), 0.1887, atol=1e-4)
