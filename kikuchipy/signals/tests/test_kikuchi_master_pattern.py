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
from orix.crystal_map import Phase

import kikuchipy as kp


class TestKikuchiMasterPattern:
    def test_set_signal_type(self):
        """Custom properties of projection, hemisphere and phase carry
        over in a call to `set_signal_type()`.
        """
        phase = Phase("a", point_group="m-3m")
        s = kp.signals.EBSDMasterPattern(
            np.zeros((2, 10, 11, 11)),
            projection="lambert",
            hemisphere="both",
            phase=phase,
        )

        # Carry over
        s.set_signal_type("ECPMasterPattern")
        assert s.phase.name == "a"
        assert np.may_share_memory(s.phase.point_group.data, phase.point_group.data)
        assert s.projection == "lambert"
        assert s.hemisphere == "both"

        # Does not carry over
        s.set_signal_type("EBSD")
        assert not hasattr(s, "_phase")
