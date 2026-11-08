#
# Copyright 2019-2026 the kikuchipy developers
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
#

from __future__ import annotations

from kikuchipy.signals._kikuchi_master_pattern import KikuchiMasterPattern
from kikuchipy.signals._kikuchipy_signal import LazyKikuchipySignal2D


class ECPMasterPattern(KikuchiMasterPattern):
    """Simulated Electron Channeling Pattern (ECP) master pattern.

    This class extends HyperSpy's Signal2D class for ECP master
    patterns.

    See the documentation of
    :class:`~hyperspy._signals.signal2d.Signal2D` for the list of
    inherited attributes and methods.

    Parameters
    ----------
    *args
        See :class:`~hyperspy._signals.signal2d.Signal2D`.
    hemisphere : str
        Which hemisphere the data contains, either "upper", "lower", or
        "both".
    phase : ~orix.crystal_map.Phase
        The phase describing the crystal structure used in the master
        pattern simulation.
    projection : str
        Which projection the pattern is in, "stereographic" or
        "lambert".
    **kwargs
        See :class:`~hyperspy._signals.signal2d.Signal2D`.
    """

    _signal_type = "ECPMasterPattern"
    _alias_signal_types = ["ecp_master_pattern"]


class LazyECPMasterPattern(LazyKikuchipySignal2D, ECPMasterPattern):
    """Lazy implementation of
    :class:`~kikuchipy.signals.ECPMasterPattern`.

    See the documentation of ``ECPMasterPattern`` for attributes and
    methods.

    This class extends HyperSpy's
    :class:`~hyperspy.signals.LazySignal2D` class for ECP master
    patterns. See the documentation of that class for how to create this
    signal and the list of inherited attributes and methods.
    """

    def compute(self, *args, **kwargs) -> None:
        super().compute(*args, **kwargs)
