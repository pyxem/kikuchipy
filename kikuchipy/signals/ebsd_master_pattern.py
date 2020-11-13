# -*- coding: utf-8 -*-
# Copyright 2019-2020 The kikuchipy developers
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

from hyperspy._signals.signal2d import Signal2D
from hyperspy._lazy_signals import LazySignal2D
from orix.crystal_map import Phase

from kikuchipy.signals._common_image import CommonImage


class EBSDMasterPattern(CommonImage, Signal2D):
    """Simulated Electron Backscatter Diffraction (EBSD) master pattern.

    This class extends HyperSpy's Signal2D class for EBSD master
    patterns. Methods inherited from HyperSpy can be found in the
    HyperSpy user guide. See the docstring of
    :class:`hyperspy.signal.BaseSignal` for a list of additional
    attributes.

    Attributes
    ----------
    projection : str
        Which projection the pattern is in: "spherical" or "lambert".
    hemisphere : str
        Which hemisphere the data contains: "north", "south" or "both".
    phase : orix.crystal_map.phase_list.Phase
        Phase describing the crystal structure used in the master
        pattern simulation.
    """

    _signal_type = "EBSDMasterPattern"
    _alias_signal_types = ["ebsd_master_pattern", "master_pattern"]
    _lazy = False

    phase = Phase()
    projection = None
    hemisphere = None

    def __init__(self, *args, **kwargs):
        """Create an :class:`~kikuchipy.signals.EBSDMasterPattern`
        object from a :class:`hyperspy.signals.Signal2D` or a
        :class:`numpy.ndarray`.
        """
        Signal2D.__init__(self, *args, **kwargs)
        self.phase = kwargs.pop("phase", Phase())
        self.projection = kwargs.pop("projection", None)
        self.hemisphere = kwargs.pop("hemisphere", None)


class LazyEBSDMasterPattern(EBSDMasterPattern, LazySignal2D):
    """Lazy implementation of the :class:`EBSDMasterPattern` class.

    This class extends HyperSpy's LazySignal2D class for EBSD master
    patterns. Methods inherited from HyperSpy can be found in the
    HyperSpy user guide. See docstring of :class:`EBSDMasterPattern`
    for attributes and methods.
    """

    _lazy = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
