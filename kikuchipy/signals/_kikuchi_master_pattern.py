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

import copy
from typing import Tuple, Union

from hyperspy._lazy_signals import LazySignal2D
from hyperspy._signals.signal2d import Signal2D
import numpy as np
from orix.crystal_map import Phase

from kikuchipy.signals._common_image import CommonImage


class KikuchiMasterPattern(CommonImage, Signal2D):
    """Common class for simulated Kikuchi master patterns.

    This class is is not meant to be used directly, see the derived
    classes :class:`~kikuchipy.signals.EBSDMasterPattern` (electron
    backscatter diffraction),
    :class:`~kikuchipy.signals.TKDMasterPattern` (transmission kikuchi
    diffraction), and :class:`~kikuchipy.signals.ECPMasterPattern`
    (electron channeling pattern).

    Attributes
    ----------
    projection : str
        Which projection the pattern is in, "stereographic" or
        "lambert".
    hemisphere : str
        Which hemisphere the data contains: "north", "south" or "both".
    phase : orix.crystal_map.phase_list.Phase
        Phase describing the crystal structure used in the master
        pattern simulation.
    """

    _signal_type = "KikuchiMasterPattern"
    _alias_signal_types = ["kikuchi_master_pattern"]
    _lazy = False

    # ---------------------- Custom properties ----------------------- #

    phase = Phase()
    projection = None
    hemisphere = None

    def __init__(self, *args, **kwargs):
        """Create an :class:`~kikuchipy.signals.KikuchiMasterPattern`
        instance from a :class:`hyperspy.signals.Signal2D` or a
        :class:`numpy.ndarray`. See the docstring of
        :class:`hyperspy.signal.BaseSignal` for optional input
        parameters.
        """
        Signal2D.__init__(self, *args, **kwargs)
        self.phase = kwargs.pop("phase", Phase())
        self.projection = kwargs.pop("projection", None)
        self.hemisphere = kwargs.pop("hemisphere", None)

    # ------ Methods overwritten from hyperspy.signals.Signal2D ------ #

    def deepcopy(self):
        new = super().deepcopy()
        new.phase = self.phase.deepcopy()
        new.projection = copy.deepcopy(self.projection)
        new.hemisphere = copy.deepcopy(self.hemisphere)
        return new

    # ------------------------ Private methods ----------------------- #

    def _get_master_pattern_arrays_from_energy(
        self, energy: Union[int, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return northern and southern master patterns created with a
        single, given energy.

        Parameters
        ----------
        energy
            Acceleration voltage in kV. If only a single energy is
            present in the signal, this will be returned no matter its
            energy.

        Returns
        -------
        master_north, master_south
            Northern and southern hemispheres of master pattern.
        """
        if "energy" in [i.name for i in self.axes_manager.navigation_axes]:
            master_patterns = self.inav[float(energy)].data
        else:  # Assume a single energy
            master_patterns = self.data
        if self.hemisphere == "both":
            master_north, master_south = master_patterns
        else:
            master_north = master_south = master_patterns
        return master_north, master_south

    def _is_suitable_for_projection(self, raise_if_not: bool = False):
        """Check whether the master pattern is suitable for projection
        onto a detector and return a bool or raise an error message if
        desired.
        """
        suitable = True
        error = None
        if self.projection != "lambert":
            error = NotImplementedError(
                "Master pattern must be in the square Lambert projection"
            )
            suitable = False
        pg = self.phase.point_group
        if pg is None:
            error = AttributeError(
                "Master pattern `phase` attribute must have a valid point group"
            )
            suitable = False
        elif self.hemisphere != "both" and not pg.contains_inversion:
            error = AttributeError(
                "For point groups without inversion symmetry, like the current "
                f"{pg.name}, both hemispheres must be present in the master pattern "
                "signal"
            )
            suitable = False
        if not suitable and raise_if_not:
            raise error
        else:
            return suitable


class LazyKikuchiMasterPattern(KikuchiMasterPattern, LazySignal2D):
    """Lazy implementation of the :class:`KikuchiMasterPattern` class.

    This class extends HyperSpy's LazySignal2D class for Kikuchi master
    patterns. Methods inherited from HyperSpy can be found in the
    HyperSpy user guide. See docstring of :class:`KikuchiMasterPattern`
    for attributes and methods.
    """

    _lazy = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
