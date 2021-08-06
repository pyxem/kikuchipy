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
import gc
import sys
from typing import Union

import dask.array as da
from dask.diagnostics import ProgressBar
from hyperspy._lazy_signals import LazySignal2D
from hyperspy._signals.signal2d import Signal2D
import numpy as np
from orix.crystal_map import CrystalMap, Phase, PhaseList
from orix.quaternion import Rotation
from orix.vector import Vector3d

from kikuchipy.detectors.ebsd_detector import EBSDDetector
from kikuchipy.signals import LazyEBSD, EBSD
from kikuchipy.signals._common_image import CommonImage
from kikuchipy.signals.util._dask import get_chunking
from kikuchipy.signals.util._master_pattern import (
    _get_direction_cosines_from_detector,
    _get_patterns_chunk,
)


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
        Which projection the pattern is in, "stereographic" or
        "lambert".
    hemisphere : str
        Which hemisphere the data contains: "north", "south" or "both".
    phase : orix.crystal_map.phase_list.Phase
        Phase describing the crystal structure used in the master
        pattern simulation.
    """

    _signal_type = "EBSDMasterPattern"
    _alias_signal_types = ["ebsd_master_pattern", "master_pattern"]
    _lazy = False

    # ---------------------- Custom properties ----------------------- #

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

    def get_patterns(
        self,
        rotations: Rotation,
        detector: EBSDDetector,
        energy: Union[int, float],
        dtype_out: type = np.float32,
        compute: bool = False,
        **kwargs,
    ) -> Union[EBSD, LazyEBSD]:
        """Return a dictionary of EBSD patterns projected onto a
        detector from a master pattern in the square Lambert
        projection :cite:`callahan2013dynamical`, for a set of crystal
        rotations relative to the EDAX TSL sample reference frame (RD,
        TD, ND) and a fixed detector-sample geometry.

        Parameters
        ----------
        rotations
            Set of crystal rotations to get patterns from. The shape of
            this object, a maximum of two dimensions, determines the
            navigation shape of the output signal.
        detector
            EBSD detector describing the detector dimensions and the
            detector-sample geometry with a single, fixed
            projection/pattern center.
        energy
            Acceleration voltage, in kV, used to simulate the desired
            master pattern to create a dictionary from.
        dtype_out
            Data type of the returned patterns, by default np.float32.
        compute
            Whether to return a lazy result, by default False. For more
            information see :func:`~dask.array.Array.compute`.
        kwargs
            Keyword arguments passed to
            :func:`~kikuchipy.signals.util.get_chunking` to control the
            number of chunks the dictionary creation and the output data
            array is split into. Only `chunk_shape`, `chunk_bytes` and
            `dtype_out` (to `dtype`) are passed on.

        Returns
        -------
        EBSD or LazyEBSD
            Signal with navigation and signal shape equal to the
            rotation object's and detector shape, respectively.

        Notes
        -----
        If the master pattern phase has a non-centrosymmetric point
        group, both the northern and southern hemispheres must be
        provided. For more details regarding the reference frame visit
        the reference frame user guide.
        """
        if self.projection != "lambert":
            raise NotImplementedError(
                "Master pattern must be in the square Lambert projection"
            )
        if len(detector.pc) > 1:
            raise NotImplementedError(
                "Detector must have exactly one projection center"
            )

        # Get suitable chunks when iterating over the rotations. The
        # signal axes are not chunked.
        nav_shape = rotations.shape
        nav_dim = len(nav_shape)
        if nav_dim > 2:
            raise ValueError(
                "The rotations object can only have one or two dimensions, but an "
                f"object with {nav_dim} was passed"
            )
        data_shape = nav_shape + detector.shape
        chunks = get_chunking(
            data_shape=data_shape,
            nav_dim=nav_dim,
            sig_dim=len(detector.shape),
            chunk_shape=kwargs.pop("chunk_shape", None),
            chunk_bytes=kwargs.pop("chunk_bytes", None),
            dtype=dtype_out,
        )

        # Get the master pattern arrays created by a desired energy
        north_slice = ()
        if "energy" in [i.name for i in self.axes_manager.navigation_axes]:
            energies = self.axes_manager["energy"].axis
            north_slice += ((np.abs(energies - energy)).argmin(),)
        south_slice = north_slice
        if self.hemisphere == "both":
            north_slice = (0,) + north_slice
            south_slice = (1,) + south_slice
        elif not self.phase.point_group.contains_inversion:
            raise AttributeError(
                "For crystals of point groups without inversion symmetry, like the "
                "current {self.phase.point_group.name}, both hemispheres must be "
                "present in the master pattern signal"
            )
        master_north = self.data[north_slice]
        master_south = self.data[south_slice]

        # Whether to rescale pattern intensities after projection
        rescale = False
        if dtype_out != np.float32:
            rescale = True

        # Get direction cosines for each detector pixel relative to the
        # source point
        dc = _get_direction_cosines_from_detector(detector)
        dc_v = Vector3d(dc)

        # Get dask array from rotations
        r_da = da.from_array(rotations.data, chunks=chunks[:nav_dim] + (-1,))

        # Which axes to drop and add when iterating over the rotations
        # dask array to produce the EBSD signal array, i.e. drop the
        # (4,)-shape quaternion axis and add detector shape axes, e.g.
        # (60, 60)
        if nav_dim == 1:
            drop_axis = 1
            new_axis = (1, 2)
        else:  # nav_dim == 2
            drop_axis = 2
            new_axis = (2, 3)

        # Project simulated patterns onto detector
        npx, npy = self.axes_manager.signal_shape
        scale = (npx - 1) / 2
        simulated = r_da.map_blocks(
            _get_patterns_chunk,
            dc=dc_v,
            master_north=master_north,
            master_south=master_south,
            npx=npx,
            npy=npy,
            scale=scale,
            rescale=rescale,
            dtype_out=dtype_out,
            drop_axis=drop_axis,
            new_axis=new_axis,
            chunks=chunks,
            dtype=dtype_out,
        )

        # Add crystal map and detector to keyword arguments
        kwargs = dict(
            xmap=CrystalMap(phase_list=PhaseList(self.phase), rotations=rotations),
            detector=detector,
        )

        # Specify navigation and signal axes for signal initialization
        names = ["y", "x", "dy", "dx"]
        scales = np.ones(4)
        ndim = simulated.ndim
        if ndim == 3:
            names = names[1:]
            scales = scales[1:]
        axes = [
            dict(
                size=data_shape[i],
                index_in_array=i,
                name=names[i],
                scale=scales[i],
                offset=0.0,
                units="px",
            )
            for i in range(ndim)
        ]

        if compute:
            with ProgressBar():
                print(
                    f"Creating a dictionary of {nav_shape} simulated patterns:",
                    file=sys.stdout,
                )
                patterns = simulated.compute()
            out = EBSD(patterns, axes=axes, **kwargs)
        else:
            out = LazyEBSD(simulated, axes=axes, **kwargs)
        gc.collect()

        return out

    # ------ Methods overwritten from hyperspy.signals.Signal2D ------ #
    def deepcopy(self):
        new = super().deepcopy()
        new.phase = self.phase.deepcopy()
        new.projection = copy.deepcopy(self.projection)
        new.hemisphere = copy.deepcopy(self.hemisphere)
        return new


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
