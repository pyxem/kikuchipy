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


import sys
from typing import Optional, Union

import dask.array as da
from dask.diagnostics import ProgressBar
from hyperspy._lazy_signals import LazySignal2D
from hyperspy._signals.signal2d import Signal2D
import numpy as np
from orix.crystal_map import CrystalMap, Phase, PhaseList
from orix.vector import Vector3d
from orix.quaternion import Rotation

from kikuchipy.detectors.ebsd_detector import EBSDDetector
from kikuchipy.pattern import rescale_intensity
from kikuchipy.projections.lambert_projection import LambertProjection
from kikuchipy.signals import LazyEBSD, EBSD
from kikuchipy.signals._common_image import CommonImage
from kikuchipy.signals.util._dask import get_chunking


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
        the reference frame user guide at:
        https://kikuchipy.org/en/latest/reference_frames.html.
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
                "The rotations object can only have one or two dimensions, but "
                f"an object with {nav_dim} was passed"
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
                "For crystals of point groups without inversion symmetry, like "
                f"the current {self.phase.point_group.name}, both hemispheres "
                "must be present in the master pattern signal"
            )
        master_north = self.data[north_slice]
        master_south = self.data[south_slice]

        # Whether to rescale pattern intensities after projection
        rescale = False
        if dtype_out != np.float32:
            rescale = True

        # Get direction cosines for each detector pixel relative to the
        # source point
        dc = _get_direction_cosines(detector)

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
            dc=dc,
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
            xmap=CrystalMap(
                phase_list=PhaseList(self.phase), rotations=rotations,
            ),
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

        return out


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


def _get_direction_cosines(detector: EBSDDetector) -> Vector3d:
    """Get the direction cosines between the detector and sample as done
    in EMsoft and :cite:`callahan2013dynamical`.

    Parameters
    ----------
    detector : EBSDDetector
        EBSDDetector object with a certain detector geometry and one
        projection center.

    Returns
    -------
    Vector3d
        Direction cosines for each detector pixel.
    """
    pc = detector.pc_emsoft()
    xpc = pc[..., 0]
    ypc = pc[..., 1]
    L = pc[..., 2]

    # Detector coordinates in microns
    det_x = (
        -((-xpc - (1.0 - detector.ncols) * 0.5) - np.arange(0, detector.ncols))
        * detector.px_size
    )
    det_y = (
        (ypc - (1.0 - detector.nrows) * 0.5) - np.arange(0, detector.nrows)
    ) * detector.px_size

    # Auxilliary angle to rotate between reference frames
    theta_c = np.radians(detector.tilt)
    sigma = np.radians(detector.sample_tilt)

    alpha = (np.pi / 2) - sigma + theta_c
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    # TODO: Enable detector azimuthal angle
    omega = np.radians(0)  # angle between normal of sample and detector
    cw = np.cos(omega)
    sw = np.sin(omega)

    r_g_array = np.zeros((detector.nrows, detector.ncols, 3))

    Ls = -sw * det_x + L * cw
    Lc = cw * det_x + L * sw

    i, j = np.meshgrid(
        np.arange(detector.nrows - 1, -1, -1),
        np.arange(detector.ncols),
        indexing="ij",
    )

    r_g_array[..., 0] = det_y[i] * ca + sa * Ls[j]
    r_g_array[..., 1] = Lc[j]
    r_g_array[..., 2] = -sa * det_y[i] + ca * Ls[j]
    r_g = Vector3d(r_g_array)

    return r_g.unit


def _get_lambert_interpolation_parameters(
    rotated_direction_cosines: Vector3d,
    npx: int,
    npy: int,
    scale: Union[int, float],
) -> tuple:
    """Get interpolation parameters in the square Lambert projection, as
    implemented in EMsoft.

    Parameters
    ----------
    rotated_direction_cosines
        Rotated direction cosines vector.
    npx
        Number of pixels on the master pattern in the x direction.
    npy
        Number of pixels on the master pattern in the y direction.
    scale
        Factor to scale up from the square Lambert projection to the
        master pattern.

    Returns
    -------
    nii : numpy.ndarray
        Row coordinate of a point.
    nij : numpy.ndarray
        Column coordinate of a point.
    niip : numpy.ndarray
        Row coordinate of neighboring point.
    nijp : numpy.ndarray
        Column coordinate of a neighboring point.
    di : numpy.ndarray
        Row interpolation weight factor.
    dj : numpy.ndarray
        Column interpolation weight factor.
    dim : numpy.ndarray
        Row interpolation weight factor.
    djm : numpy.ndarray
        Column interpolation weight factor.
    """
    # Direction cosines to the square Lambert projection
    xy = (
        scale
        * LambertProjection.project(rotated_direction_cosines)
        / (np.sqrt(np.pi / 2))
    )

    i = xy[..., 1]
    j = xy[..., 0]
    nii = (i + scale).astype(int)
    nij = (j + scale).astype(int)
    niip = nii + 1
    nijp = nij + 1
    niip = np.where(niip < npx, niip, nii).astype(int)
    nijp = np.where(nijp < npy, nijp, nij).astype(int)
    nii = np.where(nii < 0, niip, nii).astype(int)
    nij = np.where(nij < 0, nijp, nij).astype(int)
    di = i - nii + scale
    dj = j - nij + scale
    dim = 1.0 - di
    djm = 1.0 - dj

    return nii, nij, niip, nijp, di, dj, dim, djm


def _get_patterns_chunk(
    rotations_array: np.ndarray,
    dc: Vector3d,
    master_north: np.ndarray,
    master_south: np.ndarray,
    npx: int,
    npy: int,
    scale: Union[int, float],
    rescale: bool,
    dtype_out: Optional[type] = np.float32,
) -> np.ndarray:
    """Get the EBSD patterns on the detector for each rotation in the
    chunk. Each pattern is found by a bi-quadratic interpolation of the
    master pattern as described in EMsoft.

    Parameters
    ----------
    rotations_array
        Array of rotations of shape (..., 4) for a given chunk in
        quaternions.
    dc
        Direction cosines unit vector between detector and sample.
    master_north
        Northern hemisphere of the master pattern.
    master_south
        Southern hemisphere of the master pattern.
    npx
        Number of pixels in the x-direction on the master pattern.
    npy
        Number of pixels in the y-direction on the master pattern.
    scale
        Factor to scale up from square Lambert projection to the master
        pattern.
    rescale
        Whether to call rescale_intensities() or not.
    dtype_out
        Data type of the returned patterns, by default np.float32.

    Returns
    -------
    numpy.ndarray
        3D or 4D array with simulated patterns.
    """
    rotations = Rotation(rotations_array)
    rotations_shape = rotations_array.shape[:-1]
    simulated = np.empty(shape=rotations_shape + dc.shape, dtype=dtype_out)
    for i in np.ndindex(rotations_shape):
        rotated_dc = rotations[i] * dc
        (
            nii,
            nij,
            niip,
            nijp,
            di,
            dj,
            dim,
            djm,
        ) = _get_lambert_interpolation_parameters(
            rotated_direction_cosines=rotated_dc, npx=npx, npy=npy, scale=scale,
        )
        pattern = np.where(
            rotated_dc.z >= 0,
            (
                master_north[nii, nij] * dim * djm
                + master_north[niip, nij] * di * djm
                + master_north[nii, nijp] * dim * dj
                + master_north[niip, nijp] * di * dj
            ),
            (
                master_south[nii, nij] * dim * djm
                + master_south[niip, nij] * di * djm
                + master_south[nii, nijp] * dim * dj
                + master_south[niip, nijp] * di * dj
            ),
        )
        if rescale:
            pattern = rescale_intensity(pattern, dtype_out=dtype_out)
        simulated[i] = pattern
    return simulated
