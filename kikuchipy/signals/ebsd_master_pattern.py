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


from typing import Optional, Union

import dask.array as da
from hyperspy._lazy_signals import LazySignal2D
from hyperspy._signals.signal2d import Signal2D
import numpy as np
from orix.crystal_map import Phase
from orix.vector import Vector3d
from orix.quaternion import Rotation

from kikuchipy.detectors.ebsd_detector import EBSDDetector
from kikuchipy.pattern import rescale_intensity
from kikuchipy.projections.lambert_projection import LambertProjection
from kikuchipy.signals import LazyEBSD, EBSD
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

    def get_patterns(
        self,
        rotations: Rotation,
        detector: EBSDDetector,
        energy: int,
        n_chunk: Optional[int] = None,
        dtype_out: type = np.float32,
        compute: bool = False,
    ) -> Union[EBSD, LazyEBSD]:
        """Creates a dictionary of EBSD patterns for a sample in the
        EDAX TSL (RD, TD, ND) reference frame, given a set of local
        crystal lattice rotations and a detector model from a master
        pattern in the Lambert projection.

        Parameters
        ----------
        rotations : Rotation
            Set of unit cell rotations to get patterns from.
        detector : EBSDDetector
            EBSDDetector object describing the detector geometry with a
            single, fixed projection/pattern center.
        energy : int
            The wanted energy in the master pattern.
        n_chunk : int, optional
            The number of chunks the data should be split up into. By
            default, this is set so each chunk is around 100 MB.
        dtype_out : type, optional
            Data type of the returned patterns, by default np.float32.
        compute : bool, optional
            Whether or not the dask.compute() function should be called
            and read the patterns into memory, by default false.
            For more information see: :func:`dask.array.Array.compute`.

        Returns
        -------
        EBSD or LazyEBSD
            All the simulated EBSD patterns with the shape (number of
            rotations, detector pixels in x direction, detector pixels
            in y direction).

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
                "Method only supports master patterns in Lambert projection!"
            )
        pc = detector.pc_emsoft()
        if len(pc) > 1:
            raise ValueError(
                "Method only supports a single projection/pattern center!"
            )

        # 4 cases
        # Has energies, has hemis - Case 1
        if len(self.axes_manager.shape) == 4:
            energies = self.axes_manager["energy"].axis
            energy_index = (np.abs(energies - energy)).argmin()
            mpn = self.data[0, energy_index]
            mps = self.data[1, energy_index]
        # no energies, no hemis - Case 2
        elif len(self.axes_manager.shape) == 2:
            if not self.phase.point_group.contains_inversion:
                raise AttributeError(
                    "For phases without inversion symmetry, "
                    "both hemispheres must be in master pattern!"
                )
            mpn = self.data
            mps = mpn
        else:
            try:  # has energies, no hemi - Case 3
                energies = self.axes_manager["energy"].axis
                if not self.phase.point_group.contains_inversion:
                    raise AttributeError(
                        "For phases without inversion symmetry, both"
                        "hemispheres must be in master pattern!"
                    )
                energy_index = (np.abs(energies - energy)).argmin()
                mpn = self.data[energy_index]
                mps = mpn
            except ValueError:  # no energies, yes hemi - Case 4
                mpn = self.data[0]
                mps = self.data[1]

        npx, npy = self.axes_manager.signal_shape
        dc = _get_direction_cosines(detector)
        n = rotations.size
        det_y, det_x = detector.shape
        dtype_out = dtype_out

        if not n_chunk:
            n_chunk = _min_number_of_chunks(detector.shape, n, dtype_out)

        out_shape = (n, det_y, det_x)
        chunks = (int(abs(np.ceil(n / n_chunk))), det_y, det_x)

        rescale = False
        if dtype_out != np.float32:
            rescale = True

        r_da = da.from_array(rotations.data, chunks=(chunks[0], -1))

        simulated = r_da.map_blocks(
            _get_patterns_chunk,
            dc=dc,
            master_north=mpn,
            master_south=mps,
            npx=npx,
            npy=npy,
            rescale=rescale,
            dtype_out=dtype_out,
            drop_axis=1,
            new_axis=(1, 2),
            chunks=chunks,
            dtype=dtype_out,
        )

        names = ["x", "dy", "dx"]
        scales = np.ones(3)

        # Create axis objects for each axis
        axes = [
            {
                "size": out_shape[i],
                "index_in_array": i,
                "name": names[i],
                "scale": scales[i],
                "offset": 0.0,
                "units": "px",
            }
            for i in range(simulated.ndim)
        ]
        if compute:
            return EBSD(simulated.compute(), axes=axes)
        return LazyEBSD(simulated, axes=axes)


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
    in EMsoft and [Callahan2013]_.

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
    scale: Union[int, float],
    npx: int,
    npy: int,
) -> tuple:
    """Get Lambert interpolation parameters as described in EMsoft.

    Parameters
    ----------
    rotated_direction_cosines : Vector3d
        Rotated direction cosines vector.
    scale : int
        Factor to scale up from Rosca-Lambert projection to the master
        pattern.
    npx : int
        Number of pixels on the master pattern in the x direction.
    npy : int
        Number of pixels on the master pattern in the y direction.

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
    # Direction cosines to Rosca-Lambert projection
    xy = (
        scale
        * LambertProjection.project(rotated_direction_cosines)
        / (np.sqrt(np.pi / 2))
    )

    i = xy[..., 0]
    j = xy[..., 1]
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
    r: Rotation,
    dc: Vector3d,
    master_north: np.ndarray,
    master_south: np.ndarray,
    npx: int,
    npy: int,
    rescale: bool,
    dtype_out: Optional[type] = np.float32,
) -> np.ndarray:
    """Get the EBSD patterns on the detector for each rotation in the
    chunk. Each pattern is found by a bi-quadratic interpolation of the
    master pattern as described in EMsoft.

    Parameters
    ----------
    r : Rotation
        Rotation object with all the rotations for a given chunk.
    dc : Vector3d
        Direction cosines unit vector between detector and sample.
    master_north : numpy.ndarray
        Northern hemisphere of the master pattern.
    master_south : numpy.ndarray
        Southern hemisphere of the master pattern.
    npx : int
        Number of pixels in the x-direction on the master pattern.
    npy: int
        Number of pixels in the y-direction on the master pattern.
    rescale : bool
        Whether to call rescale_intensities() or not.
    dtype_out : type, optional
        Data type of the returned patterns, by default np.float32.

    Returns
    -------
    numpy.ndarray
        (n, y, x) array containing all the simulated patterns.
    """
    m = r.shape[0]
    simulated = np.empty(shape=(m,) + dc.shape, dtype=dtype_out)
    scale_factor = (npx - 1) / 2
    for i in range(m):
        rot_dc = Rotation(r[i]) * dc
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
            rotated_direction_cosines=rot_dc,
            scale=scale_factor,
            npx=npx,
            npy=npy,
        )
        pattern = np.where(
            rot_dc.z >= 0,
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


def _min_number_of_chunks(
    detector_shape: tuple, n_rotations: int, dtype_out: type
) -> int:
    """Returns the minimum number of chunks required for our detector
    model and set of unit cell rotations so that each chunk is around
    100 MB.

    Parameters
    ----------
    detector_shape : tuple
        Shape of the detector in pixels.
    n_rotations : int
        Number of rotations.
    dtype_out : type
        Data type used for the simulated patterns.

    Returns
    -------
    int
       The minimum number of chunks required so each chunk is around
       100 MB.
    """
    dy, dx = detector_shape
    nbytes = dy * dx * n_rotations * np.dtype(dtype_out).itemsize
    nbytes_goal = 100e6  # 100 MB
    n_chunks = int(np.ceil(nbytes / nbytes_goal))
    return n_chunks
