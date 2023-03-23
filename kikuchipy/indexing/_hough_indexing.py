# Copyright 2019-2023 The kikuchipy developers
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

"""Wrapping of PyEBSDIndex functionality for Hough indexing of
EBSD patterns.

Most of these tools are private and not meant to be used by users.
"""

from time import time
from typing import List, Optional, Tuple, Union

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
from orix.crystal_map import create_coordinate_arrays, CrystalMap, PhaseList
from orix.quaternion import Rotation

from kikuchipy import _pyebsdindex_installed


def xmap_from_hough_indexing_data(
    data: np.ndarray,
    phase_list: PhaseList,
    data_index: int = -1,
    navigation_shape: Optional[tuple] = None,
    step_sizes: Optional[tuple] = None,
    scan_unit: str = "px",
) -> CrystalMap:
    """Convert Hough indexing result array from :mod:`pyebsdindex` to a
    :class:`~orix.crystal_map.CrystalMap`.

    Parameters
    ----------
    data
        Array with the following data type field names: ``"quat"``,
        ``"phase"``, ``"fit"``, ``"cm"``, ``"pq"`` and ``"nmatch"``.
    phase_list
        List of phases. If ``data_index=-1``, the phase IDs in the list
        must match the phase IDs in ``data[-1]["phase"]``. If
        ``data_index`` is another ID, it must be in the phase list.
    data_index
        Index into ``data`` of which to return a crystal map from.
        Default is ``-1``, which returns the most probable (best)
        solutions in each map point. Other options depend on the number
        of phases used in indexing, and starts with ``0``.
    navigation_shape
        Navigation shape of resulting map. If not given, a 1D crystal
        map is returned. Maximum of two dimensions.
    step_sizes
        Step sizes in each navigation direction. If not given, a step
        size of ``1`` is used in each direction.
    scan_unit
        Scan unit of map. If not given, it is not set.

    Returns
    -------
    xmap
        Crystal map.
    """
    if navigation_shape is None:
        navigation_shape = (data.shape[1],)
    elif len(navigation_shape) > 2 or not all(
        isinstance(n_i, int) for n_i in navigation_shape
    ):
        raise ValueError("`nav_shape` cannot be a tuple of more than two integers")

    coords, _ = create_coordinate_arrays(navigation_shape, step_sizes)

    phase_list_id = phase_list.ids
    if data_index != -1 and data_index not in phase_list_id:
        raise ValueError(f"`phase_list` IDs {phase_list_id} must contain {data_index}")

    data_index = data[data_index]

    phase_id = data_index["phase"]
    not_indexed = phase_id == -1

    rot_arr = data_index["quat"]
    rot_arr[not_indexed] = [1, 0, 0, 0]
    rot = Rotation(rot_arr)

    xmap = CrystalMap(
        rotations=rot,
        phase_id=phase_id,
        phase_list=phase_list,
        prop=dict(
            fit=data_index["fit"],
            cm=data_index["cm"],
            pq=data_index["pq"],
            nmatch=data_index["nmatch"],
        ),
        scan_unit=scan_unit,
        **coords,
    )

    return xmap


def _get_indexer_from_detector(
    phase_list: PhaseList,
    shape: tuple,
    pc: np.ndarray,
    sample_tilt: float,
    tilt: float,
    **kwargs,
) -> "EBSDIndexer":
    """Return a PyEBSDIndex EBSD indexer.

    Parameters
    ----------
    phase_list
        List of supported phases, only ``"FCC"`` and/or ``"BCC"``.
    shape
        Detector shape (n rows, n columns).
    pc
        Projection center(s).
    sample_tilt
        Sample tilt in degrees.
    tilt
        Detector tilt in degrees.
    **kwargs
        Keyword arguments passed to
        :class:`~pyebsdindex.ebsd_index.EBSDIndexer`.

    Returns
    -------
    indexer
        EBSD indexer.

    Notes
    -----
    Requires that :mod:`pyebsdindex` is installed, which is an optional
    dependency of kikuchipy. See :ref:`optional-dependencies` for
    details.
    """
    if not _pyebsdindex_installed:  # pragma: no cover
        raise ValueError(
            "pyebsdindex must be installed. Install with pip install pyebsdindex. "
            "See https://kikuchipy.org/en/stable/user/installation.html for "
            "details."
        )

    from pyebsdindex.ebsd_index import EBSDIndexer

    phase_list_pei = _get_pyebsdindex_phaselist(phase_list)

    indexer = EBSDIndexer(
        phaselist=phase_list_pei,
        vendor="KIKUCHIPY",
        PC=pc,
        sampleTilt=sample_tilt,
        camElev=tilt,
        patDim=shape,
        **kwargs,
    )

    return indexer


def _hough_indexing(
    patterns: Union[np.ndarray, da.Array],
    phase_list: PhaseList,
    nav_shape: tuple,
    step_sizes: tuple,
    indexer,
    chunksize: int,
    verbose: int,
) -> Tuple[CrystalMap, np.ndarray, np.ndarray]:
    """Perform Hough indexing with PyEBSDIndex.

    Function can only be called if PyEBSDIndex is installed.

    Parameters
    ----------
    patterns
        Array of patterns of shape (n patterns, n detector rows, n
        detector columns).
    phase_list
        Phase list to check.
    nav_shape
        Navigation shape.
    step_sizes
        Navigation step sizes.
    indexer : pyebsdindex.ebsd_index.EBSDIndexer
        Indexer instance.
    verbose
        Whether to print indexing information. 0 - no output, 1 -
        timings, 2 - timings and the Hough transform of the first
        pattern with detected bands highlighted.

    Returns
    -------
    xmap
        Crystal map with best matching results.
    index_data
        Array of index data. See
        :meth:`~pyebsdindex.ebsd_index.EBSDIndexer.index_pats` for
        details.
    band_data
        Array of detected band data. See
        :meth:`~pyebsdindex.ebsd_index.EBSDIndexer.index_pats` for
        details.
    """
    n_patterns = patterns.shape[0]

    info_message = _get_info_message(n_patterns, chunksize, indexer)
    print(info_message)

    if verbose == 2:
        plt.figure()

    tic = time()
    index_data, band_data, _, _ = indexer.index_pats(
        patsin=patterns, verbose=verbose, chunksize=chunksize
    )
    toc = time()
    patterns_per_second = n_patterns / (toc - tic)
    print(f"  Indexing speed: {patterns_per_second:.5f} patterns/s")

    xmap = xmap_from_hough_indexing_data(
        data=index_data,
        phase_list=phase_list,
        navigation_shape=nav_shape,
        step_sizes=step_sizes,
    )

    return xmap, index_data, band_data


def _get_pyebsdindex_phaselist(phase_list: PhaseList) -> List[str]:
    """Return a phase list compatible with PyEBSDIndex or raise a
    ``ValueError`` if the given phase list is incompatible.

    Parameters
    ----------
    phase_list
        Phase list to convert to one compatible with PyEBSDIndex.

    Returns
    -------
    phase_list_pei
        Phase list compatible with PyEBSDIndex.

    Raises
    ------
    ValueError
        Raised if the phase list has more than two phases, if one of the
        phases are not FCC or BCC, or if both phases are either FCC or
        BCC.
    """
    compatible = True
    error_msg = None

    msg_supported_phases = (
        "Hough indexing only supports indexing of generic face-centered cubic (FCC) and"
        " body-centered cubic (BCC) phases.\nThus the phase list can only contain one "
        "FCC and/or one BCC phase."
    )

    if phase_list.size > 2:
        compatible = False
        error_msg = msg_supported_phases

    sg = phase_list.space_groups
    centering = None
    if compatible and any(sg_i is None for sg_i in sg):
        compatible = False
        error_msg = (
            "Space group for each phase must be set, otherwise the Bravais lattice(s) "
            "cannot be determined."
        )
    else:
        centering = [sg_i.short_name[0] for sg_i in sg]
        if "I" in centering:
            centering[centering.index("I")] = "B"
        allowed_centering = [["F"], ["B"], ["F", "B"], ["B", "F"]]
        if compatible and centering not in allowed_centering:
            compatible = False
            error_msg = msg_supported_phases

    if not compatible:
        raise ValueError(error_msg)
    else:
        phase_list_pei = [c + "CC" for c in centering]
        return phase_list_pei


def _indexer_is_compatible_with_kikuchipy(
    indexer,
    sig_shape: tuple,
    nav_size: Optional[int] = None,
    check_pc: bool = True,
    raise_if_not: bool = False,
) -> bool:
    """Check whether an indexer is compatible with kikuchipy or raise a
    ``ValueError`` if it is not.

    Parameters
    ----------
    indexer : pyebsdindex.ebsd_index.EBSDIndexer
        Indexer instance.
    sig_shape
        Signal shape (n detector rows, n detector columns), checked
        against ``indexer.patDim``.
    nav_size
        Number of patterns. If given, it is checked against
        ``indexer.PC`` if ``check_pc=True``.
    check_pc
        Whether to check ``indexer.PC`` (default is ``True``).
    raise_if_not
        Whether to raise a ``ValueError`` if the indexer is not
        compatible with the signal. Default is ``False``.

    Returns
    -------
    compatible
        Whether the indexer is compatible with the signal.
    """
    compatible = True
    error_msg = None

    # PC and orientation convention
    if indexer.vendor.lower() != "kikuchipy":
        compatible = False
        error_msg = f"`indexer.vendor` must be 'kikuchipy', but was {indexer.vendor}."

    # Detector shape
    sig_shape_indexer = tuple(indexer.bandDetectPlan.patDim)
    if compatible and sig_shape != sig_shape_indexer:
        compatible = False
        error_msg = (
            f"Indexer signal shape {sig_shape_indexer} must be equal to the signal "
            f"shape {sig_shape}."
        )

    # Projection center shape and size
    if check_pc:
        pc_shape = np.shape(indexer.PC)
        allowed_shapes = [(3,)]
        allowed_shapes_str = "(3,)"
        if nav_size is not None:
            allowed_shapes.append((nav_size, 3))
            allowed_shapes_str += f" or ({nav_size}, 3)"
        if compatible and (len(pc_shape) > 2 or pc_shape not in allowed_shapes):
            compatible = False
            error_msg = (
                f"`indexer.PC` must be an array of shape {allowed_shapes_str}, but was "
                f"{pc_shape} instead."
            )

    phase_list_pei = indexer.phaselist
    allowed_lists = [["FCC"], ["BCC"], ["FCC", "BCC"], ["BCC", "FCC"]]
    if compatible and phase_list_pei not in allowed_lists:
        compatible = False
        error_msg = f"`indexer.phaselist` must be one of {allowed_lists}"

    if raise_if_not and not compatible:
        raise ValueError(error_msg)
    else:
        return compatible


def _get_info_message(nav_size: int, chunksize: int, indexer: "EBSDIndexer") -> str:
    from kikuchipy import _pyopencl_context_available

    info = (
        "Hough indexing with PyEBSDIndex information:\n"
        f"  PyOpenCL: {_pyopencl_context_available}\n"
        "  Projection center (Bruker"
    )

    n_chunks = int(np.ceil(nav_size / chunksize))
    pc = indexer.PC.squeeze()
    if pc.size > 3:
        pc = pc.mean(0)
        info += ", mean"
    pc = pc.round(4)
    info += (
        f"): {tuple(pc)}\n" f"  Indexing {nav_size} pattern(s) in {n_chunks} chunk(s)"
    )

    return info


def _optimize_pc(
    pc0: List[float],
    patterns: Union[np.ndarray, da.Array],
    indexer: "EBSDIndexer",
    batch: bool,
    method: str,
) -> np.ndarray:
    if method == "pso":
        from pyebsdindex.pcopt import optimize_pso as optimize_func
    else:
        from pyebsdindex.pcopt import optimize as optimize_func
    return optimize_func(pats=patterns, indexer=indexer, PC0=pc0, batch=batch)
