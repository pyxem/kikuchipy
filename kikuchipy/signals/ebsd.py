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
import datetime
import gc
import numbers
import os
import sys
from typing import Union, List, Optional, Tuple, Iterable
import warnings

import dask.array as da
from dask.diagnostics import ProgressBar
from hyperspy._signals.signal2d import Signal2D
from hyperspy._lazy_signals import LazySignal2D
from hyperspy.learn.mva import LearningResults
from hyperspy.misc.utils import DictionaryTreeBrowser
from hyperspy.roi import BaseInteractiveROI
from hyperspy.api import interactive
from h5py import File
import numpy as np
from orix.crystal_map import CrystalMap
from scipy.ndimage import correlate, gaussian_filter
from skimage.util.dtype import dtype_range

from kikuchipy.detectors import EBSDDetector
from kikuchipy.filters.fft_barnes import _fft_filter, _fft_filter_setup
from kikuchipy.filters.window import Window
from kikuchipy.indexing._dictionary_indexing import _dictionary_indexing
from kikuchipy.indexing._refinement._refinement import (
    _refine_orientation,
    _refine_orientation_projection_center,
    _refine_projection_center,
)
from kikuchipy.indexing.similarity_metrics import metrics, SimilarityMetric
from kikuchipy.io._io import _save
from kikuchipy.pattern import chunk
from kikuchipy.pattern._pattern import (
    fft_frequency_vectors,
    fft_filter,
    _dynamic_background_frequency_space_setup,
)
from kikuchipy.signals.util._metadata import (
    ebsd_metadata,
    metadata_nodes,
    _update_phase_info,
    _write_parameters_to_dictionary,
)
from kikuchipy.signals.util._dask import (
    get_dask_array,
    get_chunking,
    _get_chunk_overlap_depth,
    _rechunk_learning_results,
    _update_learning_results,
)
from kikuchipy.signals.util._detector import _detector_is_compatible_with_signal
from kikuchipy.signals.util._crystal_map import _crystal_map_is_compatible_with_signal
from kikuchipy.signals.util._map_helper import (
    _get_neighbour_dot_product_matrices,
    _get_average_dot_product_map,
)
from kikuchipy.signals._common_image import CommonImage
from kikuchipy.signals.virtual_bse_image import VirtualBSEImage
from kikuchipy._util import deprecated


class EBSD(CommonImage, Signal2D):
    """Scan of Electron Backscatter Diffraction (EBSD) patterns.

    This class extends HyperSpy's Signal2D class for EBSD patterns, with
    common intensity processing methods and some analysis methods.

    Methods inherited from HyperSpy can be found in the HyperSpy user
    guide.

    See the docstring of :class:`hyperspy.signal.BaseSignal` for a list
    of attributes in addition to the ones listed below.
    """

    _signal_type = "EBSD"
    _alias_signal_types = ["electron_backscatter_diffraction"]
    _lazy = False

    def __init__(self, *args, **kwargs):
        """Create an :class:`~kikuchipy.signals.EBSD` instance from a
        :class:`hyperspy.signals.Signal2D` or a :class:`numpy.ndarray`.
        See the docstring of :class:`hyperspy.signal.BaseSignal` for
        optional input parameters.
        """
        Signal2D.__init__(self, *args, **kwargs)

        self._detector = kwargs.pop(
            "detector",
            EBSDDetector(
                shape=self.axes_manager.signal_shape,
                px_size=self.axes_manager.signal_axes[0].scale,
            ),
        )
        self._xmap = kwargs.pop("xmap", None)

        # Update metadata if object is initialised from numpy array
        if not self.metadata.has_item(metadata_nodes("ebsd")):
            md = self.metadata.as_dictionary()
            md.update(ebsd_metadata().as_dictionary())
            self.metadata = DictionaryTreeBrowser(md)
        if not self.metadata.has_item("Sample.Phases"):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
                self.set_phase_parameters()

    # ---------------------- Custom properties ----------------------- #

    @property
    def detector(self) -> EBSDDetector:
        """An :class:`~kikuchipy.detectors.ebsd_detector.EBSDDetector`
        describing the EBSD detector dimensions, the projection/pattern
        centre, and the detector-sample geometry.
        """
        return self._detector

    @detector.setter
    def detector(self, value: EBSDDetector):
        if _detector_is_compatible_with_signal(
            detector=value,
            navigation_shape=self.axes_manager.navigation_shape[::-1],
            signal_shape=self.axes_manager.signal_shape[::-1],
            raise_if_not=True,
        ):
            self._detector = value

    @property
    def xmap(self) -> CrystalMap:
        """A :class:`~orix.crystal_map.CrystalMap` containing the
        phases, unit cell rotations and auxiliary properties of the EBSD
        data set.
        """
        return self._xmap

    @xmap.setter
    def xmap(self, value: CrystalMap):
        if _crystal_map_is_compatible_with_signal(
            value, self.axes_manager.navigation_axes[::-1], raise_if_not=True
        ):
            self._xmap = value

    # ------------------------ Custom methods ------------------------ #

    @deprecated(since="0.5", removal="0.6")
    def set_experimental_parameters(
        self,
        detector=None,
        azimuth_angle=None,
        elevation_angle=None,
        sample_tilt=None,
        working_distance=None,
        binning=None,
        exposure_time=None,
        grid_type=None,
        gain=None,
        frame_number=None,
        frame_rate=None,
        scan_time=None,
        beam_energy=None,
        xpc=None,
        ypc=None,
        zpc=None,
        static_background=None,
        manufacturer=None,
        version=None,
        microscope=None,
        magnification=None,
    ):
        """Set experimental parameters in signal metadata.

        Parameters
        ----------
        azimuth_angle : float, optional
            Azimuth angle of the detector in degrees. If the azimuth is
            zero, the detector is perpendicular to the tilt axis.
        beam_energy : float, optional
            Energy of the electron beam in kV.
        binning : int, optional
            Camera binning.
        detector : str, optional
            Detector manufacturer and model.
        elevation_angle : float, optional
            Elevation angle of the detector in degrees. If the elevation
            is zero, the detector is perpendicular to the incident beam.
        exposure_time : float, optional
            Camera exposure time in µs.
        frame_number : float, optional
            Number of patterns integrated during acquisition.
        frame_rate : float, optional
            Frames per s.
        gain : float, optional
            Camera gain, typically in dB.
        grid_type : str, optional
            Scan grid type, only square grid is supported.
        manufacturer : str, optional
            Manufacturer of software used to collect patterns.
        microscope : str, optional
            Microscope used to collect patterns.
        magnification : int, optional
            Microscope magnification at which patterns were collected.
        sample_tilt : float, optional
            Sample tilt angle from horizontal in degrees.
        scan_time : float, optional
            Scan time in s.
        static_background : numpy.ndarray, optional
            Static background pattern.
        version : str, optional
            Version of software used to collect patterns.
        working_distance : float, optional
            Working distance in mm.
        xpc : float, optional
            Pattern centre horizontal coordinate with respect to
            detector centre, as viewed from the detector to the sample.
        ypc : float, optional
            Pattern centre vertical coordinate with respect to
            detector centre, as viewed from the detector to the sample.
        zpc : float, optional
            Specimen to scintillator distance.

        See Also
        --------
        ~kikuchipy.signals.EBSD.set_phase_parameters

        Examples
        --------
        >>> import kikuchipy as kp
        >>> s = kp.data.nickel_ebsd_small()
        >>> node = kp.signals.util.metadata_nodes("ebsd")
        >>> s.metadata.get_item(node + '.xpc')
        -5.64
        >>> s.set_experimental_parameters(xpc=0.50726)  # doctest: +SKIP
        >>> s.metadata.get_item(node + '.xpc')  # doctest: +SKIP
        0.50726
        """
        md = self.metadata
        sem_node, ebsd_node = metadata_nodes(["sem", "ebsd"])
        _write_parameters_to_dictionary(
            {
                "beam_energy": beam_energy,
                "magnification": magnification,
                "microscope": microscope,
                "working_distance": working_distance,
            },
            md,
            sem_node,
        )
        _write_parameters_to_dictionary(
            {
                "azimuth_angle": azimuth_angle,
                "binning": binning,
                "detector": detector,
                "elevation_angle": elevation_angle,
                "exposure_time": exposure_time,
                "frame_number": frame_number,
                "frame_rate": frame_rate,
                "gain": gain,
                "grid_type": grid_type,
                "manufacturer": manufacturer,
                "version": version,
                "sample_tilt": sample_tilt,
                "scan_time": scan_time,
                "xpc": xpc,
                "ypc": ypc,
                "zpc": zpc,
                "static_background": static_background,
            },
            md,
            ebsd_node,
        )

    @deprecated(since="0.5", removal="0.6")
    def set_phase_parameters(
        self,
        number=1,
        atom_coordinates=None,
        formula=None,
        info=None,
        lattice_constants=None,
        laue_group=None,
        material_name=None,
        point_group=None,
        setting=None,
        source=None,
        space_group=None,
        symmetry=None,
    ):
        """Set parameters for one phase in signal metadata.

        A phase node with default values is created if none is present
        in the metadata when this method is called.

        Parameters
        ----------
        number : int, optional
            Phase number.
        atom_coordinates : dict, optional
            Dictionary of dictionaries with one or more of the atoms in
            the unit cell, on the form `{'1': {'atom': 'Ni',
            'coordinates': [0, 0, 0], 'site_occupation': 1,
            'debye_waller_factor': 0}, '2': {'atom': 'O',... etc.`
            `debye_waller_factor` in units of nm^2, and
            `site_occupation` in range [0, 1].
        formula : str, optional
            Phase formula, e.g. 'Fe2' or 'Ni'.
        info : str, optional
            Whatever phase info the user finds relevant.
        lattice_constants : numpy.ndarray or list of floats, optional
            Six lattice constants a, b, c, alpha, beta, gamma.
        laue_group : str, optional
            Phase Laue group.
        material_name : str, optional
            Name of material.
        point_group : str, optional
            Phase point group.
        setting : int, optional
            Space group's origin setting.
        source : str, optional
            Literature reference for phase data.
        space_group : int, optional
            Number between 1 and 230.
        symmetry : int, optional
            Phase symmetry.

        See Also
        --------
        ~kikuchipy.signals.EBSD.set_experimental_parameters

        Examples
        --------
        >>> import kikuchipy as kp
        >>> s = kp.data.nickel_ebsd_small()
        >>> s.metadata.Sample.Phases.Number_1.atom_coordinates.Number_1
        ├── atom = Ni
        ├── coordinates = array([0, 0, 0])
        ├── debye_waller_factor = 0.0035
        └── site_occupation = 1
        >>> s.set_phase_parameters(
        ...     number=1,
        ...     atom_coordinates={'1': {
        ...         'atom': 'Fe',
        ...         'coordinates': [0, 0, 0],
        ...         'site_occupation': 1,
        ...         'debye_waller_factor': 0.005
        ...     }}
        ... )  # doctest: +SKIP
        >>> s.metadata.Sample.Phases.Number_1.atom_coordinates.Number_1  # doctest: +SKIP
        ├── atom = Fe
        ├── coordinates = array([0, 0, 0])
        ├── debye_waller_factor = 0.005
        └── site_occupation = 1
        """
        # Ensure atom coordinates are numpy arrays
        if atom_coordinates is not None:
            for phase, _ in atom_coordinates.items():
                atom_coordinates[phase]["coordinates"] = np.array(
                    atom_coordinates[phase]["coordinates"]
                )

        inputs = {
            "atom_coordinates": atom_coordinates,
            "formula": formula,
            "info": info,
            "lattice_constants": lattice_constants,
            "laue_group": laue_group,
            "material_name": material_name,
            "point_group": point_group,
            "setting": setting,
            "source": source,
            "space_group": space_group,
            "symmetry": symmetry,
        }

        # Remove None values
        phase = {k: v for k, v in inputs.items() if v is not None}
        _update_phase_info(self.metadata, phase, number)

    def set_scan_calibration(
        self, step_x: Union[int, float] = 1.0, step_y: Union[int, float] = 1.0
    ):
        """Set the step size in microns.

        Parameters
        ----------
        step_x
            Scan step size in um per pixel in horizontal direction.
        step_y
            Scan step size in um per pixel in vertical direction.

        See Also
        --------
        ~kikuchipy.signals.EBSD.set_detector_calibration

        Examples
        --------
        >>> import kikuchipy as kp
        >>> s = kp.data.nickel_ebsd_small()
        >>> s.axes_manager['x'].scale
        1.5
        >>> s.set_scan_calibration(step_x=2)  # Microns
        >>> s.axes_manager['x'].scale
        2.0
        """
        x, y = self.axes_manager.navigation_axes
        x.name, y.name = ("x", "y")
        x.scale, y.scale = (step_x, step_y)
        x.units, y.units = ["um"] * 2

    def set_detector_calibration(self, delta: Union[int, float]):
        """Set detector pixel size in microns. The offset is set to the
        the detector centre.

        Parameters
        ----------
        delta
            Detector pixel size in microns.

        See Also
        --------
        ~kikuchipy.signals.EBSD.set_scan_calibration

        Examples
        --------
        >>> import kikuchipy as kp
        >>> s = kp.data.nickel_ebsd_small()
        >>> s.axes_manager['dx'].scale  # Default value
        1.0
        >>> s.set_detector_calibration(delta=70.)
        >>> s.axes_manager['dx'].scale
        70.0
        """
        centre = delta * np.array(self.axes_manager.signal_shape) / 2
        dx, dy = self.axes_manager.signal_axes
        dx.units, dy.units = ["um"] * 2
        dx.scale, dy.scale = (delta, delta)
        dx.offset, dy.offset = -centre

    def remove_static_background(
        self,
        operation: str = "subtract",
        relative: bool = True,
        static_bg: Union[None, np.ndarray, da.Array] = None,
        scale_bg: bool = False,
    ):
        """Remove the static background in an EBSD scan inplace.

        The removal is performed by subtracting or dividing by a static
        background pattern. Resulting pattern intensities are rescaled
        keeping relative intensities or not and stretched to fill the
        available grey levels in the patterns' data type range.

        Parameters
        ----------
        operation
            Whether to "subtract" (default) or "divide" by the static
            background pattern.
        relative
            Keep relative intensities between patterns. Default is
            True.
        static_bg
            Static background pattern. If None is passed (default) we
            try to read it from the signal metadata.
        scale_bg
            Whether to scale the static background pattern to each
            individual pattern's data range before removal. Must be
            False if `relative` is True. Default is False.

        See Also
        --------
        ~kikuchipy.signals.EBSD.remove_dynamic_background

        Examples
        --------
        We assume that a static background pattern with the same shape
        and data type (e.g. 8-bit unsigned integer, ``uint8``) as the
        patterns is available in signal metadata:

        >>> import kikuchipy as kp
        >>> node = kp.signals.util.metadata_nodes("ebsd")
        >>> s = kp.data.nickel_ebsd_small()
        >>> s.metadata.get_item(node + '.static_background')
        array([[84, 87, 90, ..., 27, 29, 30],
               [87, 90, 93, ..., 27, 28, 30],
               [92, 94, 97, ..., 39, 28, 29],
               ...,
               [80, 82, 84, ..., 36, 30, 26],
               [79, 80, 82, ..., 28, 26, 26],
               [76, 78, 80, ..., 26, 26, 25]], dtype=uint8)

        The static background can be removed by subtracting or dividing
        this background from each pattern while keeping relative
        intensities between patterns (or not):

        >>> s.remove_static_background(
        ...     operation='subtract', relative=True
        ... )  # doctest: +SKIP

        If the metadata has no background pattern, this must be passed
        in the `static_bg` parameter as a numpy or dask array.
        """
        dtype_out = self.data.dtype.type

        # Get background pattern
        if static_bg is None:
            try:
                md = self.metadata
                ebsd_node = metadata_nodes("ebsd")
                static_bg_in_metadata = md.get_item(ebsd_node + ".static_background")
                if isinstance(static_bg_in_metadata, (da.Array, np.ndarray)):
                    static_bg = da.asarray(static_bg_in_metadata, chunks="auto")
                else:
                    raise ValueError
            except (AttributeError, ValueError):
                raise ValueError(
                    "`static_bg` is not a valid NumPy or Dask array or could not be "
                    "read from signal metadata"
                )
        if dtype_out != static_bg.dtype:
            raise ValueError(
                f"Static background dtype_out {static_bg.dtype} is not the same as "
                f"pattern dtype_out {dtype_out}"
            )
        pat_shape = self.axes_manager.signal_shape[::-1]
        bg_shape = static_bg.shape
        if bg_shape != pat_shape:
            raise ValueError(
                f"Signal {pat_shape} and static background {bg_shape} shapes are not "
                "identical"
            )
        dtype = np.float32
        static_bg = static_bg.astype(dtype)

        if operation == "subtract":
            operation_func = np.subtract
        else:  # operation == "divide"
            operation_func = np.divide

        # Get min./max. input patterns intensity after correction
        if relative is True and scale_bg is True:
            raise ValueError("'scale_bg' must be False if 'relative' is True.")
        elif relative is True:  # Scale relative to min./max. intensity in scan
            signal_min = self.data.min(axis=(0, 1))
            signal_max = self.data.max(axis=(0, 1))
            in_range = (
                operation_func(signal_min, static_bg).astype(dtype).min(),
                operation_func(signal_max, static_bg).astype(dtype).max(),
            )
        else:  # Scale relative to min./max. intensity in each pattern
            in_range = None

        # Create a dask array of signal patterns and do the processing on this
        dask_array = get_dask_array(signal=self, dtype=dtype)

        # Remove the static background and rescale intensities chunk by chunk
        corrected_patterns = dask_array.map_blocks(
            func=chunk.remove_static_background,
            static_bg=static_bg,
            operation_func=operation_func,
            scale_bg=scale_bg,
            in_range=in_range,
            dtype_out=dtype_out,
            dtype=dtype_out,
        )

        # Overwrite signal patterns
        if not self._lazy:
            with ProgressBar():
                print("Removing the static background:", file=sys.stdout)
                corrected_patterns.store(self.data, compute=True)
        else:
            self.data = corrected_patterns

    def remove_dynamic_background(
        self,
        operation: str = "subtract",
        filter_domain: str = "frequency",
        std: Union[None, int, float] = None,
        truncate: Union[int, float] = 4.0,
        **kwargs,
    ):
        """Remove the dynamic background in an EBSD scan inplace.

        The removal is performed by subtracting or dividing by a
        Gaussian blurred version of each pattern. Resulting pattern
        intensities are rescaled to fill the input patterns' data type
        range individually.

        Parameters
        ----------
        operation
            Whether to "subtract" (default) or "divide" by the dynamic
            background pattern.
        filter_domain
            Whether to obtain the dynamic background by applying a
            Gaussian convolution filter in the "frequency" (default) or
            "spatial" domain.
        std
            Standard deviation of the Gaussian window. If None
            (default), it is set to width/8.
        truncate
            Truncate the Gaussian window at this many standard
            deviations. Default is 4.0.
        kwargs :
            Keyword arguments passed to the Gaussian blurring function
            determined from `filter_domain`.

        See Also
        --------
        kikuchipy.signals.EBSD.remove_static_background
        kikuchipy.signals.EBSD.get_dynamic_background
        kikuchipy.pattern.remove_dynamic_background
        kikuchipy.pattern.get_dynamic_background

        Examples
        --------
        Traditional background correction includes static and dynamic
        corrections, loosing relative intensities between patterns after
        dynamic corrections (whether `relative` is set to True or
        False in :meth:`~remove_static_background`):

        >>> import kikuchipy as kp
        >>> s = kp.data.nickel_ebsd_small()
        >>> s.remove_static_background(operation="subtract")  # doctest: +SKIP
        >>> s.remove_dynamic_background(
        ...     operation="subtract",  # Default
        ...     filter_domain="frequency",  # Default
        ...     truncate=4.0,  # Default
        ...     std=5,
        ... )  # doctest: +SKIP
        """
        # Create a dask array of signal patterns and do the processing on this
        dtype = np.float32
        dask_array = get_dask_array(signal=self, dtype=dtype)

        if std is None:
            std = self.axes_manager.signal_shape[0] / 8

        # Get filter function and set up necessary keyword arguments
        if filter_domain == "frequency":
            # FFT filter setup for Connelly Barnes' algorithm
            filter_func = _fft_filter
            (
                kwargs["fft_shape"],
                kwargs["window_shape"],
                kwargs["transfer_function"],
                kwargs["offset_before_fft"],
                kwargs["offset_after_ifft"],
            ) = _dynamic_background_frequency_space_setup(
                pattern_shape=self.axes_manager.signal_shape[::-1],
                std=std,
                truncate=truncate,
            )
        elif filter_domain == "spatial":
            filter_func = gaussian_filter
            kwargs["sigma"] = std
            kwargs["truncate"] = truncate
        else:
            filter_domains = ["frequency", "spatial"]
            raise ValueError(f"{filter_domain} must be either of {filter_domains}.")

        if operation == "subtract":
            operation_func = np.subtract
        else:  # operation == "divide"
            operation_func = np.divide

        # Get output data type and output data type intensity range
        dtype_out = self.data.dtype.type
        out_range = dtype_range[dtype_out]

        corrected_patterns = dask_array.map_blocks(
            func=chunk.remove_dynamic_background,
            filter_func=filter_func,
            operation_func=operation_func,
            dtype_out=dtype_out,
            out_range=out_range,
            dtype=dtype_out,
            **kwargs,
        )

        # Overwrite signal patterns
        if not self._lazy:
            with ProgressBar():
                print("Removing the dynamic background:", file=sys.stdout)
                corrected_patterns.store(self.data, compute=True)
        else:
            self.data = corrected_patterns

    def get_dynamic_background(
        self,
        filter_domain: str = "frequency",
        std: Union[None, int, float] = None,
        truncate: Union[int, float] = 4.0,
        dtype_out: Optional[np.dtype] = None,
        **kwargs,
    ):
        """Get the dynamic background per EBSD pattern in a scan.

        Parameters
        ----------
        filter_domain
            Whether to apply a Gaussian convolution filter in the
            "frequency" (default) or "spatial" domain.
        std
            Standard deviation of the Gaussian window. If None
            (default), it is set to width/8.
        truncate
            Truncate the Gaussian filter at this many standard
            deviations. Default is 4.0.
        dtype_out
            Data type of the background patterns. If None (default), it
            is set to the same data type as the input pattern.
        kwargs :
            Keyword arguments passed to the Gaussian blurring function
            determined from `filter_domain`.

        Returns
        -------
        background_signal : EBSD or LazyEBSD
            Signal with the large scale variations across the detector.
        """
        if std is None:
            std = self.axes_manager.signal_shape[-1] / 8

        # Get filter function and set up necessary keyword arguments
        if filter_domain == "frequency":
            filter_func = _fft_filter
            # FFT filter setup for Connelly Barnes' algorithm
            (
                kwargs["fft_shape"],
                kwargs["window_shape"],
                kwargs["transfer_function"],
                kwargs["offset_before_fft"],
                kwargs["offset_after_ifft"],
            ) = _dynamic_background_frequency_space_setup(
                pattern_shape=self.axes_manager.signal_shape[::-1],
                std=std,
                truncate=truncate,
            )
        elif filter_domain == "spatial":
            filter_func = gaussian_filter
            kwargs["sigma"] = std
            kwargs["truncate"] = truncate
        else:
            filter_domains = ["frequency", "spatial"]
            raise ValueError(f"{filter_domain} must be either of {filter_domains}.")

        if dtype_out is None:
            dtype_out = self.data.dtype.type
        dask_array = get_dask_array(self, dtype=dtype_out)

        background_patterns = dask_array.map_blocks(
            func=chunk.get_dynamic_background,
            filter_func=filter_func,
            dtype_out=dtype_out,
            dtype=dtype_out,
            **kwargs,
        )

        if not self._lazy:
            background_return = np.empty(
                shape=background_patterns.shape, dtype=dtype_out
            )
            with ProgressBar():
                print("Getting the dynamic background:", file=sys.stdout)
                background_patterns.store(background_return, compute=True)
                background_signal = EBSD(background_return)
        else:
            background_signal = LazyEBSD(background_patterns)

        return background_signal

    def adaptive_histogram_equalization(
        self,
        kernel_size: Optional[Union[Tuple[int, int], List[int]]] = None,
        clip_limit: Union[int, float] = 0,
        nbins: int = 128,
    ):
        """Enhance the local contrast in an EBSD scan inplace using
        adaptive histogram equalization.

        This method uses :func:`skimage.exposure.equalize_adapthist`.

        Parameters
        ----------
        kernel_size
            Shape of contextual regions for adaptive histogram
            equalization, default is 1/4 of image height and 1/4 of
            image width.
        clip_limit
            Clipping limit, normalized between 0 and 1 (higher values
            give more contrast). Default is 0.
        nbins
            Number of gray bins for histogram ("data range"), default is
            128.

        See also
        --------
        ~kikuchipy.signals.EBSD.rescale_intensity
        ~kikuchipy.signals.EBSD.normalize_intensity

        Examples
        --------
        To best understand how adaptive histogram equalization works,
        we plot the histogram of the same image before and after
        equalization:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> import kikuchipy as kp
        >>> s = kp.data.nickel_ebsd_small()
        >>> s2 = s.inav[0, 0].deepcopy()
        >>> s2.adaptive_histogram_equalization()  # doctest: +SKIP
        >>> hist, _ = np.histogram(
        ...     s.inav[0, 0].data, bins=255, range=(0, 255)
        ... )
        >>> hist2, _ = np.histogram(s2.data, bins=255, range=(0, 255))
        >>> fig, ax = plt.subplots(nrows=2, ncols=2)
        >>> _ = ax[0, 0].imshow(s.inav[0, 0].data)
        >>> _ = ax[1, 0].plot(hist)
        >>> _ = ax[0, 1].imshow(s2.data)
        >>> _ = ax[1, 1].plot(hist2)

        Notes
        -----
        * It is recommended to perform adaptive histogram equalization
          only *after* static and dynamic background corrections,
          otherwise some unwanted darkening towards the edges might
          occur.
        * The default window size might not fit all pattern sizes, so it
          may be necessary to search for the optimal window size.
        """
        # Determine window size (shape of contextual region)
        sig_shape = self.axes_manager.signal_shape
        if kernel_size is None:
            kernel_size = (sig_shape[0] // 4, sig_shape[1] // 4)
        elif isinstance(kernel_size, numbers.Number):
            kernel_size = (kernel_size,) * self.axes_manager.signal_dimension
        elif len(kernel_size) != self.axes_manager.signal_dimension:
            raise ValueError(f"Incorrect value of `shape`: {kernel_size}")
        kernel_size = [int(k) for k in kernel_size]

        # Create dask array of signal patterns and do processing on this
        dask_array = get_dask_array(signal=self)

        # Local contrast enhancement
        equalized_patterns = dask_array.map_blocks(
            func=chunk.adaptive_histogram_equalization,
            kernel_size=kernel_size,
            clip_limit=clip_limit,
            nbins=nbins,
            dtype=self.data.dtype,
        )

        # Overwrite signal patterns
        if not self._lazy:
            with ProgressBar():
                print("Adaptive histogram equalization:", file=sys.stdout)
                equalized_patterns.store(self.data, compute=True)
        else:
            self.data = equalized_patterns

    def get_image_quality(self, normalize: bool = True) -> np.ndarray:
        """Compute the image quality map of patterns in an EBSD scan.

        The image quality is calculated based on the procedure defined
        by Krieger Lassen [Lassen1994]_.

        Parameters
        ----------
        normalize
            Whether to normalize patterns to a mean of zero and standard
            deviation of 1 before calculating the image quality. Default
            is True.

        Returns
        -------
        image_quality_map : numpy.ndarray
            Image quality map of same shape as signal navigation axes.

        References
        ----------
        .. [Lassen1994] N. C. K. Lassen, "Automated Determination of \
            Crystal Orientations from Electron Backscattering \
            Patterns," Institute of Mathematical Modelling, (1994).

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> import kikuchipy as kp
        >>> s = kp.data.nickel_ebsd_small()
        >>> iq = s.get_image_quality(normalize=True)  # doctest: +SKIP
        >>> plt.imshow(iq)  # doctest: +SKIP

        See Also
        --------
        kikuchipy.pattern.get_image_quality
        """
        # Data set to operate on
        dtype_out = np.float32
        dask_array = get_dask_array(self, dtype=dtype_out)

        # Calculate frequency vectors
        sx, sy = self.axes_manager.signal_shape
        frequency_vectors = fft_frequency_vectors((sy, sx))
        inertia_max = np.sum(frequency_vectors) / (sy * sx)

        # Calculate image quality per chunk
        image_quality_map = dask_array.map_blocks(
            func=chunk.get_image_quality,
            frequency_vectors=frequency_vectors,
            inertia_max=inertia_max,
            normalize=normalize,
            dtype=dtype_out,
            drop_axis=self.axes_manager.signal_indices_in_array,
        )

        if not self._lazy:
            with ProgressBar():
                print("Calculating the image quality:", file=sys.stdout)
                image_quality_map = image_quality_map.compute()

        return image_quality_map

    def dictionary_indexing(
        self,
        dictionary,
        metric: Union[SimilarityMetric, str] = "ncc",
        keep_n: int = 20,
        n_per_iteration: Optional[int] = None,
        signal_mask: Optional[np.ndarray] = None,
        rechunk: bool = False,
        dtype: Union[np.dtype, type, None] = None,
    ) -> CrystalMap:
        """Match each experimental pattern to a dictionary of simulated
        patterns of known orientations to index the them
        :cite:`chen2015dictionary,jackson2019dictionary`.

        A suitable similarity metric, the normalized cross-correlation
        (:class:`~kikuchipy.indexing.similarity_metrics.NormalizedCrossCorrelationMetric`),
        is used by default, but a valid user-defined similarity metric
        may be used instead. The metric must be a class implementing the
        :class:`~kikuchipy.indexing.similarity_metrics.SimilarityMetric`
        abstract class methods. The normalized dot product
        (:class:`~kikuchipy.indexing.similarity_metrics.NormalizedDotProductMetric`)
        is available as well.

        A :class:`~orix.crystal_map.CrystalMap` with "scores" and
        "simulation_indices" as properties is returned.

        Parameters
        ----------
        dictionary : EBSD
            EBSD signal with dictionary patterns. The signal must have a
            1D navigation axis, an *xmap* property with crystal
            orientations set, and equal detector shape.
        metric
            Similarity metric, by default "ncc" (normalized
            cross-correlation). "ndp" (normalized dot product) is also
            available.
        keep_n
            Number of best matches to keep, by default 20 or the number
            of dictionary patterns if fewer than 20 are available.
        n_per_iteration
            Number of dictionary patterns to compare to all experimental
            patterns in each indexing iteration. If not given, and the
            dictionary is a LazyEBSD signal, it is equal to the chunk
            size of the first pattern array axis, while if if is an EBSD
            signal, it is set equal to the number of dictionary
            patterns, yielding only one iteration. This parameter can be
            increased to use less memory during indexing, but this will
            increase the computation time.
        signal_mask
            A boolean mask equal to the experimental patterns' detector
            shape (n rows, n columns), where only pixels equal to False
            are matched. If not given, all pixels are used.
        rechunk
            Whether *metric* is allowed to rechunk experimental and
            dictionary patterns before matching. Default is False. If a
            custom *metric* is passed, whatever *metric.rechunk* is set
            to will be used. Rechunking usually makes indexing faster,
            but uses more memory.
        dtype
            Which data type *metric* shall cast the patterns to before
            matching. If not given, :class:`~numpy.float32` will be
            used unless a custom *metric* is passed and it has set the
            *dtype* attribute, which will then be used instead.
            :class:`~numpy.float32` and :class:`~numpy.float64` is
            allowed for the available "ncc" and "ndp" metrics.

        Returns
        -------
        xmap : ~orix.crystal_map.CrystalMap
            A crystal map with *keep_n* rotations per point with the
            sorted best matching orientations in the dictionary. The
            corresponding best scores and indices into the dictionary
            are stored in the *xmap.prop* dictionary as "scores" and
            "simulation_indices".

        Notes
        -----
        Merging of single phase crystal maps into one multi phase map
        and calculations of an orientation similarity map can be done
        afterwards with
        :func:`~kikuchipy.indexing.merge_crystal_maps` and
        :func:`~kikuchipy.indexing.orientation_similarity_map`,
        respectively.

        .. versionchanged:: 0.5
           Only one dictionary can be passed, the *n_per_iteration*
           parameter replaced *n_slices*, and the
           *return_merged_crystal_map* and
           *get_orientation_similarity_map* parameters were removed.

        .. versionadded:: 0.5
           The *signal_mask*, *rechunk*, and *dtype* parameters.

        See Also
        --------
        ~kikuchipy.indexing.similarity_metrics.SimilarityMetric
        ~kikuchipy.indexing.similarity_metrics.NormalizedCrossCorrelationMetric
        ~kikuchipy.indexing.similarity_metrics.NormalizedDotProductMetric
        """
        exp_am = self.axes_manager
        dict_am = dictionary.axes_manager
        dict_size = dict_am.navigation_size

        if n_per_iteration is None:
            if isinstance(dictionary.data, da.Array):
                n_per_iteration = dictionary.data.chunksize[0]
            else:
                n_per_iteration = dict_size

        exp_sig_shape = exp_am.signal_shape[::-1]
        dict_sig_shape = dict_am.signal_shape[::-1]
        if exp_sig_shape != dict_sig_shape:
            raise ValueError(
                f"Experimental {exp_sig_shape} and dictionary {dict_sig_shape} signal "
                "shapes must be identical"
            )

        dict_xmap = dictionary.xmap
        if dict_xmap is None or dict_xmap.shape != (dict_size,):
            raise ValueError(
                "Dictionary signal must have a non-empty `EBSD.xmap` property of equal "
                "size as the number of dictionary patterns, and both the signal and"
                "crystal map must have only one navigation dimension"
            )

        metric = self._prepare_metric(metric, signal_mask, dtype, rechunk, dict_size)

        return _dictionary_indexing(
            experimental=self.data,
            experimental_nav_shape=exp_am.navigation_shape[::-1],
            dictionary=dictionary.data,
            step_sizes=tuple(a.scale for a in exp_am.navigation_axes[::-1]),
            dictionary_xmap=dictionary.xmap,
            keep_n=keep_n,
            n_per_iteration=n_per_iteration,
            metric=metric,
        )

    def refine_orientation(
        self,
        xmap: CrystalMap,
        detector: EBSDDetector,
        master_pattern,
        energy: Union[int, float],
        mask: Optional[np.ndarray] = None,
        method: Optional[str] = "minimize",
        method_kwargs: Optional[dict] = None,
        trust_region: Optional[list] = None,
        compute: bool = True,
        rechunk: bool = True,
        chunk_kwargs: Optional[dict] = None,
    ):
        r"""Refine orientations by searching orientation space around
        the best indexed solution using fixed projection centers.

        Refinement attempts to optimize (maximize) the similarity
        between patterns in this signal and simulated patterns
        projected from a master pattern. The only supported similarity
        metric is the normalized cross-correlation (NCC). The
        orientation, represented by three Euler angles
        (:math:`\phi_1`, :math:`\Phi`, :math:`\phi_2`), is changed
        during projection, while the sample-detector geometry,
        represented by the three projection center (PC) parameters
        (PCx, PCy, PCz), are fixed.

        A subset of the optimization methods in SciPy are available:
            - Local optimization:
                - :func:`~scipy.optimize.minimize`
                  (includes Nelder-Mead, Powell etc.)
            - Global optimization:
                - :func:`~scipy.optimize.differential_evolution`
                - :func:`~scipy.optimize.dual_annealing`
                - :func:`~scipy.optimize.basinhopping`
                - :func:`~scipy.optimize.shgo`

        Parameters
        ----------
        xmap
            Single phase crystal map with at least one orientation per
            point. The orientations are assumed to be relative to the
            EDAX TSL sample reference frame RD-TD-ND.
        detector
            Detector describing the detector-sample geometry with either
            one PC to be used for all map points or one for each point.
        master_pattern : EBSDMasterPattern
            Master pattern in the square Lambert projection of the same
            phase as the one in the crystal map.
        energy
            Accelerating voltage of the electron beam in kV specifying
            which master pattern energy to use during projection of
            simulated patterns.
        mask
            Boolean mask of signal shape to be applied to the simulated
            pattern before comparison. Pixels set to `True` are masked
            away. If not given, all pixels are matched.
        method : str, optional
            Name of the :mod:`scipy.optimize` optimization method, among
            "minimize", "differential_evolution", "dual_annealing",
            "basinhopping", and "shgo". Default is "minimize", which
            by default performs local optimization with the Nelder-Mead
            method unless another "minimize" method is passed to
            `method_kwargs`.
        method_kwargs
            Keyword arguments passed to the :mod:`scipy.optimize`
            `method`. For example, to perform refinement with the
            modified Powell algorithm, pass `method="minimize"` and
            `method_kwargs=dict(method="Powell")`.
        trust_region
            List of +/- angular deviation in degrees as bound
            constraints on the three Euler angles. If not given and
            `method` requires bounds, they are set to [1, 1, 1]. If
            given, `method` is assumed to support bounds and they are
            passed to `method`.
        compute
            Whether to refine now (True) or later (False). Default is
            True. See :meth:`~dask.array.Array.compute` for more
            details.
        rechunk
            If True (default), rechunk the dask array with patterns used
            in refinement (not the signal data inplace) if it is
            returned from :func:`~kikuchipy.signals.util.get_dask_array`
            in a single chunk. This ensures small data sets are
            rechunked so as to utilize multiple CPUs.
        chunk_kwargs
            Keyword arguments passed to
            :func:`~kikuchipy.signals.util.get_chunking` if `rechunk` is
            True and the dask array with patterns used in refinement is
            returned from :func:`~kikuchipy.signals.util.get_dask_array`
            in a single chunk.

        Returns
        -------
        :class:`~orix.crystal_map.CrystalMap` or :class:`~dask.array.Array`
            Crystal map with refined orientations and similarity metrics
            in a "scores" property if `compute` is True. If
            `compute` is False, a dask array of navigation shape + (4,)
            is returned, to be computed later. See
            :func:`~kikuchipy.indexing.compute_refine_orientation_results`.
            Each navigation point has the optimized score and the three
            Euler angles in radians in element 0, 1, 2, and 3,
            respectively.

        See Also
        --------
        scipy.optimize
        refine_projection_center
        refine_orientation_projection_center
        """
        self._check_refinement_parameters(xmap=xmap, detector=detector, mask=mask)
        patterns = self._get_dask_array_for_refinement(
            rechunk=rechunk, chunk_kwargs=chunk_kwargs
        )
        return _refine_orientation(
            xmap=xmap,
            detector=detector,
            master_pattern=master_pattern,
            energy=energy,
            patterns=patterns,
            signal_indices_in_array=self.axes_manager.signal_indices_in_array,
            mask=mask,
            method=method,
            method_kwargs=method_kwargs,
            trust_region=trust_region,
            compute=compute,
        )

    def refine_projection_center(
        self,
        xmap: CrystalMap,
        detector: EBSDDetector,
        master_pattern,
        energy: Union[int, float],
        mask: Optional[np.ndarray] = None,
        method: Optional[str] = "minimize",
        method_kwargs: Optional[dict] = None,
        trust_region: Optional[list] = None,
        compute: bool = True,
        rechunk: bool = True,
        chunk_kwargs: Optional[dict] = None,
    ):
        """Refine projection centers by searching the parameter space
        using fixed orientations.

        Refinement attempts to optimize (maximize) the similarity
        between patterns in this signal and simulated patterns projected
        from a master pattern. The only supported similarity metric is
        the normalized cross-correlation (NCC). The sample-detector
        geometry, represented by the three projection center (PC)
        parameters (PCx, PCy, PCz), is changed during projection, while
        the orientations are fixed.

        A subset of the optimization methods in SciPy are available:
            - Local optimization:
                - :func:`~scipy.optimize.minimize`
                  (includes Nelder-Mead, Powell etc.)
            - Global optimization:
                - :func:`~scipy.optimize.differential_evolution`
                - :func:`~scipy.optimize.dual_annealing`
                - :func:`~scipy.optimize.basinhopping`
                - :func:`~scipy.optimize.shgo`

        Parameters
        ----------
        xmap
            Single phase crystal map with at least one orientation per
            point. The orientations are assumed to be relative to the
            EDAX TSL sample reference frame RD-TD-ND.
        detector
            Detector describing the detector-sample geometry with either
            one PC to be used for all map points or one for each point.
        master_pattern : EBSDMasterPattern
            Master pattern in the square Lambert projection of the same
            phase as the one in the crystal map.
        energy
            Accelerating voltage of the electron beam in kV specifying
            which master pattern energy to use during projection of
            simulated patterns.
        mask
            Boolean mask of signal shape to be applied to the simulated
            pattern before comparison. Pixels set to `True` are masked
            away. If not given, all pixels are matched.
        method : str, optional
            Name of the :mod:`scipy.optimize` optimization method, among
            "minimize", "differential_evolution", "dual_annealing",
            "basinhopping", and "shgo". Default is "minimize", which
            by default performs local optimization with the Nelder-Mead
            method unless another "minimize" method is passed to
            `method_kwargs`.
        method_kwargs
            Keyword arguments passed to the :mod:`scipy.optimize`
            `method`. For example, to perform refinement with the
            modified Powell algorithm, pass `method="minimize"` and
            `method_kwargs=dict(method="Powell")`.
        trust_region
            List of +/- percentage deviations as bound constraints on
            the PC parameters in the Bruker convention. The parameter
            range is [0, 1]. If not given and `method` requires bounds,
            they are set to [0.05, 0.05, 0.05]. If given, `method` is
            assumed to support bounds and they are passed to `method`.
        compute
            Whether to refine now (True) or later (False). Default is
            True. See :meth:`~dask.array.Array.compute` for more
            details.
        rechunk
            If True (default), rechunk the dask array with patterns used
            in refinement (not the signal data inplace) if it is
            returned from :func:`~kikuchipy.signals.util.get_dask_array`
            in a single chunk. This ensures small data sets are
            rechunked so as to utilize multiple CPUs.
        chunk_kwargs
            Keyword arguments passed to
            :func:`~kikuchipy.signals.util.get_chunking` if `rechunk` is
            True and the dask array with patterns used in refinement is
            returned from :func:`~kikuchipy.signals.util.get_dask_array`
            in a single chunk.

        Returns
        -------
        :class:`numpy.ndarray` and :class:`~kikuchipy.detectors.EBSDDetector`,\
        or :class:`~dask.array.Array`
            New similarity metrics and a new EBSD detector instance with
            the refined PCs if `compute` is True. If `compute` is False,
            a dask array of navigation shape + (4,) is returned, to be
            computed later. See
            :func:`~kikuchipy.indexing.compute_refine_projection_center_results`.
            Each navigation point has the optimized score and the three
            PC parameters in the Bruker convention in element 0, 1, 2,
            and 3, respectively.

        See Also
        --------
        scipy.optimize
        refine_orientation
        refine_orientation_projection_center
        """
        self._check_refinement_parameters(xmap=xmap, detector=detector, mask=mask)
        patterns = self._get_dask_array_for_refinement(
            rechunk=rechunk, chunk_kwargs=chunk_kwargs
        )
        return _refine_projection_center(
            xmap=xmap,
            detector=detector,
            master_pattern=master_pattern,
            energy=energy,
            patterns=patterns,
            signal_indices_in_array=self.axes_manager.signal_indices_in_array,
            mask=mask,
            method=method,
            method_kwargs=method_kwargs,
            trust_region=trust_region,
            compute=compute,
        )

    def refine_orientation_projection_center(
        self,
        xmap: CrystalMap,
        detector: EBSDDetector,
        master_pattern,
        energy: Union[int, float],
        mask: Optional[np.ndarray] = None,
        method: Optional[str] = "minimize",
        method_kwargs: Optional[dict] = None,
        trust_region: Optional[list] = None,
        compute: bool = True,
        rechunk: bool = True,
        chunk_kwargs: Optional[dict] = None,
    ):
        r"""Refine orientations and projection centers simultaneously by
        searching the orientation and PC parameter space.

        Refinement attempts to optimize (maximize) the similarity
        between patterns in this signal and simulated patterns
        projected from a master pattern. The only supported
        similarity metric is the normalized cross-correlation (NCC).
        The orientation, represented by three Euler angles
        (:math:`\phi_1`, :math:`\Phi`, :math:`\phi_2`), and the
        sample-detector geometry, represented by the three projection
        center (PC) parameters (PCx, PCy, PCz), are changed during
        projection.

        A subset of the optimization methods in SciPy are available:
            - Local optimization:
                - :func:`~scipy.optimize.minimize`
                  (includes Nelder-Mead, Powell etc.)
            - Global optimization:
                - :func:`~scipy.optimize.differential_evolution`
                - :func:`~scipy.optimize.dual_annealing`
                - :func:`~scipy.optimize.basinhopping`
                - :func:`~scipy.optimize.shgo`

        Parameters
        ----------
        xmap
            Single phase crystal map with at least one orientation per
            point. The orientations are assumed to be relative to the
            EDAX TSL sample reference frame RD-TD-ND.
        detector
            Detector describing the detector-sample geometry with either
            one PC to be used for all map points or one for each point.
        master_pattern : EBSDMasterPattern
            Master pattern in the square Lambert projection of the same
            phase as the one in the crystal map.
        energy
            Accelerating voltage of the electron beam in kV specifying
            which master pattern energy to use during projection of
            simulated patterns.
        mask
            Boolean mask of signal shape to be applied to the simulated
            pattern before comparison. Pixels set to `True` are masked
            away. If not given, all pixels are matched.
        method : str, optional
            Name of the :mod:`scipy.optimize` optimization method, among
            "minimize", "differential_evolution", "dual_annealing",
            "basinhopping", and "shgo". Default is "minimize", which
            by default performs local optimization with the Nelder-Mead
            method unless another "minimize" method is passed to
            `method_kwargs`.
        method_kwargs
            Keyword arguments passed to the :mod:`scipy.optimize`
            `method`. For example, to perform refinement with the
            modified Powell algorithm, pass `method="minimize"` and
            `method_kwargs=dict(method="Powell")`.
        trust_region
            List of +/- angular deviations in degrees as bound
            constraints on the three Euler angles and +/- percentage
            deviations as bound constraints on the PC parameters in the
            Bruker convention. The latter parameter range is [0, 1]. If
            not given and `method` requires bounds, they are set to
            [1, 1, 1, 0.05, 0.05, 0.05]. If given, `method` is assumed
            to support bounds and they are passed to `method`.
        compute
            Whether to refine now (True) or later (False). Default is
            True. See :meth:`~dask.array.Array.compute` for more
            details.
        rechunk
            If True (default), rechunk the dask array with patterns used
            in refinement (not the signal data inplace) if it is
            returned from :func:`~kikuchipy.signals.util.get_dask_array`
            in a single chunk. This ensures small data sets are
            rechunked so as to utilize multiple CPUs.
        chunk_kwargs
            Keyword arguments passed to
            :func:`~kikuchipy.signals.util.get_chunking` if `rechunk` is
            True and the dask array with patterns used in refinement is
            returned from :func:`~kikuchipy.signals.util.get_dask_array`
            in a single chunk.

        Returns
        -------
        :class:`~orix.crystal_map.CrystalMap` and :class:`~kikuchipy.detectors.EBSDDetector`, or :class:`~dask.array.Array`
            Crystal map with refined orientations and a new EBSD
            detector instance with the refined PCs, if `compute` is
            True. If `compute` is False, a dask array of navigation
            shape + (7,) is returned, to be computed later. See
            :func:`~kikuchipy.indexing.compute_refine_orientation_projection_center_results`.
            Each navigation point has the optimized score, the three
            Euler angles in radians, and the three PC parameters in the
            Bruker convention in element 0, 1, 2, 3, 4, 5, and 6,
            respectively.

        See Also
        --------
        scipy.optimize
        refine_orientation
        refine_projection_center

        Notes
        -----
        The method attempts to refine the orientations and projection
        center at the same time for each map point. The optimization
        landscape is sloppy :cite:`pang2020optimization`, where the
        orientation and PC can make up for each other. Thus, it is
        possible that the set of parameters that yield the highest
        similarity is incorrect. It is left to the user to ensure that
        the output is reasonable.
        """
        self._check_refinement_parameters(xmap=xmap, detector=detector, mask=mask)
        patterns = self._get_dask_array_for_refinement(
            rechunk=rechunk, chunk_kwargs=chunk_kwargs
        )
        return _refine_orientation_projection_center(
            xmap=xmap,
            detector=detector,
            master_pattern=master_pattern,
            energy=energy,
            patterns=patterns,
            signal_indices_in_array=self.axes_manager.signal_indices_in_array,
            mask=mask,
            method=method,
            method_kwargs=method_kwargs,
            trust_region=trust_region,
            compute=compute,
        )

    def fft_filter(
        self,
        transfer_function: Union[np.ndarray, Window],
        function_domain: str,
        shift: bool = False,
    ):
        """Filter an EBSD scan inplace in the frequency domain.

        Patterns are transformed via the Fast Fourier Transform (FFT) to
        the frequency domain, where their spectrum is multiplied by the
        `transfer_function`, and the filtered spectrum is subsequently
        transformed to the spatial domain via the inverse FFT (IFFT).
        Filtered patterns are rescaled to input data type range.

        Note that if `function_domain` is "spatial", only real valued
        FFT and IFFT is used.

        Parameters
        ----------
        transfer_function
            Filter to apply to patterns. This can either be a transfer
            function in the frequency domain of pattern shape or a
            kernel in the spatial domain. What is passed is determined
            from `function_domain`.
        function_domain
            Options are "frequency" and "spatial", indicating,
            respectively, whether the filter function passed to
            `filter_function` is a transfer function in the frequency
            domain or a kernel in the spatial domain.
        shift
            Whether to shift the zero-frequency component to the centre.
            Default is False. This is only used when
            `function_domain="frequency"`.

        Examples
        --------
        Applying a Gaussian low pass filter with a cutoff frequency of
        20 to an EBSD object ``s``:

        >>> import kikuchipy as kp
        >>> s = kp.data.nickel_ebsd_small()
        >>> pattern_shape = s.axes_manager.signal_shape[::-1]
        >>> w = kp.filters.Window(
        ...     "lowpass", cutoff=20, shape=pattern_shape
        ... )
        >>> s.fft_filter(
        ...     transfer_function=w,
        ...     function_domain="frequency",
        ...     shift=True,
        ... )  # doctest: +SKIP

        See Also
        --------
        ~kikuchipy.filters.window.Window
        """
        dtype_out = self.data.dtype

        dtype = np.float32
        dask_array = get_dask_array(signal=self, dtype=dtype)

        kwargs = {}
        if function_domain == "frequency":
            filter_func = fft_filter
            kwargs["shift"] = shift
        elif function_domain == "spatial":
            filter_func = _fft_filter  # Barnes
            kwargs["window_shape"] = transfer_function.shape

            # FFT filter setup
            (
                kwargs["fft_shape"],
                transfer_function,  # Padded
                kwargs["offset_before_fft"],
                kwargs["offset_after_ifft"],
            ) = _fft_filter_setup(
                image_shape=self.axes_manager.signal_shape[::-1],
                window=transfer_function,
            )
        else:
            function_domains = ["frequency", "spatial"]
            raise ValueError(f"{function_domain} must be either of {function_domains}.")

        filtered_patterns = dask_array.map_blocks(
            func=chunk.fft_filter,
            filter_func=filter_func,
            transfer_function=transfer_function,
            dtype_out=dtype_out,
            dtype=dtype_out,
            **kwargs,
        )

        # Overwrite signal patterns
        if not self._lazy:
            with ProgressBar():
                print("FFT filtering:", file=sys.stdout)
                filtered_patterns.store(self.data, compute=True)
        else:
            self.data = filtered_patterns

    def get_neighbour_dot_product_matrices(
        self,
        window: Optional[Window] = None,
        zero_mean: bool = True,
        normalize: bool = True,
        dtype_out: np.dtype = np.float32,
    ) -> Union[np.ndarray, da.Array]:
        """Get an array with dot products of a pattern and its
        neighbours within a window.

        Parameters
        ----------
        window
            Window with integer coefficients defining the neighbours to
            calculate the dot products with. If None (default), the four
            nearest neighbours are used. Must have the same number of
            dimensions as signal navigation dimensions.
        zero_mean
            Whether to subtract the mean of each pattern individually to
            center the intensities about zero before calculating the
            dot products. Default is True.
        normalize
            Whether to normalize the pattern intensities to a standard
            deviation of 1 before calculating the dot products. This
            operation is performed after centering the intensities if
            `zero_mean` is True. Default is True.
        dtype_out
            Data type of the output map. Default is
            :class:`numpy.float32`.

        Returns
        -------
        """
        if self.axes_manager.navigation_dimension == 0:
            raise ValueError("Signal must have at least one navigation dimension")

        # Create dask array of signal patterns and do processing on this
        dask_array = get_dask_array(signal=self)

        # Default to the nearest neighbours
        nav_dim = self.axes_manager.navigation_dimension
        if window is None:
            window = Window(window="circular", shape=(3, 3)[:nav_dim])

        # Set overlap depth between navigation chunks equal to the max.
        # number of nearest neighbours in each navigation axis
        overlap_depth = _get_chunk_overlap_depth(
            window=window,
            axes_manager=self.axes_manager,
            chunksize=dask_array.chunksize,
        )

        dp_matrices = dask_array.map_overlap(
            _get_neighbour_dot_product_matrices,
            window=window,
            sig_dim=self.axes_manager.signal_dimension,
            sig_size=self.axes_manager.signal_size,
            zero_mean=zero_mean,
            normalize=normalize,
            dtype_out=dtype_out,
            drop_axis=self.axes_manager.signal_indices_in_array[::-1],
            new_axis=tuple(np.arange(window.ndim) + nav_dim),
            dtype=dtype_out,
            depth=overlap_depth,
            boundary="none",
        )

        if not self._lazy:
            with ProgressBar():
                print("Calculating neighbour dot product matrices:", file=sys.stdout)
                dp_matrices = dp_matrices.compute()

        return dp_matrices

    def get_average_neighbour_dot_product_map(
        self,
        window: Optional[Window] = None,
        zero_mean: bool = True,
        normalize: bool = True,
        dtype_out: np.dtype = np.float32,
        dp_matrices: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, da.Array]:
        """Get a map of the average dot product between patterns and
        their neighbours within an averaging window.

        Parameters
        ----------
        window
            Window with integer coefficients defining the neighbours to
            calculate the average with. If None (default), the four
            nearest neighbours are used. Must have the same number of
            dimensions as signal navigation dimensions.
        zero_mean
            Whether to subtract the mean of each pattern individually to
            center the intensities about zero before calculating the
            dot products. Default is True.
        normalize
            Whether to normalize the pattern intensities to a standard
            deviation of 1 before calculating the dot products. This
            operation is performed after centering the intensities if
            `zero_mean` is True. Default is True.
        dtype_out
            Data type of the output map. Default is
            :class:`numpy.float32`.
        dp_matrices
            Optional pre-calculated dot product matrices, by default
            None. If an array is passed, the average dot product map
            is calculated from this array. The `dp_matrices` array can
            be obtained from :meth:`get_neighbour_dot_product_matrices`.
            It's shape must correspond to the signal's navigation shape
            and the window's shape.

        Returns
        -------
        """
        if self.axes_manager.navigation_dimension == 0:
            raise ValueError("Signal must have at least one navigation dimension")

        # Default to the nearest neighbours
        nav_dim = self.axes_manager.navigation_dimension
        if window is None:
            window = Window(window="circular", shape=(3, 3)[:nav_dim])

        if dp_matrices is not None:
            nan_slices = [slice(None) for _ in range(nav_dim)]
            nan_slices += [slice(i, i + 1) for i in window.origin]
            dp_matrices2 = dp_matrices.copy()
            dp_matrices2[tuple(nan_slices)] = np.nan
            if nav_dim == 1:
                mean_axis = 1
            else:  # == 2
                mean_axis = (2, 3)
            return np.nanmean(dp_matrices2, axis=mean_axis)

        # Create dask array of signal data and do processing on this
        dask_array = get_dask_array(signal=self)

        # Set overlap depth between navigation chunks equal to the max.
        # number of nearest neighbours in each navigation axis
        overlap_depth = _get_chunk_overlap_depth(
            window=window,
            axes_manager=self.axes_manager,
            chunksize=dask_array.chunksize,
        )

        adp = dask_array.map_overlap(
            _get_average_dot_product_map,
            window=window,
            sig_dim=self.axes_manager.signal_dimension,
            sig_size=self.axes_manager.signal_size,
            zero_mean=zero_mean,
            normalize=normalize,
            dtype_out=dtype_out,
            drop_axis=self.axes_manager.signal_indices_in_array,
            dtype=dtype_out,
            depth=overlap_depth,
            boundary="none",
        )
        chunks = get_chunking(
            data_shape=self.axes_manager.navigation_shape[::-1],
            nav_dim=nav_dim,
            sig_dim=0,
            dtype=dtype_out,
        )
        adp = adp.rechunk(chunks=chunks)

        if not self._lazy:
            with ProgressBar():
                print("Calculating average neighbour dot product map:", file=sys.stdout)
                adp = adp.compute()

        return adp

    def average_neighbour_patterns(
        self,
        window: Union[str, np.ndarray, da.Array, Window] = "circular",
        window_shape: Tuple[int, ...] = (3, 3),
        **kwargs,
    ):
        """Average patterns in an EBSD scan inplace with its neighbours
        within a window.

        The amount of averaging is specified by the window coefficients.
        All patterns are averaged with the same window. Map borders are
        extended with zeros. Resulting pattern intensities are rescaled
        to fill the input patterns' data type range individually.

        Averaging is accomplished by correlating the window with the
        extended array of patterns using
        :func:`scipy.ndimage.correlate`.

        Parameters
        ----------
        window
            Name of averaging window or an array. Available types are
            listed in :func:`scipy.signal.windows.get_window`, in
            addition to a "circular" window (default) filled with ones
            in which corner coefficients are set to zero. A window
            element is considered to be in a corner if its radial
            distance to the origin (window centre) is shorter or equal
            to the half width of the window's longest axis. A 1D or 2D
            :class:`numpy.ndarray`, :class:`dask.array.Array` or
            :class:`~kikuchipy.filters.Window` can also be passed.
        window_shape
            Shape of averaging window. Not used if a custom window or
            :class:`~kikuchipy.util.window.Window` object is passed to
            `window`. This can be either 1D or 2D, and can be
            asymmetrical. Default is (3, 3).
        **kwargs :
            Keyword arguments passed to the available window type listed
            in :func:`scipy.signal.windows.get_window`. If none are
            passed, the default values of that particular window are used.

        See Also
        --------
        ~kikuchipy.filters.window.Window
        :func:`scipy.signal.windows.get_window`
        :func:`scipy.ndimage.correlate`
        """
        if isinstance(window, Window) and window.is_valid():
            averaging_window = copy.copy(window)
        else:
            averaging_window = Window(window=window, shape=window_shape, **kwargs)

        # Do nothing if a window of shape (1, ) or (1, 1) is passed
        nav_shape = self.axes_manager.navigation_shape[::-1]
        window_shape = averaging_window.shape
        if window_shape in [(1,), (1, 1)]:
            return warnings.warn(
                f"A window of shape {window_shape} was passed, no averaging is "
                "therefore performed."
            )
        elif len(nav_shape) > len(window_shape):
            averaging_window = averaging_window.reshape(window_shape + (1,))

        # Get sum of window data for each pattern, to normalize with
        # after correlation
        window_sums = correlate(
            input=np.ones(nav_shape, dtype=int),
            weights=averaging_window,
            mode="constant",
            cval=0,
        )

        # Add signal dimensions to window array to enable its use with Dask's
        # map_blocks()
        sig_dim = self.axes_manager.signal_dimension
        averaging_window = averaging_window.reshape(
            averaging_window.shape + (1,) * sig_dim
        )

        # Create dask array of signal patterns and do processing on this
        dask_array = get_dask_array(signal=self)

        # Add signal dimensions to array be able to use with Dask's map_blocks()
        nav_dim = self.axes_manager.navigation_dimension
        for i in range(sig_dim):
            window_sums = np.expand_dims(window_sums, axis=window_sums.ndim)
        window_sums = da.from_array(
            window_sums, chunks=dask_array.chunks[:nav_dim] + (1,) * sig_dim
        )

        # Create overlap between chunks to enable correlation with the window
        # using Dask's map_blocks()
        window_dim = averaging_window.ndim
        overlap_depth = {}
        for i in range(nav_dim):
            if i < window_dim and dask_array.chunks[i][0] < dask_array.shape[i]:
                overlap_depth[i] = (window_shape[i] // 2) + 1
            else:
                overlap_depth[i] = 1
        overlap_depth.update(
            {i: 0 for i in self.axes_manager.signal_indices_in_array[::-1]}
        )

        dtype_out = self.data.dtype
        averaged_patterns = da.overlap.map_overlap(
            chunk.average_neighbour_patterns,
            dask_array,
            window_sums,
            window=averaging_window,
            dtype_out=dtype_out,
            dtype=dtype_out,
            depth=overlap_depth,
            boundary=np.nan,
        )

        # Overwrite signal patterns
        if not self._lazy:
            with ProgressBar():
                print("Averaging with the neighbour patterns:", file=sys.stdout)
                averaged_patterns.store(self.data, compute=True)
        else:
            self.data = averaged_patterns

    def plot_virtual_bse_intensity(
        self,
        roi: BaseInteractiveROI,
        out_signal_axes: Union[None, Iterable[int], Iterable[str]] = None,
        **kwargs,
    ):
        """Plot an interactive virtual backscatter electron (VBSE)
        image formed from intensities within a specified and adjustable
        region of interest (ROI) on the detector.

        Adapted from
        :meth:`pyxem.signals.common_diffraction.CommonDiffraction.plot_integrated_intensity`.

        Parameters
        ----------
        roi
            Any interactive ROI detailed in HyperSpy.
        out_signal_axes
            Which navigation axes to use as signal axes in the virtual
            image. If None (default), the first two navigation axes are
            used.
        **kwargs:
            Keyword arguments passed to the `plot` method of the virtual
            image.

        Examples
        --------
        >>> import hyperspy.api as hs
        >>> import kikuchipy as kp
        >>> s = kp.data.nickel_ebsd_small()
        >>> rect_roi = hs.roi.RectangularROI(
        ...     left=0, right=5, top=0, bottom=5
        ... )
        >>> s.plot_virtual_bse_intensity(rect_roi)

        See Also
        --------
        ~kikuchipy.signals.EBSD.get_virtual_bse_intensity
        """
        # Plot signal if necessary
        if self._plot is None or not self._plot.is_active:
            self.plot()

        # Get the sliced signal from the ROI
        sliced_signal = roi.interactive(self, axes=self.axes_manager.signal_axes)

        # Create an output signal for the virtual backscatter electron
        # calculation
        out = self._get_sum_signal(self, out_signal_axes)
        out.metadata.General.title = "Virtual backscatter electron intensity"

        # Create the interactive signal
        interactive(
            f=sliced_signal.nansum,
            axis=sliced_signal.axes_manager.signal_axes,
            event=roi.events.changed,
            recompute_out_event=None,
            out=out,
        )

        # Plot the result
        out.plot(**kwargs)

    def get_virtual_bse_intensity(
        self,
        roi: BaseInteractiveROI,
        out_signal_axes: Union[None, Iterable[int], Iterable[str]] = None,
    ) -> VirtualBSEImage:
        """Get a virtual backscatter electron (VBSE) image formed from
        intensities within a region of interest (ROI) on the detector.

        Adapted from
        :meth:`pyxem.signals.common_diffraction.CommonDiffraction.get_integrated_intensity`.

        Parameters
        ----------
        roi
            Any interactive ROI detailed in HyperSpy.
        out_signal_axes
            Which navigation axes to use as signal axes in the virtual
            image. If None (default), the first two navigation axes are
            used.

        Returns
        -------
        virtual_image : kikuchipy.signals.VirtualBSEImage
            VBSE image formed from detector intensities within an ROI
            on the detector.

        Examples
        --------
        >>> import hyperspy.api as hs
        >>> import kikuchipy as kp
        >>> rect_roi = hs.roi.RectangularROI(
        ...     left=0, right=5, top=0, bottom=5
        ... )
        >>> s = kp.data.nickel_ebsd_small()
        >>> vbse_image = s.get_virtual_bse_intensity(rect_roi)

        See Also
        --------
        ~kikuchipy.signals.EBSD.plot_virtual_bse_intensity
        """
        vbse = roi(self, axes=self.axes_manager.signal_axes)
        vbse_sum = self._get_sum_signal(vbse, out_signal_axes)
        vbse_sum.metadata.General.title = "Virtual backscatter electron image"
        vbse_sum.set_signal_type("VirtualBSEImage")
        return vbse_sum

    # ------ Methods overwritten from hyperspy.signals.Signal2D ------ #

    def deepcopy(self):
        new = super().deepcopy()
        if self.xmap is not None:
            new._xmap = self.xmap.deepcopy()
        else:
            new._xmap = copy.deepcopy(self.xmap)
        new._detector = self.detector.deepcopy()
        return new

    def save(
        self,
        filename: Optional[str] = None,
        overwrite: Optional[bool] = None,
        extension: Optional[str] = None,
        **kwargs,
    ):
        """Write the signal to file in the specified format.

        The function gets the format from the extension: `h5`, `hdf5` or
        `h5ebsd` for kikuchipy's specification of the the h5ebsd
        format, `dat` for the NORDIF binary format or `hspy` for
        HyperSpy's HDF5 specification. If no extension is provided the
        signal is written to a file in kikuchipy's h5ebsd format. Each
        format accepts a different set of parameters.

        For details see the specific format documentation under "See
        Also" below.

        This method is a modified version of HyperSpy's function
        :meth:`hyperspy.signal.BaseSignal.save`.

        Parameters
        ----------
        filename
            If None (default) and `tmp_parameters.filename` and
            `tmp_parameters.folder` in signal metadata are defined, the
            filename and path will be taken from there. A valid
            extension can be provided e.g. "data.h5", see `extension`.
        overwrite
            If None and the file exists, it will query the user. If
            True (False) it (does not) overwrite the file if it exists.
        extension
            Extension of the file that defines the file format. Options
            are "h5"/"hdf5"/"h5ebsd"/"dat"/"hspy". "h5"/"hdf5"/"h5ebsd"
            are equivalent. If None, the extension is determined from
            the following list in this order: i) the filename, ii)
            `tmp_parameters.extension` or iii) "h5" (kikuchipy's h5ebsd
            format).
        **kwargs :
            Keyword arguments passed to writer.

        See Also
        --------
        kikuchipy.io.plugins.h5ebsd.file_writer
        kikuchipy.io.plugins.nordif.file_writer
        """
        if filename is None:
            tmp_params = self.tmp_parameters
            if tmp_params.has_item("filename") and tmp_params.has_item("folder"):
                filename = os.path.join(tmp_params.folder, tmp_params.filename)
                extension = tmp_params.extension if not extension else extension
            elif self.metadata.has_item("General.original_filename"):
                filename = self.metadata.General.original_filename
            else:
                raise ValueError("Filename not defined.")
        if extension is not None:
            basename, _ = os.path.splitext(filename)
            filename = basename + "." + extension
        _save(filename, self, overwrite=overwrite, **kwargs)

    def get_decomposition_model(
        self,
        components: Union[None, int, List[int]] = None,
        dtype_out: np.dtype = np.float32,
    ):
        """Get the model signal generated with the selected number of
        principal components from a decomposition.

        Calls HyperSpy's
        :meth:`hyperspy.learn.mva.MVA.get_decomposition_model`.
        Learning results are preconditioned before this call, doing the
        following: (1) set :class:`numpy.dtype` to desired
        `dtype_out`, (2) remove unwanted components, and (3) rechunk,
        if :class:`dask.array.Array`, to suitable chunks.

        Parameters
        ----------
        components
            If None (default), rebuilds the signal from all components.
            If int, rebuilds signal from `components` in range 0-given
            int. If list of ints, rebuilds signal from only `components`
            in given list.
        dtype_out
            Data type to cast learning results to (default is
            :class:`numpy.float32`). Note that HyperSpy casts them to
            :class:`numpy.float64`.

        Returns
        -------
        s_model : EBSD or LazyEBSD
        """
        # Keep original results to revert back after updating
        factors_orig = self.learning_results.factors.copy()
        loadings_orig = self.learning_results.loadings.copy()

        # Change data type, keep desired components and rechunk if lazy
        (
            self.learning_results.factors,
            self.learning_results.loadings,
        ) = _update_learning_results(
            learning_results=self.learning_results,
            dtype_out=dtype_out,
            components=components,
        )

        # Call HyperSpy's function
        s_model = super().get_decomposition_model()

        # Revert learning results to original results
        self.learning_results.factors = factors_orig
        self.learning_results.loadings = loadings_orig

        # Remove learning results from model signal
        s_model.learning_results = LearningResults()

        return s_model

    # ------------------------ Private methods ----------------------- #

    def _check_refinement_parameters(
        self,
        xmap: CrystalMap,
        detector: EBSDDetector,
        mask: Union[np.ndarray, None],
    ):
        """Raise ValueError if EBSD refinement input is invalid."""
        _crystal_map_is_compatible_with_signal(
            xmap=xmap,
            navigation_axes=self.axes_manager.navigation_axes[::-1],
            raise_if_not=True,
        )
        sig_shape = self.axes_manager.signal_shape[::-1]
        _detector_is_compatible_with_signal(
            detector=detector,
            navigation_shape=self.axes_manager.navigation_shape[::-1],
            signal_shape=sig_shape,
            raise_if_not=True,
        )
        if len(xmap.phases.ids) != 1:
            raise ValueError("Crystal map must have exactly one phase")
        if mask is not None and sig_shape != mask.shape:
            raise ValueError("Mask and signal must have the same shape")

    def _get_dask_array_for_refinement(
        self, rechunk: bool, chunk_kwargs: Optional[dict] = None
    ) -> da.Array:
        """Possibly rechunk pattern array before refinement."""
        patterns = get_dask_array(signal=self)

        # Rechunk if (1) only one chunk and (2) it's allowed
        if (patterns.chunksize == patterns.shape) and rechunk:
            if chunk_kwargs is None:
                chunk_kwargs = dict(chunk_shape=16, chunk_bytes=None)
            chunks = get_chunking(signal=self, **chunk_kwargs)
            patterns = patterns.rechunk(chunks)

        return patterns

    @staticmethod
    def _get_sum_signal(signal, out_signal_axes: Optional[List] = None):
        out = signal.nansum(signal.axes_manager.signal_axes)
        if out_signal_axes is None:
            out_signal_axes = list(
                np.arange(min(signal.axes_manager.navigation_dimension, 2))
            )
        if len(out_signal_axes) > signal.axes_manager.navigation_dimension:
            raise ValueError(
                "The length of 'out_signal_axes' cannot be longer than the navigation "
                "dimension of the signal."
            )
        out.set_signal_type("")
        return out.transpose(out_signal_axes)

    def _prepare_metric(
        self,
        metric: Union[SimilarityMetric, str],
        signal_mask: Union[np.ndarray, None],
        dtype: Optional[np.dtype],
        rechunk: bool,
        n_dictionary_patterns: int,
    ) -> SimilarityMetric:
        if isinstance(metric, str) and metric in metrics:
            metric_class = metrics[metric]
            metric = metric_class()
            metric.rechunk = rechunk
        if not isinstance(metric, SimilarityMetric):
            raise ValueError(
                f"'{metric}' must be either of {metrics.keys()} or a custom metric "
                "class inheriting from SimilarityMetric. See "
                "kikuchipy.indexing.similarity_metrics.SimilarityMetric"
            )
        metric.n_experimental_patterns = max(self.axes_manager.navigation_size, 1)
        metric.n_dictionary_patterns = max(n_dictionary_patterns, 1)
        if signal_mask is not None:
            metric.signal_mask = ~signal_mask
        if dtype is not None:
            metric.dtype = dtype
        metric.raise_error_if_invalid()
        return metric


class LazyEBSD(LazySignal2D, EBSD):
    """Lazy implementation of the :class:`EBSD` class.

    This class extends HyperSpy's LazySignal2D class for EBSD patterns.

    Methods inherited from HyperSpy can be found in the HyperSpy user
    guide.

    See docstring of :class:`EBSD` for attributes and methods.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self, *args, **kwargs):
        xmap = self.xmap
        super().compute(*args, **kwargs)
        gc.collect()  # Don't sink
        self._xmap = xmap

    def get_decomposition_model_write(
        self,
        components: Union[None, int, List[int]] = None,
        dtype_learn: np.dtype = np.float32,
        mbytes_chunk: int = 100,
        dir_out: Optional[str] = None,
        fname_out: Optional[str] = None,
    ):
        """Write the model signal generated from the selected number of
        principal components directly to an .hspy file.

        The model signal intensities are rescaled to the original
        signals' data type range, keeping relative intensities.

        Parameters
        ----------
        components
            If None (default), rebuilds the signal from all
            `components`. If int, rebuilds signal from `components` in
            range 0-given int. If list of ints, rebuilds signal from
            only `components` in given list.
        dtype_learn
            Data type to set learning results to (default is
            :class:`numpy.float32`) before multiplication.
        mbytes_chunk
            Size of learning results chunks in MB, default is 100 MB as
            suggested in the Dask documentation.
        dir_out
            Directory to place output signal in.
        fname_out
            Name of output signal file.

        Notes
        -----
        Multiplying the learning results' factors and loadings in memory
        to create the model signal cannot sometimes be done due to too
        large matrices. Here, instead, learning results are written to
        file, read into dask arrays and multiplied using
        :func:`dask.array.matmul`, out of core.
        """
        # Change data type, keep desired components and rechunk if lazy
        factors, loadings = _update_learning_results(
            self.learning_results, components=components, dtype_out=dtype_learn
        )

        # Write learning results to HDF5 file
        if dir_out is None:
            try:
                dir_out = self.original_metadata.General.original_filepath
            except AttributeError:
                raise AttributeError("Output directory has to be specified")

        t_str = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        file_learn = os.path.join(dir_out, "learn_" + t_str + ".h5")
        with File(file_learn, mode="w") as f:
            f.create_dataset(name="factors", data=factors)
            f.create_dataset(name="loadings", data=loadings)

        # Matrix multiplication
        with File(file_learn, mode="r") as f:
            # Read learning results from HDF5 file
            chunks = _rechunk_learning_results(
                factors=factors, loadings=loadings, mbytes_chunk=mbytes_chunk
            )
            factors = da.from_array(f["factors"], chunks=chunks[0])
            loadings = da.from_array(f["loadings"], chunks=chunks[1])

            # Perform matrix multiplication
            loadings = loadings.T
            res = factors @ loadings
            res = res.T  # Transpose

            # Create new signal from result of matrix multiplication
            s_model = self.deepcopy()
            s_model.learning_results = LearningResults()
            s_model.data = res.reshape(s_model.data.shape)
            s_model.data = s_model.data.rechunk(chunks=(1, 1, -1, -1))

            # Rescale intensities
            s_model.rescale_intensity(dtype_out=self.data.dtype, relative=True)

            # Write signal to file
            if fname_out is None:
                fname_out = "model_" + t_str
            file_model = os.path.join(dir_out, fname_out)
            s_model.save(file_model)

        # Delete temporary files
        os.remove(file_learn)
        gc.collect()  # Don't sink
