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

from kikuchipy.io._io import _save
from kikuchipy.filters.fft_barnes import (
    _fft_filter,
    _fft_filter_setup,
)
from kikuchipy.filters.window import Window
from kikuchipy.pattern import chunk
from kikuchipy.pattern._pattern import (
    fft_frequency_vectors,
    fft_filter,
    _dynamic_background_frequency_space_setup,
)
from kikuchipy.indexing import StaticPatternMatching
from kikuchipy.indexing.similarity_metrics import SimilarityMetric
from kikuchipy.signals.util._metadata import (
    ebsd_metadata,
    metadata_nodes,
    _update_phase_info,
    _write_parameters_to_dictionary,
)
from kikuchipy.signals.util._dask import (
    _get_dask_array,
    _rechunk_learning_results,
    _update_learning_results,
)
from kikuchipy.signals.virtual_bse_image import VirtualBSEImage
from kikuchipy.signals._common_image import CommonImage
from kikuchipy.detectors import EBSDDetector


class EBSD(CommonImage, Signal2D):
    """Scan of Electron Backscatter Diffraction (EBSD) patterns.

    This class extends HyperSpy's Signal2D class for EBSD patterns, with
    common intensity processing methods and some analysis methods.

    Methods inherited from HyperSpy can be found in the HyperSpy user
    guide.

    See the docstring of :class:`hyperspy.signal.BaseSignal` for a list
    of attributes.
    """

    _signal_type = "EBSD"
    _alias_signal_types = ["electron_backscatter_diffraction"]
    _lazy = False

    def __init__(self, *args, **kwargs):
        """Create an :class:`~kikuchipy.signals.EBSD` object from a
        :class:`hyperspy.signals.Signal2D` or a :class:`numpy.ndarray`.
        """
        Signal2D.__init__(self, *args, **kwargs)

        #        if "detector_dict" in kwargs:
        #            self.detector = EBSDDetector(kwargs.pop("detector_dict"))
        #        else:
        self.detector = EBSDDetector(
            shape=self.axes_manager.signal_shape,
            px_size=self.axes_manager.signal_axes[0].scale,
        )

        self._xmap = kwargs.pop("xmap", None)

        # Update metadata if object is initialised from numpy array
        if not self.metadata.has_item(metadata_nodes("ebsd")):
            md = self.metadata.as_dictionary()
            md.update(ebsd_metadata().as_dictionary())
            self.metadata = DictionaryTreeBrowser(md)
        if not self.metadata.has_item("Sample.Phases"):
            self.set_phase_parameters()

    @property
    def xmap(self) -> CrystalMap:
        """A :class:`~orix.crystal_map.CrystalMap` containing the
        phases, unit cell rotations and auxiliary properties of the EBSD
        data set.
        """
        return self._xmap

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
        >>> ebsd_node = metadata_nodes("ebsd")
        >>> s.metadata.get_item(ebsd_node + '.xpc')
        1.0
        >>> s.set_experimental_parameters(xpc=0.50726)
        >>> s.metadata.get_item(ebsd_node + '.xpc')
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
        >>> s.metadata.Sample.Phases.Number_1.atom_coordinates.Number_1
        ├── atom =
        ├── coordinates = array([0., 0., 0.])
        ├── debye_waller_factor = 0.0
        └── site_occupation = 0.0
        >>> s.set_phase_parameters(
        ...     number=1, atom_coordinates={
        ...         '1': {'atom': 'Ni', 'coordinates': [0, 0, 0],
        ...         'site_occupation': 1,
        ...         'debye_waller_factor': 0.0035}})
        >>> s.metadata.Sample.Phases.Number_1.atom_coordinates.Number_1
        ├── atom = Ni
        ├── coordinates = array([0., 0., 0.])
        ├── debye_waller_factor = 0.0035
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
        >>> s.axes_manager.['x'].scale  # Default value
        1.0
        >>> s.set_scan_calibration(step_x=1.5)  # Microns
        >>> s.axes_manager['x'].scale
        1.5
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
        >>> ebsd_node = kp.signals.util.metadata_nodes("ebsd")
        >>> s.metadata.get_item(ebsd_node + '.static_background')
        [[84 87 90 ... 27 29 30]
        [87 90 93 ... 27 28 30]
        [92 94 97 ... 39 28 29]
        ...
        [80 82 84 ... 36 30 26]
        [79 80 82 ... 28 26 26]
        [76 78 80 ... 26 26 25]]

        The static background can be removed by subtracting or dividing
        this background from each pattern while keeping relative
        intensities between patterns (or not):

        >>> s.remove_static_background(
        ...     operation='subtract', relative=True)

        If the metadata has no background pattern, this must be passed
        in the `static_bg` parameter as a numpy or dask array.
        """
        dtype_out = self.data.dtype.type

        # Get background pattern
        if not isinstance(static_bg, (np.ndarray, da.Array)):
            try:
                md = self.metadata
                ebsd_node = metadata_nodes("ebsd")
                static_bg = da.from_array(
                    md.get_item(ebsd_node + ".static_background"),
                    chunks="auto",
                )
            except AttributeError:
                raise OSError(
                    "The static background is not a numpy or dask array or "
                    "could not be read from signal metadata."
                )
        if dtype_out != static_bg.dtype:
            raise ValueError(
                f"The static background dtype_out {static_bg.dtype} is not the "
                f"same as pattern dtype_out {dtype_out}."
            )
        pat_shape = self.axes_manager.signal_shape
        bg_shape = static_bg.shape
        if bg_shape != pat_shape:
            raise OSError(
                f"The pattern {pat_shape} and static background {bg_shape} "
                "shapes are not identical."
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
        dask_array = _get_dask_array(signal=self, dtype=dtype)

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
        range.

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

        >>> s.remove_static_background(operation="subtract")
        >>> s.remove_dynamic_background(
        ...     operation="subtract",  # Default
        ...     filter_domain="frequency",  # Default
        ...     truncate=4.0,  # Default
        ...     std=5,
        ... )
        """
        # Create a dask array of signal patterns and do the processing on this
        dtype = np.float32
        dask_array = _get_dask_array(signal=self, dtype=dtype)

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
            raise ValueError(
                f"{filter_domain} must be either of {filter_domains}."
            )

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
            raise ValueError(
                f"{filter_domain} must be either of {filter_domains}."
            )

        if dtype_out is None:
            dtype_out = self.data.dtype.type
        dask_array = _get_dask_array(self, dtype=dtype_out)

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
        >>> s2 = s.inav[0, 0]
        >>> s2.adaptive_histogram_equalization()
        >>> imin = np.iinfo(s.data.dtype_out).min
        >>> imax = np.iinfo(s.data.dtype_out).max + 1
        >>> hist, _ = np.histogram(
        ...     s.inav[0, 0].data, bins=imax, range=(imin, imax))
        >>> hist2, _ = np.histogram(
        ...     s2.inav[0, 0].data, bins=imax, range=(imin, imax))
        >>> fig, ax = plt.subplots(nrows=2, ncols=2)
        >>> ax[0, 0].imshow(s.inav[0, 0].data)
        >>> ax[1, 0].plot(hist)
        >>> ax[0, 1].imshow(s2.inav[0, 0].data)
        >>> ax[1, 1].plot(hist2)

        Notes
        -----
        * It is recommended to perform adaptive histogram equalization
          only *after* static and dynamic background corrections,
          otherwise some unwanted darkening towards the edges might
          occur.
        * The default window size might not fit all image sizes, so it
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
        dask_array = _get_dask_array(signal=self)

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
        >>> iq = s.get_image_quality(normalize=True)  # Default
        >>> plt.imshow(iq)

        See Also
        --------
        kikuchipy.pattern.get_image_quality
        """
        # Data set to operate on
        dtype_out = np.float32
        dask_array = _get_dask_array(self, dtype=dtype_out)

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

    def match_patterns(
        self,
        simulations,
        metric: Union[str, SimilarityMetric] = "ncc",
        keep_n: int = 50,
        n_slices: int = 1,
        return_merged_crystal_map: bool = False,
        get_orientation_similarity_map: bool = False,
    ) -> Union[CrystalMap, List[CrystalMap]]:
        """Match each experimental pattern to all simulated patterns, of
        known crystal orientations in pre-computed dictionaries
        :cite:`chen2015dictionary,jackson2019dictionary`, to determine
        their phase and orientation.

        A suitable similarity metric, the normalized cross-correlation
        (:func:`~kikuchipy.indexing.similarity_metrics.ncc`), is used by
        default, but a valid user-defined similarity metric may be used
        instead (see
        :func:`~kikuchipy.indexing.similarity_metrics.make_similarity_metric`).

        :class:`~orix.crystal_map.crystal_map.CrystalMap`'s for each
        dictionary with "scores" and "simulation_indices" as properties
        are returned.

        Parameters
        ----------
        simulations : EBSD or list of EBSD
            An EBSD signal or a list of EBSD signals with simulated
            patterns (dictionaries). The signals must have a 1D
            navigation axis and the `xmap` property with crystal
            orientations set.
        metric : str or SimilarityMetric, optional
            Similarity metric, by default "ncc" (normalized
            cross-correlation).
        keep_n : int, optional
            Number of best matches to keep, by default 50 or the number
            of simulated patterns if fewer than 50 are available.
        n_slices : int, optional
            Number of simulation slices to process sequentially, by
            default 1 (no slicing).
        return_merged_crystal_map : bool, optional
            Whether to return a merged crystal map, the best matches
            determined from the similarity scores, in addition to the
            single phase maps. By default False.
        get_orientation_similarity_map : bool, optional
            Add orientation similarity maps to the returned crystal
            maps' properties named "osm". By default False.

        Returns
        -------
        xmaps : ~orix.crystal_map.crystal_map.CrystalMap or list of \
                ~orix.crystal_map.crystal_map.CrystalMap
            A crystal map for each dictionary loaded and one merged map
            if `return_merged_crystal_map = True`.

        Notes
        -----
        Merging of crystal maps and calculations of orientation
        similarity maps can be done afterwards with
        :func:`~kikuchipy.indexing.merge_crystal_maps` and
        :func:`~kikuchipy.indexing.orientation_similarity_map`,
        respectively.

        See Also
        --------
        ~kikuchipy.indexing.similarity_metrics.make_similarity_metric
        ~kikuchipy.indexing.similarity_metrics.ndp
        """
        sdi = StaticPatternMatching(simulations)
        return sdi(
            signal=self,
            metric=metric,
            keep_n=keep_n,
            n_slices=n_slices,
            return_merged_crystal_map=return_merged_crystal_map,
            get_orientation_similarity_map=get_orientation_similarity_map,
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

        >>> pattern_shape = s.axes_manager.signal_shape[::-1]
        >>> w = kp.filters.Window(
        ...     "lowpass", cutoff=20, shape=pattern_shape)
        >>> s.fft_filter(
        ...     transfer_function=w,
        ...     function_domain="frequency",
        ...     shift=True,
        ... )

        See Also
        --------
        ~kikuchipy.filters.window.Window
        """
        dtype_out = self.data.dtype

        dtype = np.float32
        dask_array = _get_dask_array(signal=self, dtype=dtype)

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
            raise ValueError(
                f"{function_domain} must be either of {function_domains}."
            )

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
        extended with zeros.

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

        Examples
        --------
        >>> import numpy as np
        >>> import kikuchipy as kp
        >>> s = kp.signals.EBSD(np.ones((4, 4, 1, 1)))
        >>> k = 1
        >>> for i in range(4):
        ...     for j in range(4):
        ...         s.inav[j, i].data *= k
        ...         k += 1
        >>> s.data[:, :, 0, 0]
        array([[ 1.,  2.,  3.,  4.],
               [ 5.,  6.,  7.,  8.],
               [ 9., 10., 11., 12.],
               [13., 14., 15., 16.]])
        >>> s2 = s.deepcopy()  # For use below
        >>> s.average_neighbour_patterns(
        ...     window="circular", shape=(3, 3))
        >>> s.data[:, :, 0, 0]
        array([[ 2.66666667,  3.        ,  4.        ,  5.        ],
               [ 5.25      ,  6.        ,  7.        ,  7.75      ],
               [ 9.25      , 10.        , 11.        , 11.75      ],
               [12.        , 13.        , 14.        , 14.33333333]])

        A window object can also be passed

        >>> w = kp.filters.Window(window="gaussian", std=2)
        >>> w
        Window (3, 3) gaussian
        [[0.7788 0.8825 0.7788]
         [0.8825 1.     0.8825]
         [0.7788 0.8825 0.7788]]
        >>> s2.average_neighbour_patterns(w)
        >>> s2.data[:, :, 0, 0]
        array([[ 3.34395304,  3.87516243,  4.87516264,  5.40637176],
               [ 5.46879047,  5.99999985,  6.99999991,  7.53120901],
               [ 9.46879095,  9.99999959, 11.00000015, 11.53120913],
               [11.59362845, 12.12483732, 13.1248368 , 13.65604717]])

        This window can subsequently be plotted and saved

        >>> figure, image, colorbar = w.plot()
        >>> figure.savefig('averaging_window.png')
        """
        if isinstance(window, Window) and window.is_valid():
            averaging_window = copy.copy(window)
        else:
            averaging_window = Window(
                window=window,
                shape=window_shape,
                **kwargs,
            )
        averaging_window.shape_compatible(self.axes_manager.signal_shape)

        # Do nothing if a window of shape (1, ) or (1, 1) is passed
        window_shape = averaging_window.shape
        nav_shape = self.axes_manager.navigation_shape
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
            input=np.ones(self.axes_manager.navigation_shape[::-1]),
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
        #        averaging_window._add_axes(self.axes_manager.signal_dimension)

        # Create dask array of signal patterns and do processing on this
        dask_array = _get_dask_array(signal=self)

        # Add signal dimensions to array be able to use with Dask's map_blocks()
        nav_dim = self.axes_manager.navigation_dimension
        for i in range(sig_dim):
            window_sums = np.expand_dims(window_sums, axis=window_sums.ndim)
        window_sums = da.from_array(
            window_sums, chunks=dask_array.chunksize[:nav_dim] + (1,) * sig_dim
        )

        # Create overlap between chunks to enable correlation with the window
        # using Dask's map_blocks()
        data_dim = len(self.axes_manager.shape)
        overlap_depth = {}
        for i in range(data_dim):
            if i < len(window_shape):
                overlap_depth[i] = window_shape[i] // 2
            else:
                overlap_depth[i] = 0
        overlap_boundary = {i: "none" for i in range(data_dim)}
        overlapped_dask_array = da.overlap.overlap(
            dask_array,
            depth=overlap_depth,
            boundary=overlap_boundary,
        )

        # Must also be overlapped, since the patterns are overlapped
        overlapped_window_sums = da.overlap.overlap(
            window_sums, depth=overlap_depth, boundary=overlap_boundary
        )

        # Finally, average patterns by correlation with the window and
        # subsequent division by the number of neighbours correlated with
        dtype_out = self.data.dtype
        overlapped_averaged_patterns = da.map_blocks(
            chunk.average_neighbour_patterns,
            overlapped_dask_array,
            overlapped_window_sums,
            window=averaging_window,
            dtype_out=dtype_out,
            dtype=dtype_out,
        )

        # Trim overlapping patterns
        averaged_patterns = da.overlap.trim_overlap(
            overlapped_averaged_patterns,
            depth=overlap_depth,
            boundary=overlap_boundary,
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
        >>> roi = hs.roi.RectangularROI(
        ...     left=0, right=5, top=0, bottom=5)
        >>> s.plot_virtual_bse_intensity(roi)

        See Also
        --------
        ~kikuchipy.signals.EBSD.get_virtual_bse_intensity
        """
        # Plot signal if necessary
        if self._plot is None or not self._plot.is_active:
            self.plot()

        # Get the sliced signal from the ROI
        sliced_signal = roi.interactive(
            self, axes=self.axes_manager.signal_axes
        )

        # Create an output signal for the virtual backscatter electron
        # calculation
        out = self._get_sum_signal(self, out_signal_axes)
        out.metadata.General.title = "Virtual backscatter electron intensity"

        # Create the interactive signal
        interactive(
            f=sliced_signal.sum,
            axis=sliced_signal.axes_manager.signal_axes,
            event=roi.events.changed,
            recompute_out_event=None,
            out=out,
        )

        # Plot the result
        out.plot(**kwargs)

    @staticmethod
    def _get_sum_signal(signal, out_signal_axes: Optional[List] = None):
        out = signal.sum(signal.axes_manager.signal_axes)
        if out_signal_axes is None:
            out_signal_axes = list(
                np.arange(min(signal.axes_manager.navigation_dimension, 2))
            )
        if len(out_signal_axes) > signal.axes_manager.navigation_dimension:
            raise ValueError(
                "The length of 'out_signal_axes' cannot be longer than the "
                "navigation dimension of the signal."
            )
        out.set_signal_type("")
        return out.transpose(out_signal_axes)

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
        >>> roi = hs.roi.RectangularROI(
        ...     left=0, right=5, top=0, bottom=5)
        >>> vbse_image = s.get_virtual_bse_intensity(roi)

        See Also
        --------
        ~kikuchipy.signals.EBSD.plot_virtual_bse_intensity
        """
        vbse = roi(self, axes=self.axes_manager.signal_axes)
        vbse_sum = self._get_sum_signal(vbse, out_signal_axes)
        vbse_sum.metadata.General.title = "Virtual backscatter electron image"
        vbse_sum.set_signal_type("VirtualBSEImage")
        return vbse_sum

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
            if self.tmp_parameters.has_item(
                "filename"
            ) and self.tmp_parameters.has_item("folder"):
                filename = os.path.join(
                    self.tmp_parameters.folder, self.tmp_parameters.filename
                )
                extension = (
                    self.tmp_parameters.extension
                    if not extension
                    else extension
                )
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

    def rebin(self, new_shape=None, scale=None, crop=True, out=None):
        s_out = super().rebin(
            new_shape=new_shape, scale=scale, crop=crop, out=out
        )

        return_signal = True
        if s_out is None:
            s_out = out
            return_signal = False

        # Update binning in metadata to signal dimension with largest or lowest
        # binning if downscaling or upscaling, respectively
        md = s_out.metadata
        ebsd_node = metadata_nodes("ebsd")
        if scale is None:
            sx, sy = self.axes_manager.signal_shape
            signal_idx = self.axes_manager.signal_indices_in_array
            scale = (
                sx / new_shape[signal_idx[0]],
                sy / new_shape[signal_idx[1]],
            )
        upscaled_dimensions = np.where(np.array(scale) < 1)[0]
        if len(upscaled_dimensions):
            new_binning = np.min(scale)
        else:
            new_binning = np.max(scale)
        original_binning = abs(md.get_item(ebsd_node + ".binning"))
        md.set_item(ebsd_node + ".binning", original_binning * new_binning)

        if return_signal:
            return s_out


class LazyEBSD(EBSD, LazySignal2D):
    """Lazy implementation of the :class:`EBSD` class.

    This class extends HyperSpy's LazySignal2D class for EBSD patterns.

    Methods inherited from HyperSpy can be found in the HyperSpy user
    guide.

    See docstring of :class:`EBSD` for attributes and methods.
    """

    _lazy = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
