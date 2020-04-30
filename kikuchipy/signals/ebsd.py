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
from typing import Union
import warnings

import dask.array as da
from dask.diagnostics import ProgressBar
from hyperspy._signals.signal2d import Signal2D
from hyperspy._lazy_signals import LazySignal2D
from hyperspy.learn.mva import LearningResults
from hyperspy.misc.utils import DictionaryTreeBrowser
from h5py import File
import numpy as np
from pyxem.signals.diffraction2d import Diffraction2D
from scipy.ndimage import correlate, gaussian_filter
from skimage.util.dtype import dtype_range

from kikuchipy.io._io import save
import kikuchipy as kp


class EBSD(Signal2D):
    """Scan of Electron Backscatter Diffraction (EBSD) patterns.

    This class extends HyperSpy's Signal2D class for EBSD patterns, with
    many common intensity processing methods and some analysis methods.

    Methods inherited from HyperSpy can be found in the HyperSpy user
    guide.

    See the docstring of :meth:`hyperspy.signal.BaseSignal.__init__` for
    a list of attributes.

    """

    _signal_type = "EBSD"
    _alias_signal_types = ["electron_backscatter_diffraction"]
    _lazy = False

    def __init__(self, *args, **kwargs):
        """Create an :class:`~kikuchipy.signals.EBSD` object from a
        :class:`hyperspy.signals.Signal2D` or a :class:`numpy.ndarray`.

        """

        Signal2D.__init__(self, *args, **kwargs)

        # Update metadata if object is initialised from numpy array
        if not self.metadata.has_item(kp.util.io.metadata_nodes(sem=False)):
            md = self.metadata.as_dictionary()
            md.update(kp.util.io.kikuchipy_metadata().as_dictionary())
            self.metadata = DictionaryTreeBrowser(md)
        if not self.metadata.has_item("Sample.Phases"):
            self.set_phase_parameters()

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
        set_phase_parameters

        Examples
        --------
        >>> import kikuchipy as kp
        >>> ebsd_node = kp.util.io.metadata_nodes(sem=False)
        >>> s.metadata.get_item(ebsd_node + '.xpc')
        1.0
        >>> s.set_experimental_parameters(xpc=0.50726)
        >>> s.metadata.get_item(ebsd_node + '.xpc')
        0.50726

        """

        md = self.metadata
        sem_node, ebsd_node = kp.util.io.metadata_nodes()
        kp.util.general._write_parameters_to_dictionary(
            {
                "beam_energy": beam_energy,
                "magnification": magnification,
                "microscope": microscope,
                "working_distance": working_distance,
            },
            md,
            sem_node,
        )
        kp.util.general._write_parameters_to_dictionary(
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
        space_group : int, optional
            Number between 1 and 230.
        symmetry : int, optional
            Phase symmetry.

        See Also
        --------
        set_experimental_parameters

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
            for phase, val in atom_coordinates.items():
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
            "space_group": space_group,
            "symmetry": symmetry,
        }

        # Remove None values
        phase = {k: v for k, v in inputs.items() if v is not None}
        kp.util.phase._update_phase_info(self.metadata, phase, number)

    def set_scan_calibration(self, step_x=1.0, step_y=1.0):
        """Set the step size in microns.

        Parameters
        ----------
        step_x : float
            Scan step size in um per pixel in horizontal direction.
        step_y : float
            Scan step size in um per pixel in vertical direction.

        See Also
        --------
        set_detector_calibration

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
        x.units, y.units = ["\u03BC" + "m"] * 2

    def set_detector_calibration(self, delta):
        """Set detector pixel size in microns. The offset is set to the
        the detector centre.

        Parameters
        ----------
        delta : float
            Detector pixel size in microns.

        See Also
        --------
        set_scan_calibration

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
        dx.units, dy.units = ["\u03BC" + "m"] * 2
        dx.scale, dy.scale = (delta, delta)
        dx.offset, dy.offset = -centre

    def remove_static_background(
        self,
        operation="subtract",
        relative=True,
        static_bg=None,
        scale_bg=False,
    ):
        """Remove the static background in an EBSD scan inplace.

        The removal is performed by subtracting or dividing by a static
        background pattern. Resulting pattern intensities are rescaled
        keeping relative intensities or not and stretched to fill the
        available grey levels in the patterns' data type range.

        Parameters
        ----------
        operation : "subtract" or "divide", optional
            Whether to subtract (default) or divide by the static
            background pattern.
        relative : bool, optional
            Keep relative intensities between patterns (default is
            True).
        static_bg : None, numpy.ndarray, or dask.array.Array, optional
            Static background pattern. If None is passed (default) we
            try to read it from the signal metadata.
        scale_bg : bool, optional
            Whether to scale the static background pattern to each
            individual pattern's data range before removal (default is
            False). Must be False if `relative` is True.

        See Also
        --------
        remove_dynamic_background
        kikuchipy.util.chunk.remove_static_background

        Examples
        --------
        We assume that a static background pattern with the same shape
        and data type (e.g. 8-bit unsigned integer, ``uint8``) as the
        patterns is available in signal metadata:

        >>> import kikuchipy as kp
        >>> ebsd_node = kp.util.io.metadata_nodes(sem=False)
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
        intensities between patterns (or not).

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
                ebsd_node = kp.util.io.metadata_nodes(sem=False)
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
        dask_array = kp.util.dask._get_dask_array(signal=self, dtype=dtype)

        # Remove the static background and rescale intensities chunk by chunk
        corrected_patterns = dask_array.map_blocks(
            func=kp.util.chunk.remove_static_background,
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
        intensities are rescaled to fill the input data type range.

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
        dynamic_background
        remove_static_background
        kikuchipy.util.chunk.remove_dynamic_background
        kikuchipy.util.pattern.remove_dynamic_background

        Examples
        --------
        Traditional background correction includes static and dynamic
        corrections, loosing relative intensities between patterns after
        dynamic corrections (whether `relative` is set to True or
        False in :meth:`~remove_static_background`):

        >>> s.remove_static_background(operation="subtract")
        >>> s.remove_dynamic_background(operation="subtract", std=2)

        """

        # Create a dask array of signal patterns and do the processing on this
        dtype = np.float32
        dask_array = kp.util.dask._get_dask_array(signal=self, dtype=dtype)

        if std is None:
            std = self.axes_manager.signal_shape[0] / 8

        # Get filter function and set up necessary keyword arguments
        if filter_domain == "frequency":
            # FFT filter setup for Connelly Barnes' algorithm
            filter_func = kp.util.barnes_fftfilter._fft_filter
            (
                kwargs["fft_shape"],
                kwargs["kernel_shape"],
                kwargs["kernel_fft"],
                kwargs["offset_before_fft"],
                kwargs["offset_after_ifft"],
            ) = kp.util.pattern._dynamic_background_frequency_space_setup(
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
        out_range = (0, dtype_range[dtype_out][-1])

        corrected_patterns = dask_array.map_blocks(
            func=kp.util.chunk.remove_dynamic_background,
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
        std=None,
        filter_domain="frequency",
        truncate=4.0,
        dtype_out=None,
        **kwargs,
    ):
        """Get the dynamic background per EBSD pattern in a scan.

        Parameters
        ----------
        std : None, int or float, optional
            Standard deviation of the Gaussian window. If None
            (default), it is set to width/8.
        filter_domain : "frequency", "spatial", optional
            Whether to apply a Gaussian convolution filter in the
            frequency (default) or spatial domain.
        truncate : float, optional
            Truncate the Gaussian filter at this many standard
            deviations. Default is 4.0.
        dtype_out : np.dtype, optional
            Data type of the background patterns. If None (default), it
            is set to the same data type as the input pattern.
        kwargs :
            Keyword arguments passed to the Gaussian blurring function
            determined from `filter_domain`.

        Returns
        -------
        background_signal : kikuchipy.signals.ebsd.EBSD or \
                kikuchipy.signals.ebsd.LazyEBSD
            Signal with the large scale variations across the detector.

        """

        if std is None:
            std = self.axes_manager.signal_shape[0] / 8

        # Get filter function and set up necessary keyword arguments
        if filter_domain == "frequency":
            filter_func = kp.util.barnes_fftfilter._fft_filter
            # FFT filter setup for Connelly Barnes' algorithm
            (
                kwargs["fft_shape"],
                kwargs["kernel_shape"],
                kwargs["kernel_fft"],
                kwargs["offset_before_fft"],
                kwargs["offset_after_ifft"],
            ) = kp.util.pattern._dynamic_background_frequency_space_setup(
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
        dask_array = kp.util.dask._get_dask_array(self, dtype=dtype_out)

        background_patterns = dask_array.map_blocks(
            func=kp.util.chunk.get_dynamic_background,
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
                background_signal = kp.signals.EBSD(background_return)
        else:
            background_signal = kp.signals.LazyEBSD(background_patterns)

        return background_signal

    def rescale_intensity(
        self,
        relative=False,
        in_range=None,
        out_range=None,
        dtype_out=None,
        percentiles=None,
    ):
        """Rescale pattern intensities in an EBSD scan inplace.

        Output min./max. intensity is determined from `out_range` or the
        data type range of the :class:`numpy.dtype` passed to
        `dtype_out` if `out_range` is None.

        This method is based on
        :func:`skimage.exposure.rescale_intensity`.

        Parameters
        ----------
        relative : bool, optional
            Whether to keep relative intensities between patterns
            (default is False). If True, `in_range` must be None,
            because `in_range` is in this case set to the global
            min./max. intensity.
        in_range : None or tuple of int or float, optional
            Min./max. intensity of input patterns. If None (default),
            stretching is performed when `in_range` is set to a narrower
            `in_range` is set to pattern min./max intensity. Contrast
            intensity range than the input patterns. Must be None if
            `relative` is True or `percentiles` are passed.
        out_range : None or tuple of int or float, optional
            Min./max. intensity of output patterns. If None (default),
            `out_range` is set to `dtype_out` min./max according to
            `skimage.util.dtype.dtype_range`, with min. equal to zero.
        dtype_out : None or numpy.dtype, optional
            Data type of rescaled patterns, default is input patterns'
            data type.
        percentiles : None or tuple of int or float, optional
            Disregard intensities outside these percentiles. Calculated
            per pattern. Must be None if `in_range` or `relative` is
            passed (default is None).

        See Also
        --------
        adaptive_histogram_equalization
        normalize_intensity
        :func:`skimage.exposure.rescale_intensity`
        kikuchipy.util.chunk.rescale_intensity
        kikuchipy.util.pattern.rescale_intensity

        Examples
        --------
        Pattern intensities are stretched to fill the available grey
        levels in the input patterns' data type range or any
        :class:`numpy.dtype` range passed to `dtype_out`, either
        keeping relative intensities between patterns or not:

        >>> print(s.data.dtype_out, s.data.min(), s.data.max(),
        ...       s.inav[0, 0].data.min(), s.inav[0, 0].data.max())
        uint8 20 254 24 233
        >>> s2 = s.deepcopy()
        >>> s.rescale_intensity(dtype_out=np.uint16)
        >>> print(s.data.dtype_out, s.data.min(), s.data.max(),
        ...       s.inav[0, 0].data.min(), s.inav[0, 0].data.max())
        uint16 0 65535 0 65535
        >>> s2.rescale_intensity(relative=True)
        >>> print(s2.data.dtype_out, s2.data.min(), s2.data.max(),
        ...       s2.inav[0, 0].data.min(), s2.inav[0, 0].data.max())
        uint8 0 255 4 232

        """

        # Determine min./max. intensity of input pattern to rescale to
        if in_range is not None and percentiles is not None:
            raise ValueError(
                "'percentiles' must be None if 'in_range' is not None."
            )
        elif relative is True and in_range is not None:
            raise ValueError("'in_range' must be None if 'relative' is True.")
        elif relative:  # Scale relative to min./max. intensity in scan
            in_range = (self.data.min(), self.data.max())

        if dtype_out is None:
            dtype_out = self.data.dtype.type

        if out_range is None:
            dtype_out_pass = dtype_out
            if isinstance(dtype_out, np.dtype):
                dtype_out_pass = dtype_out.type
            out_range = (0, dtype_range[dtype_out_pass][-1])

        # Create dask array of signal patterns and do processing on this
        dask_array = kp.util.dask._get_dask_array(signal=self)

        # Rescale patterns
        rescaled_patterns = dask_array.map_blocks(
            func=kp.util.chunk.rescale_intensity,
            in_range=in_range,
            out_range=out_range,
            dtype_out=dtype_out,
            percentiles=percentiles,
            dtype=dtype_out,
        )

        # Overwrite signal patterns
        if not self._lazy:
            with ProgressBar():
                if self.data.dtype != rescaled_patterns.dtype:
                    self.change_dtype(dtype_out)
                print("Rescaling the pattern intensities:", file=sys.stdout)
                rescaled_patterns.store(self.data, compute=True)
        else:
            self.data = rescaled_patterns

    def adaptive_histogram_equalization(
        self, kernel_size=None, clip_limit=0, nbins=128
    ):
        """Enhance the local contrast in an EBSD scan inplace using
        adaptive histogram equalization.

        This method uses :func:`skimage.exposure.equalize_adapthist`.

        Parameters
        ----------
        kernel_size : int or list-like, optional
            Shape of contextual regions for adaptive histogram
            equalization, default is 1/4 of image height and 1/4 of
            image width.
        clip_limit : float, optional
            Clipping limit, normalized between 0 and 1 (higher values
            give more contrast). Default is 0.
        nbins : int, optional
            Number of gray bins for histogram ("data range"), default is
            128.

        See also
        --------
        remove_static_background
        remove_dynamic_background
        rescale_intensity

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
        dask_array = kp.util.dask._get_dask_array(signal=self)

        # Local contrast enhancement
        equalized_patterns = dask_array.map_blocks(
            func=kp.util.chunk.adaptive_histogram_equalization,
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

    def get_image_quality(self, normalize=True):
        """Compute the image quality map of patterns in an EBSD scan.

        The image quality is calculated based on the procedure defined
        by Krieger Lassen [Lassen1994]_.

        Parameters
        ----------
        normalize : bool, optional
            Whether to normalize patterns to a mean of zero and standard
            deviation of 1 before calculating the image quality (default
            is True).

        Returns
        -------
        image_quality_map : np.ndarray
            Image quality map of same shape as signal navigation axes.

        References
        ----------
        .. [Lassen1994] N. C. K. Lassen, "Automated Determination of \
            Crystal Orientations from Electron Backscattering \
            Patterns," Institute of Mathematical Modelling, (1994).

        """

        # Data set to operate on
        dtype_out = np.float32
        dask_array = kp.util.dask._get_dask_array(self, dtype=dtype_out)

        # Calculate frequency vectors
        sx, sy = self.axes_manager.signal_shape
        frequency_vectors = kp.util.experimental.fft_frequency_vectors((sy, sx))
        inertia_max = np.sum(frequency_vectors) / (sy * sx)

        # Calculate image quality per chunk
        image_quality_map = dask_array.map_blocks(
            func=kp.util.chunk.get_image_quality,
            frequency_vectors=frequency_vectors,
            inertia_max=inertia_max,
            normalize=normalize,
            dtype_out=dtype_out,
            dtype=dtype_out,
            drop_axis=self.axes_manager.signal_indices_in_array,
        )

        if not self._lazy:
            with ProgressBar():
                print("Calculating the image quality:", file=sys.stdout)
                image_quality_map = image_quality_map.compute()

        return image_quality_map

    def fft_filter(self, transfer_function, shift=False, **kwargs):
        """Filter an EBSD scan inplace in the frequency domain.

        Patterns are transformed via the Fast Fourier Transform (FFT) to
        the frequency domain, where their spectrum is multiplied by a
        filter `transfer_function`, and the filtered spectrum is
        subsequently transformed to the spatial domain via the inverse
        FFT (IFFT). Filtered patterns are rescaled to input data type
        range.

        Parameters
        ----------
        transfer_function : numpy.ndarray or kikuchipy.util.Window
            Filter transfer function in the frequency domain of pattern
            shape.
        shift : bool, optional
            Whether to shift the zero-frequency component to the centre
            (default is False).
        kwargs :
            Keyword arguments passed to func:`scipy.fft.fft2`.

        """

        dtype_out = self.data.dtype

        dtype = np.float32
        dask_array = kp.util.dask._get_dask_array(signal=self, dtype=dtype)

        filtered_patterns = dask_array.map_blocks(
            func=kp.util.chunk.fft_filter,
            transfer_function=transfer_function,
            shift=shift,
            dtype_out=dtype_out,
            dtype=dtype,
            **kwargs,
        )

        # Overwrite signal patterns
        if not self._lazy:
            with ProgressBar():
                print("FFT filtering:", file=sys.stdout)
                filtered_patterns.store(self.data, compute=True)
        else:
            self.data = filtered_patterns

    def normalize_intensity(
        self, num_std=1, divide_by_square_root=False, dtype_out=None
    ):
        """Normalize pattern intensities in an EBSD scan inplace to a
        mean of zero with a given standard deviation.

        Parameters
        ----------
        num_std : int, optional
            Number of standard deviations of the output intensities
            (default is 1).
        divide_by_square_root : bool, optional
            Whether to divide output intensities by the square root of
            the image size (default is False).
        dtype_out : None or np.dtype, optional
            Output data type of normalized patterns. If None (default),
            the patterns' data type is used.

        Notes
        -----
        Data type should always be changed to floating point, e.g.
        ``np.float32`` with :meth:`~change_dtype`, before normalizing
        the intensities.

        """

        if dtype_out is None:
            dtype_out = self.data.dtype

        dask_array = kp.util.dask._get_dask_array(self, dtype=np.float32)

        normalized_patterns = dask_array.map_blocks(
            func=kp.util.chunk.normalize_intensity,
            num_std=num_std,
            divide_by_square_root=divide_by_square_root,
            dtype_out=dtype_out,
            dtype=dtype_out,
        )

        # Change data type if requested
        if dtype_out != self.data.dtype:
            self.change_dtype(dtype_out)

        # Overwrite signal patterns
        if not self._lazy:
            with ProgressBar():
                print("Normalizing the pattern intensities:", file=sys.stdout)
                normalized_patterns.store(self.data, compute=True)
        else:
            self.data = normalized_patterns

    def average_neighbour_patterns(
        self, window="circular", window_shape=(3, 3), **kwargs
    ):
        """Average patterns in an EBSD scan with its neighbours within a
        window.

        The amount of averaging is specified by the window coefficients.
        All patterns are averaged with the same window. Map borders are
        extended with zeros. The method operates inplace.

        Averaging is accomplished by correlating the window with the
        extended array of patterns using :func:`scipy.ndimage.correlate`.

        Parameters
        ----------
        window : kikuchipy.util.window.Window, "circular",\
                "rectangular", "gaussian", str, numpy.ndarray, or\
                dask.array.Array, optional
            Averaging window. Available types are listed in
            :func:`scipy.signal.windows.get_window`, in addition to a
            circular window (default) filled with ones in which corner
            coefficients are set to zero. A window element is
            considered to be in a corner if its radial distance to the
            origin (window centre) is shorter or equal to the half width
            of the window's longest axis. A 1D or 2D
            :class:`numpy.ndarray`, :class:`dask.array.Array` or
            :class:`~kikuchipy.util.window.Window` can also be passed.
        window_shape : sequence of ints, optional
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
        :class:`~kikuchipy.util.window.Window`
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

        >>> w = kp.util.Window(window="gaussian", std=2)
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

        if isinstance(window, kp.util.Window) and window.is_valid():
            averaging_window = copy.copy(window)
        else:
            averaging_window = kp.util.Window(
                window=window, shape=window_shape, **kwargs,
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
        dask_array = kp.util.dask._get_dask_array(signal=self)

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
            dask_array, depth=overlap_depth, boundary=overlap_boundary,
        )

        # Must also be overlapped, since the patterns are overlapped
        overlapped_window_sums = da.overlap.overlap(
            window_sums, depth=overlap_depth, boundary=overlap_boundary
        )

        # Finally, average patterns by correlation with the window and
        # subsequent division by the number of neighbours correlated with
        dtype_out = self.data.dtype
        overlapped_averaged_patterns = da.map_blocks(
            kp.util.chunk.average_neighbour_patterns,
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

    def virtual_backscatter_electron_imaging(self, roi, **kwargs):
        """Plot an interactive virtual backscatter electron (VBSE)
        image formed from intensities within a specified and adjustable
        region of interest (ROI) on the detector.

        Adapted from
        :meth:`pyxem.signals.diffraction2d.Diffraction2D.plot_interactive_virtual_image`.

        Parameters
        ----------
        roi : hyperspy.roi.BaseInteractiveROI
            Any interactive ROI detailed in HyperSpy.
        **kwargs:
            Keyword arguments to be passed to
            :meth:`hyperspy.signal.BaseSignal.plot`.

        Examples
        --------
        >>> import hyperspy.api as hs
        >>> roi = hs.roi.RectangularROI(
        ...     left=0, right=5, top=0, bottom=5)
        >>> s.virtual_backscatter_electron_imaging(roi)

        """

        return Diffraction2D.plot_interactive_virtual_image(self, roi, **kwargs)

    def get_virtual_image(self, roi):
        """Get a virtual backscatter electron (VBSE) image formed from
        intensities within a region of interest (ROI) on the detector.

        Adapted from
        :meth:`pyxem.signals.diffraction2d.Diffraction2D.get_virtual_image`.

        Parameters
        ----------
        roi : hyperspy.roi.BaseInteractiveROI
            Any interactive ROI detailed in HyperSpy.

        Returns
        -------
        virtual_image : hyperspy.signal.BaseSignal
            VBSE image formed from detector intensities within an ROI
            on the detector.

        Examples
        --------
        >>> import hyperspy.api as hs
        >>> roi = hs.roi.RectangularROI(
        ...     left=0, right=5, top=0, bottom=5)
        >>> vbse_image = s.get_virtual_image(roi)

        """

        return Diffraction2D.get_virtual_image(self, roi)

    def save(self, filename=None, overwrite=None, extension=None, **kwargs):
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
        filename : str or None, optional
            If ``None`` (default) and ``tmp_parameters.filename`` and
            ``tmp_parameters.folder`` in signal metadata are defined,
            the filename and path will be taken from there. A valid
            extension can be provided e.g. "data.h5", see ``extension``.
        overwrite : None or bool, optional
            If ``None`` and the file exists, it will query the user. If
            ``True`` (``False``) it (does not) overwrite the file if it
            exists.
        extension : None, 'h5', 'hdf5', 'h5ebsd', 'dat' or 'hspy',\
                optional
            Extension of the file that defines the file format. 'h5',
            'hdf5' and 'h5ebsd' are equivalent. If ``None``, the
            extension is determined from the following list in this
            order: i) the filename, ii) ``tmp_parameters.extension`` or
            iii) 'h5' (kikuchipy's h5ebsd format).
        **kwargs :
            Keyword arguments passed to writer.

        See Also
        --------
        kikuchipy.io.plugins.h5ebsd.file_writer,\
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
            basename, ext = os.path.splitext(filename)
            filename = basename + "." + extension
        save(filename, self, overwrite=overwrite, **kwargs)

    def get_decomposition_model(self, components=None, dtype_out=np.float32):
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
        components : None, int or list of ints, optional
            If None (default), rebuilds the signal from all components.
            If int, rebuilds signal from `components` in range 0-given
            int. If list of ints, rebuilds signal from only `components`
            in given list.
        dtype_out : numpy.float16, numpy.float32, numpy.float64,\
                optional
            Data type to cast learning results to (default is
            :class:`numpy.float32`). Note that HyperSpy casts them to
            :class:`numpy.float64`.

        Returns
        -------
        s_model : kikuchipy.signals.ebsd.EBSD or \
                kikuchipy.signals.ebsd.LazyEBSD

        """

        # Keep original results to revert back after updating
        factors_orig = self.learning_results.factors.copy()
        loadings_orig = self.learning_results.loadings.copy()

        # Change data type, keep desired components and rechunk if lazy
        (
            self.learning_results.factors,
            self.learning_results.loadings,
        ) = kp.util.decomposition._update_learning_results(
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
        ebsd_node = kp.util.io.metadata_nodes(sem=False)
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
        original_binning = md.get_item(ebsd_node + ".binning")
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
        components=None,
        dtype_learn=np.float32,
        mbytes_chunk=100,
        dir_out=None,
        fname_out=None,
    ):
        """Write the model signal generated from the selected number of
        principal components directly to an .hspy file.

        The model signal intensities are rescaled to the original
        signals' data type range, keeping relative intensities.

        Parameters
        ----------
        components : None, int or list of ints, optional
            If None (default), rebuilds the signal from all
            `components`. If int, rebuilds signal from `components` in
            range 0-given int. If list of ints, rebuilds signal from
            only `components` in given list.
        dtype_learn : numpy.float16, numpy.float32, or numpy.float64,\
                optional
            Data type to set learning results to (default is
            :class:`numpy.float32`) before multiplication.
        mbytes_chunk : int, optional
            Size of learning results chunks in MB, default is 100 MB as
            suggested in the Dask documentation.
        dir_out : str, optional
            Directory to place output signal in.
        fname_out : str, optional
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
        factors, loadings = kp.util.decomposition._update_learning_results(
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
            chunks = kp.util.dask._rechunk_learning_results(
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
