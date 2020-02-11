# -*- coding: utf-8 -*-
# Copyright 2019-2020 The KikuchiPy developers
#
# This file is part of KikuchiPy.
#
# KikuchiPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# KikuchiPy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with KikuchiPy. If not, see <http://www.gnu.org/licenses/>.

import datetime
import gc
import logging
import numbers
import os
import sys

import dask.array as da
import dask.diagnostics as dd
from hyperspy._signals.signal2d import Signal2D
from hyperspy._lazy_signals import LazySignal2D
from hyperspy.learn.mva import LearningResults
from hyperspy.misc.utils import DictionaryTreeBrowser
from h5py import File
import numpy as np
from pyxem.signals.diffraction2d import Diffraction2D

from kikuchipy.io._io import save
import kikuchipy as kp

_logger = logging.getLogger(__name__)


class EBSD(Signal2D):
    _signal_type = "EBSD"
    _alias_signal_types = ["electron_backscatter_diffraction"]
    _lazy = False

    def __init__(self, *args, **kwargs):
        """Create an :class:`~kikuchipy.signals.ebsd.EBSD` object from a
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
        """Set experimental parameters in signal ``metadata``.

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
        static_background : :class:`numpy.ndarray`, optional
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
        >>> print(s.metadata.get_item(ebsd_node + '.xpc')
        1.0
        >>> s.set_experimental_parameters(xpc=0.50726)
        >>> print(s.metadata.get_item(ebsd_node + '.xpc'))
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
        """Set parameters for one phase in signal ``metadata``, using
        the International Tables for Crystallography, Volume A.

        A phase node with default values is created if none is present
        in the ``metadata`` when this method is called.

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
        lattice_constants : :class:`numpy.ndarray` or list of\
                floats, optional
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
        >>> print(s.metadata.Sample.Phases.Number_1.atom_coordinates.
                Number_1)
        ├── atom =
        ├── coordinates = array([0., 0., 0.])
        ├── debye_waller_factor = 0.0
        └── site_occupation = 0.0
        >>> s.set_phase_parameters(
                number=1, atom_coordinates={
                    '1': {'atom': 'Ni', 'coordinates': [0, 0, 0],
                    'site_occupation': 1,
                    'debye_waller_factor': 0.0035}})
        >>> print(s.metadata.Sample.Phases.Number_1.atom_coordinates.
                Number_1)
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
        """Set the step size in um.

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
        >>> print(s.axes_manager.['x'].scale)  # Default value
        1.0
        >>> s.set_scan_calibration(step_x=1.5)  # Microns
        >>> print(s.axes_manager['x'].scale)
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
        >>> print(s.axes_manager['dx'].scale)  # Default value
        1.0
        >>> s.set_detector_calibration(delta=70.)
        >>> print(s.axes_manager['dx'].scale)
        70.0
        """

        centre = np.array(self.axes_manager.signal_shape) / 2 * delta
        dx, dy = self.axes_manager.signal_axes
        dx.units, dy.units = ["\u03BC" + "m"] * 2
        dx.scale, dy.scale = (delta, delta)
        dx.offset, dy.offset = -centre

    def static_background_correction(
        self, operation="subtract", relative=False, static_bg=None
    ):
        """Correct static background inplace by subtracting/dividing by
        a static background pattern.

        Resulting pattern intensities are rescaled keeping relative
        intensities or not and stretched to fill the available grey
        levels in the patterns' :class:`numpy.dtype` range.

        Parameters
        ----------
        operation : 'subtract' or 'divide', optional
            Subtract (default) or divide by static background pattern.
        relative : bool, optional
            Keep relative intensities between patterns (default is
            ``False``).
        static_bg : :class:`numpy.ndarray`,\
                :class:`dask.array.Array` or None, optional
            Static background pattern. If not passed we try to read it
            from the signal metadata.

        See Also
        --------
        dynamic_background_correction

        Examples
        --------
        Assuming that a static background pattern with same shape and
        data type (e.g. 8-bit unsigned integer, ``uint8``) as patterns
        is available in signal metadata:

        >>> import kikuchipy as kp
        >>> ebsd_node = kp.util.io.metadata_nodes(sem=False)
        >>> print(s.metadata.get_item(ebsd_node + '.static_background'))
        [[84 87 90 ... 27 29 30]
        [87 90 93 ... 27 28 30]
        [92 94 97 ... 39 28 29]
        ...
        [80 82 84 ... 36 30 26]
        [79 80 82 ... 28 26 26]
        [76 78 80 ... 26 26 25]]

        Static background can be corrected by subtracting or dividing
        this background from each pattern while keeping relative
        intensities between patterns (or not).

        >>> s.static_background_correction(
                operation='subtract', relative=True)

        If metadata has no background pattern, this must be passed in
        the ``static_bg`` parameter as a numpy or dask array.
        """

        dtype_out = self.data.dtype.type

        # Set up background pattern
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
                    "Static background is not a numpy or dask array or could "
                    "not be read from signal metadata."
                )
        if dtype_out != static_bg.dtype:
            raise ValueError(
                f"Static background dtype_out {static_bg.dtype} is not the "
                f"same as pattern dtype_out {dtype_out}."
            )
        pat_shape = self.axes_manager.signal_shape[::-1]
        bg_shape = static_bg.shape
        if bg_shape != pat_shape:
            raise OSError(
                f"Pattern {pat_shape} and static background {bg_shape} shapes "
                "are not identical."
            )
        dtype = np.float32
        static_bg = static_bg.astype(dtype)

        # Get min./max. input patterns intensity after correction
        if relative:  # Scale relative to min./max. intensity in scan
            signal_min = self.data.min(axis=(0, 1))
            signal_max = self.data.max(axis=(0, 1))
            if operation == "subtract":
                imin = (signal_min - static_bg).astype(dtype).min()
                imax = (signal_max - static_bg).astype(dtype).max()
            else:  # Divide
                imin = (signal_min / static_bg).astype(dtype).min()
                imax = (signal_max / static_bg).astype(dtype).max()
            in_range = (imin, imax)
        else:  # Scale relative to min./max. intensity in each pattern
            in_range = None

        # Create dask array of signal patterns and do processing on this
        dask_array = kp.util.dask._get_dask_array(signal=self, dtype=dtype)

        # Correct static background and rescale intensities chunk by chunk
        corrected_patterns = dask_array.map_blocks(
            kp.util.experimental._static_background_correction_chunk,
            static_bg=static_bg,
            operation=operation,
            in_range=in_range,
            dtype_out=dtype_out,
            dtype=dtype_out,
        )

        # Overwrite signal patterns
        if not self._lazy:
            with dd.ProgressBar():
                print("Static background correction:", file=sys.stdout)
                corrected_patterns.store(self.data, compute=True)
        else:
            self.data = corrected_patterns

    def dynamic_background_correction(self, operation="subtract", sigma=None):
        """Correct dynamic background inplace by subtracting or dividing
        by a blurred version of each pattern.

        Resulting pattern intensities are rescaled to fill the
        available grey levels in the patterns' :class:`numpy.dtype`
        range.

        Parameters
        ----------
        operation : 'subtract' or 'divide', optional
            Subtract (default) or divide by dynamic background pattern.
        sigma : int, float or None, optional
            Standard deviation of the gaussian kernel. If None
            (default), a deviation of pattern width/30 is chosen.

        See Also
        --------
        static_background_correction

        Examples
        --------
        Traditional background correction includes static and dynamic
        corrections, loosing relative intensities between patterns after
        dynamic corrections (whether ``relative`` is set to ``True`` or
        ``False`` in :meth:`~static_background_correction`):

        >>> s.static_background_correction(operation='subtract')
        >>> s.dynamic_background_correction(
                operation='subtract', sigma=2.0)
        """

        dtype_out = self.data.dtype.type
        dtype = np.float32

        # Create dask array of signal patterns and do processing on this
        dask_array = kp.util.dask._get_dask_array(signal=self, dtype=dtype)

        if sigma is None:
            sigma = self.axes_manager.signal_axes[0].size / 30

        corrected_patterns = dask_array.map_blocks(
            kp.util.experimental._dynamic_background_correction_chunk,
            operation=operation,
            sigma=sigma,
            dtype_out=dtype_out,
            dtype=dtype_out,
        )

        # Overwrite signal patterns
        if not self._lazy:
            with dd.ProgressBar():
                print("Dynamic background correction:", file=sys.stdout)
                corrected_patterns.store(self.data, compute=True)
        else:
            self.data = corrected_patterns

    def rescale_intensities(self, relative=False, dtype_out=None):
        """Rescale pattern intensities inplace to desired
        :class:`numpy.dtype` range specified by ``dtype_out`` keeping
        relative intensities or not.

        This method makes use of
        :func:`skimage.exposure.rescale_intensity`.

        Parameters
        ----------
        relative : bool, optional
            Keep relative intensities between patterns, default is
            ``False``.
        dtype_out : numpy.dtype, optional
            Data type of rescaled patterns, default is input patterns'
            data type.

        See Also
        --------
        adaptive_histogram_equalization

        Examples
        --------
        Pattern intensities are stretched to fill the available grey
        levels in the input patterns' data type range or any
        :class:`numpy.dtype` range passed to ``dtype_out``, either
        keeping relative intensities between patterns or not:

        >>> print(s.data.dtype_out, s.data.min(), s.data.max(),
                  s.inav[0, 0].data.min(), s.inav[0, 0].data.max())
        uint8 20 254 24 233
        >>> s2 = s.deepcopy()
        >>> s.rescale_intensities(dtype_out=np.uint16)
        >>> print(s.data.dtype_out, s.data.min(), s.data.max(),
                  s.inav[0, 0].data.min(), s.inav[0, 0].data.max())
        uint16 0 65535 0 65535
        >>> s2.rescale_intensities(relative=True)
        >>> print(s2.data.dtype_out, s2.data.min(), s2.data.max(),
                  s2.inav[0, 0].data.min(), s2.inav[0, 0].data.max())
        uint8 0 255 4 232
        """

        if dtype_out is None:
            dtype_out = self.data.dtype.type

        # Determine min./max. intensity of input pattern to rescale to
        if relative:  # Scale relative to min./max. intensity in scan
            in_range = (self.data.min(), self.data.max())
        else:  # Scale relative to min./max. intensity in each pattern
            in_range = None

        # Create dask array of signal patterns and do processing on this
        dask_array = kp.util.dask._get_dask_array(signal=self)

        # Rescale patterns
        rescaled_patterns = dask_array.map_blocks(
            kp.util.experimental._rescale_pattern_chunk,
            in_range=in_range,
            dtype_out=dtype_out,
            dtype=dtype_out,
        )

        # Overwrite signal patterns
        if not self._lazy:
            with dd.ProgressBar():
                if self.data.dtype != rescaled_patterns.dtype:
                    self.data = self.data.astype(dtype_out)
                print("Rescaling patterns:", file=sys.stdout)
                rescaled_patterns.store(self.data, compute=True)
        else:
            self.data = rescaled_patterns

    def adaptive_histogram_equalization(
        self, kernel_size=None, clip_limit=0, nbins=128
    ):
        """Local contrast enhancement inplace with adaptive histogram
        equalization.

        This method makes use of
        :func:`skimage.exposure.equalize_adapthist`.

        Parameters
        ----------
        kernel_size : int or list-like, optional
            Shape of contextual regions for adaptive histogram
            equalization, default is 1/4 of pattern height and 1/4 of
            pattern width.
        clip_limit : float, optional
            Clipping limit, normalized between 0 and 1 (higher values
            give more contrast). Default is 0.
        nbins : int, optional
            Number of gray bins for histogram ("data range"), default is
            128.

        See also
        --------
        dynamic_background_correction, rescale_intensities,
        static_background_correction

        Examples
        --------
        To best understand how adaptive histogram equalization works,
        we plot the histogram of the same pattern before and after
        equalization:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> s2 = s.inav[0, 0]
        >>> s2.adaptive_histogram_equalization()
        >>> imin = np.iinfo(s.data.dtype_out).min
        >>> imax = np.iinfo(s.data.dtype_out).max + 1
        >>> hist, _ = np.histogram(s.inav[0, 0].data, bins=imax,
                          range=(imin, imax))
        >>> hist2, _ = np.histogram(s2.inav[0, 0].data, bins=imax,
                           range=(imin, imax))
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
        * The default kernel size might not fit all pattern sizes, so it
          may be necessary to search for the optimal kernel size.
        """

        # Determine kernel size (shape of contextual region)
        sig_shape = self.axes_manager.signal_shape
        if kernel_size is None:
            kernel_size = (sig_shape[0] // 4, sig_shape[1] // 4)
        elif isinstance(kernel_size, numbers.Number):
            kernel_size = (kernel_size,) * self.axes_manager.signal_dimension
        elif len(kernel_size) != self.axes_manager.signal_dimension:
            raise ValueError(f"Incorrect value of `kernel_size`: {kernel_size}")
        kernel_size = [int(k) for k in kernel_size]

        # Create dask array of signal patterns and do processing on this
        dask_array = kp.util.dask._get_dask_array(signal=self)

        # Local contrast enhancement
        equalized_patterns = dask_array.map_blocks(
            kp.util.experimental._adaptive_histogram_equalization_chunk,
            kernel_size=kernel_size,
            clip_limit=clip_limit,
            nbins=nbins,
            dtype=self.data.dtype,
        )

        # Overwrite signal patterns
        if not self._lazy:
            with dd.ProgressBar():
                print("Adaptive histogram equalization:", file=sys.stdout)
                equalized_patterns.store(self.data, compute=True)
        else:
            self.data = equalized_patterns

    def virtual_backscatter_electron_imaging(self, roi, **kwargs):
        """Plot an interactive virtual backscatter electron (VBSE)
        image formed from detector intensities within a specified and
        adjustable region of interest (ROI).

        Adapted from
        meth:`pyxem.signals.diffraction2d.Diffraction2D.plot_interactive_virtual_image`.

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
                left=0, right=5, top=0, bottom=5)
        >>> s.virtual_backscatter_electron_imaging(roi)
        """

        return Diffraction2D.plot_interactive_virtual_image(self, roi, **kwargs)

    def get_virtual_image(self, roi):
        """Return a virtual backscatter electron (VBSE) image
        formed from detector intensities within a region of interest
        (ROI) on the detector.

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
                left=0, right=5, top=0, bottom=5)
        >>> vbse_image = s.get_virtual_image(roi)
        """

        return Diffraction2D.get_virtual_image(self, roi)

    def save(self, filename=None, overwrite=None, extension=None, **kwargs):
        """Write signal to the specified format.

        The function gets the format from the extension: `h5`, `hdf5` or
        `h5ebsd` for KikuchiPy's specification of the the h5ebsd
        format, `dat` for the NORDIF binary format or `hspy` for
        HyperSpy's HDF5 specification. If no extension is provided the
        signal is written to a file in KikuchiPy's h5ebsd format. Each
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
            iii) 'h5' (KikuchiPy's h5ebsd format).
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

    def decomposition(self, *args, **kwargs):
        super().decomposition(*args, **kwargs)
        self.__class__ = EBSD

    def get_decomposition_model(self, components=None, dtype_out=np.float16):
        """Return the model signal generated with the selected number of
        principal components from a decomposition.

        Calls HyperSpy's
        :meth:`hyperspy.learn.mva.MVA.get_decomposition_model`.
        Learning results are preconditioned before this call, doing the
        following: (1) set :class:`numpy.dtype` to desired
        ``dtype_out``, (2) remove unwanted components, and (3) rechunk,
        if :class:`dask.array.Array`, to suitable chunks.

        Parameters
        ----------
        components : None, int or list of ints, optional
            If ``None`` (default), rebuilds the signal from all
            components. If ``int``, rebuilds signal from ``components``
            in range 0-given ``int``. If list of ``ints``, rebuilds
            signal from only ``components`` in given list.
        dtype_out : numpy.float16, numpy.float32, numpy.float64,\
                optional
            Data to cast learning results to (default is
            :class:`numpy.float16`). Note that HyperSpy casts them
            to :class:`numpy.float64`.

        Returns
        -------
        s_model : :class:`~kikuchipy.signals.ebsd.EBSD` or \
                :class:`~kikuchipy.signals.ebsd.LazyEBSD`
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

        # Revert class
        assign_class = EBSD
        if self._lazy:
            assign_class = LazyEBSD
        self.__class__ = assign_class
        s_model.__class__ = assign_class

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

    def as_lazy(self, *args, **kwargs):
        """Create a :class:`~kikuchipy.signals.ebsd.LazyEBSD` object
        from an :class:`~kikuchipy.signals.ebsd.EBSD` object.

        Returns
        -------
        lazy_signal : :class:`~kikuchipy.signals.ebsd.LazyEBSD`
            Lazy signal.
        """

        lazy_signal = super().as_lazy(*args, **kwargs)
        lazy_signal.__class__ = LazyEBSD
        lazy_signal.__init__(**lazy_signal._to_dictionary())
        return lazy_signal

    def change_dtype(self, dtype, rechunk=True):
        super().change_dtype(dtype=dtype, rechunk=rechunk)
        self.__class__ = EBSD


class LazyEBSD(EBSD, LazySignal2D):

    _lazy = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def change_dtype(self, dtype, rechunk=True):
        super().change_dtype(dtype=dtype, rechunk=rechunk)
        self.__class__ = LazyEBSD

    def compute(self, *args, **kwargs):
        with dd.ProgressBar(*args, **kwargs):
            self.data = self.data.compute(*args, **kwargs)
        gc.collect()
        self.__class__ = EBSD
        self._lazy = False

    def decomposition(self, *args, **kwargs):
        super().decomposition(*args, **kwargs)
        self.__class__ = LazyEBSD

    def get_decomposition_model_write(
        self,
        components=None,
        dtype_learn=np.float16,
        mbytes_chunk=100,
        dir_out=None,
        fname_out=None,
    ):
        """Write the model signal generated from the selected number of
        principal components directly to a .hspy file.

        The model signal intensities are rescaled to the original
        signals' data type range, keeping relative intensities.

        Parameters
        ----------
        components : None, int or list of ints, optional
            If ``None`` (default), rebuilds the signal from all
            ``components``. If ``int``, rebuilds signal from
            ``components`` in range 0-given ``int``. If list of ints,
            rebuilds signal from only ``components`` in given list.
        dtype_learn : :class:`numpy.float16`,\
                :class:`numpy.float32` or :class:`numpy.float64`,\
                optional
            Data type to set learning results to (default is
            :class:`numpy.float16`) before multiplication.
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
            s_model.rescale_intensities(
                dtype_out=self.data.dtype, relative=True
            )

            # Write signal to file
            if fname_out is None:
                fname_out = "model_" + t_str
            file_model = os.path.join(dir_out, fname_out)
            s_model.save(file_model)

        # Delete temporary files
        os.remove(file_learn)
        gc.collect()  # Don't sink
