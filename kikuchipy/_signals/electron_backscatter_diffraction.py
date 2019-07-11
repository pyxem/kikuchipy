# -*- coding: utf-8 -*-
# Copyright 2019 The KikuchiPy developers
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

"""Signal class for Electron Backscatter Diffraction (EBSD) data."""
import warnings
import numpy as np
import dask.array as da
import os
import gc
import datetime
import tqdm
import numbers
from h5py import File
from hyperspy.signals import Signal2D
from hyperspy._lazy_signals import LazySignal2D
from hyperspy.learn.mva import LearningResults
from skimage.transform import radon
from pyxem.signals.electron_diffraction import ElectronDiffraction
from dask.diagnostics import ProgressBar
from hyperspy.misc.utils import dummy_context_manager
from kikuchipy import io
from kikuchipy._signals.radon_transform import RadonTransform
from kikuchipy.utils.expt_utils import (correct_background, remove_dead,
                                        find_deadpixels_single_pattern,
                                        plot_markers_single_pattern,
                                        rescale_pattern_intensity,
                                        equalize_adapthist_pattern)
from kikuchipy.utils.io_utils import metadata_nodes


class EBSD(Signal2D):
    _signal_type = 'EBSD'
    _alias_signal_types = ['electron_backscatter_diffraction']
    _lazy = False

    def __init__(self, *args, **kwargs):
        if self._lazy and args:
            Signal2D.__init__(self, data=args[0], **kwargs)
        else:
            Signal2D.__init__(self, *args, **kwargs)
        self.set_experimental_parameters(deadpixels_corrected=False)

    def set_experimental_parameters(self, detector=None,
                                    azimuth_angle=None,
                                    elevation_angle=None,
                                    sample_tilt=None,
                                    working_distance=None, binning=None,
                                    detector_pixel_size=None,
                                    exposure_time=None, grid_type=None,
                                    step_x=None, step_y=None, gain=None,
                                    frame_number=None, frame_rate=None,
                                    scan_time=None, beam_energy=None,
                                    xpc=None, ypc=None, zpc=None,
                                    deadpixels_corrected=None,
                                    deadvalue=None, deadpixels=None,
                                    deadthreshold=None, static_background=None,
                                    manufacturer=None, version=None,
                                    microscope=None, magnification=None):
        """Set experimental parameters in metadata.

        Parameters
        ----------
        azimuth_angle : float, optional
            Azimuth angle of the detector in degrees. If the azimuth is
            zero, the detector is perpendicular to the tilt axis.
        beam_energy : float, optional
            Energy of the electron beam in kV.
        binning : int, optional
            Camera binning.
        deadpixels_corrected : bool, optional
            Whether deadpixels in patterns have been corrected.
        deadpixels : list of tuple, optional
            Pattern indices for dead pixels.
        deadvalue : string, optional
            How dead pixels have been corrected for (average or nan).
        deadthreshold : int, optional
            Threshold for detecting dead pixels.
        detector : str, optional
            Detector manufacturer and model.
        detector_pixel_size : float, optional
            Camera pixel size on the scintillator surface in µm.
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
            Camera gain in dB.
        grid_type : str, optional
            Scan grid type, only square grid is supported.
        manufacturer : str, optional
            Manufacturer used to collect patterns.
        microscope : str, optional
            Microscope used to collect patterns.
        magnification : int, optional
            Microscope magnification at which patterns were collected.
        sample_tilt : float, optional
            Sample tilt angle from horizontal in degrees.
        scan_time : float, optional
            Scan time in s.
        static_background : np.ndarray, optional
            Static background pattern.
        step_x : float, optional
            Scan step size in fast scan direction (east).
        step_y : float, optional
            Scan step size in slow scan direction (south).
        version : str, optional
            Version of manufacturer software used to collect patterns.
        working_distance : float, optional
            Working distance in mm.
        xpc : float, optional
            Pattern centre horizontal coordinate with respect to
            detector centre.
        ypc : float, optional
            Pattern centre horizontal coordinate with respect to
            detector centre.
        zpc : float, optional
            Specimen to scintillator distance.
        """
        def _write_params(params, metadata, node):
            for key, val in params.items():
                if val is not None:
                    metadata.set_item(node + '.' + key, val)
        md = self.metadata
        omd = self.original_metadata
        sem_node, ebsd_node = metadata_nodes()
        _write_params({'beam_energy': beam_energy,
                       'magnification': magnification, 'microscope': microscope,
                       'working_distance': working_distance}, md, sem_node)
        _write_params({'azimuth_angle': azimuth_angle,
                       'binning': binning, 'detector': detector,
                       'detector_pixel_size': detector_pixel_size,
                       'elevation_angle': elevation_angle,
                       'exposure_time': exposure_time,
                       'frame_number': frame_number,
                       'frame_rate': frame_rate, 'gain': gain,
                       'grid_type': grid_type,
                       'manufacturer': manufacturer, 'version': version,
                       'sample_tilt': sample_tilt, 'scan_time': scan_time,
                       'step_x': step_x, 'step_y': step_y,
                       'xpc': xpc, 'ypc': ypc, 'zpc': zpc,
                       'static_background': static_background}, md, ebsd_node)
        _write_params({'deadpixels_corrected': deadpixels_corrected,
                       'deadpixels': deadpixels, 'deadvalue': deadvalue,
                       'deadthreshold': deadthreshold}, omd, ebsd_node)

    def set_scan_calibration(self, calibration):
        """Set the step size in µm.

        Parameters
        ----------
        calibration : float
            Scan step size in µm per pixel.
        """
        ElectronDiffraction.set_scan_calibration(self, calibration)
        self.axes_manager.navigation_axes[0].units = u'\u03BC'+'m'
        self.axes_manager.navigation_axes[1].units = u'\u03BC'+'m'

    def set_diffraction_calibration(self, calibration):
        """Set diffraction pattern pixel size in reciprocal Angstroms.
        The offset is set to 0 for signal_axes[0] and signal_axes[1].

        Parameters
        ----------
        calibration : float
            Diffraction pattern calibration in reciprocal Angstroms per
            pixel.
        """
        ElectronDiffraction.set_diffraction_calibration(self, calibration)
        self.axes_manager.signal_axes[0].offset = 0
        self.axes_manager.signal_axes[1].offset = 0

    def remove_background(self, static=True, dynamic=True, static_bg=None,
                          relative=False, sigma=None, **kwargs):
        """Background correction, either static, dynamic or both.

        For the static correction, a background image is subtracted
        from all patterns. For the dynamic correction, each pattern is
        blurred using a Gaussian kernel with a standard deviation set
        by you.

        Contrast stretching is done either according to a global or a
        local intensity range, the former maintaining relative
        intensities between patterns after static correction. Relative
        intensities are lost if only dynamic correction is performed.

        Input data is assumed to be a two-dimensional numpy array of
        patterns of dtype uint8.

        Parameters
        ----------
        static : bool, optional
            If True (default), static correction is performed.
        dynamic : bool, optional
            If True (default), dynamic correction is performed.
        static_bg : {str, None}, optional
            File path to static background pattern. If None (default),
            an attempt to read the pattern from the signal metadata is
            made.
        relative : bool, optional
            If True (default is False), relative intensities between
            patterns are kept after static correction.
        sigma : {int, float, None}, optional
            Standard deviation for the gaussian kernel for dynamic
            correction. If None (default), a deviation of pattern
            width/30 is chosen.
        **kwargs:
            Keyword arguments passed to map().
        """
        if static is False and dynamic is False:
            warnings.warn("No correction done.")
            return

        lazy = self._lazy
        if lazy:
            kwargs['ragged'] = False

        # Set default values for contrast stretching parameters, to be
        # overwritten if 'relative' is passed
        imin = None
        scale = None

        ebsd_node = metadata_nodes(sem=False)
        md = self.metadata
        if static:
            if static_bg is None:
                static_bg = md.get_item(ebsd_node + '.static_background')
                if not isinstance(static_bg, int) and static_bg.all() == -1:
                    raise ValueError("No static background pattern provided.")

            # Read and setup background
            static_bg = Signal2D(static_bg)

            # Correct dead pixels in static background pattern if they are
            # corrected in experimental patterns
            omd_ebsd = self.original_metadata.get_item(ebsd_node)
            if (omd_ebsd.has_item('deadpixels_corrected') and
                    omd_ebsd.has_item('deadpixels') and
                    omd_ebsd.has_item('deadvalue')):
                static_bg.data = remove_dead(static_bg.data,
                                             omd_ebsd.deadpixels,
                                             omd_ebsd.deadvalue)

            if relative and not dynamic:
                # Get lowest intensity after subtraction
                smin = self.min(self.axes_manager.navigation_axes)
                smax = self.max(self.axes_manager.navigation_axes)
                if lazy:
                    smin.compute()
                    smax.compute()
                smin.change_dtype(np.int8)
                static_bg.change_dtype(np.int8)
                imin = (smin.data - static_bg.data).min()

                # Get highest intensity after subtraction
                static_bg.data = static_bg.data.astype(np.uint8)
                imax = (smax.data - static_bg.data).max() + abs(imin)

                # Get global scaling factor, input dtype max. value in nominator
                scale = float(np.iinfo(self.data.dtype).max / imax)

        if dynamic and sigma is None:
            sigma = int(self.axes_manager.signal_axes[0].size/30)

        self.map(correct_background, static=static, dynamic=dynamic,
                 bg=static_bg, sigma=sigma, imin=imin, scale=scale, **kwargs)

    def equalize_adapthist(self, kernel_size=None, clip_limit=0.01, nbins=256,
                           **kwargs):
        """Local contrast enhancement using contrast limited adaptive
        histogram equalisation (CLAHE).

        Input data is assumed to be a two-dimensional numpy array of
        patterns of dtype uint8.

        Parameters
        ----------
        kernel_size : integer or list-like, optional
            Defines the shape of contextual regions used in the
            algorithm. By default, ``kernel_size`` is 1/8 of ``pattern``
            height by 1/8 of its width, or a minimum of 20 in either
            direction.
        clip_limit : float, optional
            Clipping limit, normalised between 0 and 1 (higher values
            give more contrast).
        nbins : int, optional
            Number of gray bins for histogram ("data range").
        **kwargs
            Arguments to be passed to map().

        Notes
        -----
        Adapted from scikit-image, without rescaling the pattern before
        equalisation and returning it with correct data type. See
        ``skimage.exposure.equalize_adapthist`` documentation for more
        details.
        """
        if self._lazy:
            kwargs['ragged'] = False

        # Set kernel size, ensuring it is at least 20 in each direction
        sdim = 2
        if kernel_size is None:
            sx, sy = self.axes_manager.signal_shape
            kernel_size = (sx // 8, sy // 8)
        elif isinstance(kernel_size, numbers.Number):
            kernel_size = (kernel_size,) * sdim
        elif len(kernel_size) != sdim:
            ValueError(
                "Incorrect value of `kernel_size`: {}".format(kernel_size))
        kernel_size = [int(k) for k in kernel_size]
        kernel_size = [20 if i < 20 else i for i in kernel_size]

        self.map(equalize_adapthist_pattern, kernel_size=kernel_size,
                 clip_limit=clip_limit, nbins=nbins, **kwargs)

    def find_deadpixels(self, pattern_number=10, threshold=2,
                        pattern_coordinates=None, plot=False,
                        mask=None, pattern_number_threshold=0.75):
        """Find dead pixels in several experimentally acquired
        diffraction patterns by comparing pixel values in a blurred
        version of a selected pattern with the original pattern. If the
        intensity difference is above a threshold the pixel is labeled
        as dead. Deadpixels are searched for in several patterns. Only
        deadpixels occurring in more than a certain number of patterns
        will be kept.

        Parameters
        ----------
        pattern_number : int, optional
            Number of patterns to find deadpixels in. If
            pattern_coordinates is passed, pattern_number is set to
            len(pattern_coordinates).
        threshold : int, optional
            Threshold for difference in pixel intensities between
            blurred and original pattern. The actual threshold is given
            as threshold*(standard deviation of the difference between
            blurred and original pattern).
        pattern_coordinates : np.array, optional
            Array of selected coordinates [[x,y]] for all the patterns
            where deadpixels will be searched for.
        plot : bool, optional
            If True (default is False), a pattern with the dead pixels
            marked is plotted.
        mask : array of bool, optional
            No deadpixels are found where mask is True. The shape must
            be equal to the signal shape.
        pattern_number_threshold : float, optional
            One deadpixel is only considered correct if it is found in a
            number of patterns that is more than
            pattern_number_threshold*pattern_number. Otherwise, the
            deadpixel is discarded.

        Returns
        -------
        deadpixels : list of tuples
            List of tuples containing pattern indices for dead pixels.

        Examples
        --------
        .. code-block:: python

            import numpy as np
            mask = np.zeros(s.axes_manager.signal_shape)
            # Threshold the first pattern, so that pixels with an
            # intensity below 60 will be masked
            mask[np.where(s.inav[0, 0].data < 60)] = True
            deadpixels = s.find_deadpixels(threshold=5, to_plot=True,
                                           mask=mask)
        """
        if pattern_coordinates is None:
            nav_shape = self.axes_manager.navigation_shape
            pattern_coordinates_x = np.random.randint(nav_shape[0],
                                                      size=pattern_number)
            pattern_coordinates_y = np.random.randint(nav_shape[1],
                                                      size=pattern_number)
            pattern_coordinates = np.array(
                list(zip(pattern_coordinates_x, pattern_coordinates_y)))
        else:
            pattern_number = len(pattern_coordinates)

        pattern_coordinates = pattern_coordinates.astype(np.int16)

        first_pattern = self.inav[pattern_coordinates[0]].data
        deadpixels_new = find_deadpixels_single_pattern(first_pattern,
                                                        threshold=threshold,
                                                        mask=mask)
        for coordinates in pattern_coordinates[1:]:
            pattern = self.inav[coordinates].data
            deadpixels_new = np.append(deadpixels_new,
                                       find_deadpixels_single_pattern(pattern,
                                                                      threshold=threshold, mask=mask),
                                       axis=0)
        # Count the number of occurrences of each deadpixel found in all
        # checked patterns.
        deadpixels_new, count_list = np.unique(deadpixels_new,
                                               return_counts=True, axis=0)
        # Only keep deadpixel if it occurs in more than an amount given by
        # pattern_number_threshold of the patterns. Otherwise discard.
        keep_list = [True if y > int(pattern_number_threshold * pattern_number)
                     else False for y in count_list]
        deadpixels = deadpixels_new[np.where(keep_list)]
        if plot:
            plot_markers_single_pattern(first_pattern, deadpixels)

        # Update original_metadata
        self.set_experimental_parameters(deadpixels=deadpixels)

        return deadpixels

    def remove_deadpixels(self, deadpixels=None, deadvalue='average',
                          inplace=True, *args, **kwargs):
        """Remove dead pixels from experimentally acquired diffraction
        patterns, either by averaging or setting to a certain value.

        Assumes signal has navigation axes, i.e. does not work on a
        single pattern.

        Uses pyXem's remove_deadpixels() function.

        Parameters
        ----------
        deadpixels : list of tuples, optional
            List of tuples of indices of dead pixels. If None (default),
            indices of dead pixels are read from
        deadvalue : string, optional
            Specify how deadpixels should be treated. 'average' sets
            the dead pixel value to the average of adjacent pixels.
            'nan'  sets the dead pixel to nan.
        inplace : bool, optional
            If True (default), signal is overwritten. Otherwise,
            returns a new signal.
        *args:
            Arguments to be passed to map().
        **kwargs:
            Keyword arguments to be passed to map().
        """
        if self._lazy:
            kwargs['ragged'] = False

        # Get detected dead pixels, if any
        if deadpixels is None:
            try:
                omd = self.original_metadata
                deadpixels = omd.get_item(
                    'Acquisition_instrument.SEM.Detector.EBSD.deadpixels')
            except ValueError:
                warnings.warn("No dead pixels provided.")

        if inplace:
            self.map(remove_dead, deadpixels=deadpixels, deadvalue=deadvalue,
                     inplace=inplace, *args, **kwargs)
            self.set_experimental_parameters(deadpixels_corrected=True,
                                             deadvalue=deadvalue)
        elif not inplace:
            s = self.map(remove_dead, deadpixels=deadpixels,
                         deadvalue=deadvalue, inplace=inplace, *args, **kwargs)
            s.set_experimental_parameters(deadpixels_corrected=True,
                                          deadvalue=deadvalue)
            return s
        else:  # No dead pixels detected
            pass

    def get_virtual_image(self, roi):
        """Method imported from
        pyXem.ElectronDiffraction.get_virtual_image(self, roi). Obtains
        a virtual image associated with a specified ROI.

        Parameters
        ----------
        roi: :obj:`hyperspy.roi.BaseInteractiveROI`
            Any interactive ROI detailed in HyperSpy.

        Returns
        -------
        dark_field_sum: :obj:`hyperspy._signals.BaseSignal`
            The virtual image signal associated with the specified roi.

        Examples
        --------
        .. code-block:: python

            import hyperspy.api as hs
            roi = hs.roi.RectangularROI(left=10, right=20, top=10,
                bottom=20)
            s.get_virtual_image(roi)
        """
        return ElectronDiffraction.get_virtual_image(self, roi)

    def plot_interactive_virtual_image(self, roi, **kwargs):
        """Method imported from
        pyXem.ElectronDiffraction.plot_interactive_virtual_image(self,
        roi). Plots an interactive virtual image formed with a
        specified and adjustable roi.

        Parameters
        ----------
        roi: :obj:`hyperspy.roi.BaseInteractiveROI`
            Any interactive ROI detailed in HyperSpy.
        **kwargs:
            Keyword arguments to be passed to `ElectronDiffraction.plot`

        Examples
        --------
        .. code-block:: python

            import hyperspy.api as hs
            roi = hs.roi.RectangularROI(left=10, right=20, top=10,
                bottom=20)
            s.plot_interactive_virtual_image(roi)
        """
        return ElectronDiffraction.plot_interactive_virtual_image(self, roi,
                                                                  **kwargs)

    def get_radon_transform(self, theta=None, circle=True,
                            show_progressbar=True, inplace=False):
        """Create a RadonTransform signal.

        Parameters
        ----------
        theta : numpy array, optional
            Projection angles in degrees. If None (default), the value
            is set to np.arange(180).
        circle : bool, optional
            If True (default), assume that the image is zero outside the
            inscribed circle. The width of each projection then becomes
            equal to the smallest signal shape.
        show_progressbar : bool, optional
            If True (default), show progressbar during transformation.
        inplace : bool, optional
            If True (default is False), the EBSD signal (self) is
            replaced by the RadonTransform signal (return).

        Returns
        -------
        sinograms: :obj:`kikuchipy.signals.RadonTransform`
            Corresponding RadonTransform signal (sinograms) computed
            from the EBSD signal. The rotation axis lie at index
            sinograms.data[0,0].shape[0]/2

        References
        ----------
        http://scikit-image.org/docs/dev/auto_examples/transform/
        plot_radon_transform.html
        http://scikit-image.org/docs/dev/api/skimage.transform.html
        #skimage.transform.radon
        """
        # TODO: Remove diagonal artifact lines.
        # TODO: Can we get it to work faster?

        warnings.filterwarnings("ignore", message="The default of `circle` \
        in `skimage.transform.radon` will change to `True` in version 0.15.")
        # Ignore this warning, since this function is mapped and would
        # otherwise slow down this function by printing many warning
        # messages.

        sinograms = self.map(radon, theta=theta, circle=circle,
                             show_progressbar=show_progressbar,
                             inplace=inplace)

        return RadonTransform(sinograms)

    def save(self, filename=None, overwrite=None, extension=None,
             **kwargs):
        """Save signal in the specified format.

        The function gets the format from the extension: `h5`, `hdf5` or
        `h5ebsd` for KikuchiPy's specification of the the h5ebsd format
        `dat` for the NORDIF binary format or `hspy` for HyperSpy's
        HDF5 specification. If no extension is provided the default
        file format as defined in the `preferences` is used. Please note
        that not all formats support saving datasets of arbitrary
        dimensions. Each format accepts a different set of parameters.

        For details see the specific format documentation in
        `kikuchipy.io_plugins.<format>.file_writer`.

        Parameters
        ----------
        filename : {str or None}, optional
            If None (default) and `tmp_parameters.filename` and
            `tmp_parameters.folder` are defined, the filename and path
            will be taken from there. A valid extension can be provided
            e.g. "data.h5", see `extension`.
        overwrite : {None, bool}, optional
            If None and the file exists, it will query the user. If
            True (False) it (does not) overwrite the file if it exists.
        extension : {None, 'h5', 'hdf5', 'h5ebsd', 'hspy', 'dat',
                     'png', 'tiff', etc.}, optional
            Extension of the file that defines the file format. 'h5',
            'hdf5' and 'h5ebsd' are equivalent. If None, the extension
            is determined from the following list in this order: i) the
            filename, ii)  `Signal.tmp_parameters.extension` or iii)
            `hspy` (HyperSpy's default extension)
        **kwargs :
            Keyword arguments passed to writer.
        """
        if filename is None:
            if (self.tmp_parameters.has_item('filename') and
                    self.tmp_parameters.has_item('folder')):
                filename = os.path.join(
                    self.tmp_parameters.folder,
                    self.tmp_parameters.filename)
                extension = (self.tmp_parameters.extension
                             if not extension
                             else extension)
            elif self.metadata.has_item('General.original_filename'):
                filename = self.metadata.General.original_filename
            else:
                raise ValueError("Filename not defined.")
        if extension is not None:
            basename, ext = os.path.splitext(filename)
            filename = basename + '.' + extension
        io.save(filename, self, overwrite=overwrite, **kwargs)

    def get_decomposition_model(self, components=None,
                                dtype_out=np.float16, *args, **kwargs):
        """Return the model signal generated with the selected number of
        principal components.

        This function calls HyperSpy's get_decomposition_model. The
        learning results are preconditioned before this call, doing the
        following: (1) set data type to desired dtype, (2) remove
        unwanted components, (3) rechunk, if dask arrays, to suitable
        chunk.

        Parameters
        ----------
        components : {None, int or list of ints}, optional
            If None (default), rebuilds the signal from all components.
            If int, rebuilds signal from components in range 0-given
            int. If list of ints, rebuilds signal from only components
            in given list.
        dtype_out : {np.float16, np.float32, np.float64}, optional
            Data type of learning results (default is float16).
            HyperSpy's ``decomposition`` returns them in float64, which
            here is assumed to be overkill.
        *args
            Passed to Hyperspy's `get_decomposition_model`.
        **kwargs
            Passed to Hyperspy's `get_decomposition_model`.

        Returns
        -------
        s_model : {kikuchipy.signals.EBSD, kikuchipy.signals.LazyEBSD}
        """
        # Change dtype
        target = self.learning_results
        factors_orig = target.factors.copy()  # Keep to revert target in the end
        loadings_orig = target.loadings.copy()
        factors = target.factors.astype(dtype_out)
        loadings = target.loadings.astype(dtype_out)

        # Extract relevant components
        if hasattr(components, '__iter__'):  # components is a list of ints
            # TODO: This should be implemented in HyperSpy
            factors = factors[:, components]
            loadings = loadings[:, components]
        else:  # components is an int
            factors = factors[:, :components]
            loadings = loadings[:, :components]

        # Update learning results
        self.learning_results.factors = factors
        self.learning_results.loadings = loadings

        # Rechunk
        if isinstance(factors, da.Array):
            chunks = self._rechunk_learning_results()
            self.learning_results.factors = factors.rechunk(chunks=chunks[0])
            self.learning_results.loadings = loadings.rechunk(chunks=chunks[1])

        # Call HyperSpy's function
        s_model = super(Signal2D, self).get_decomposition_model(*args, **kwargs)

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


class LazyEBSD(EBSD, LazySignal2D):

    _lazy = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self, progressbar=True, close_file=False):
        """Attempt to store the full signal in memory.

        Parameters
        ----------
        progressbar : bool, optional
        close_file: bool, optional
            If True, attempt to close the file associated with the dask
            array data if any. Note that closing the file will make all
            other associated lazy signals inoperative.
        """
        if progressbar:
            cm = ProgressBar
        else:
            cm = dummy_context_manager
        with cm():
            data = self.data
            data = data.compute()
            if close_file:
                self.close_file()
            self.data = data
        self._lazy = False
        self.__class__ = EBSD

        # Collect garbage (coming from where?)
        gc.collect()

    def get_decomposition_model_write(self, components=None,
                                      dtype_learn=np.float16,
                                      mbytes_chunk=100, out_dir=None,
                                      out_fname=None):
        """Write the model signal generated from the selected number of
        principal components directly to a .hspy file. The model signal
        intensities are rescaled to the original signals' data type
        range.

        Notes
        -----
        Multiplying the learning results' factors and loadings in memory
        to create the model signal cannot sometimes be done due to too
        large matrices. Here, instead, learning results are written to
        file, read into dask arrays and multiplied using dask's
        ``matmul``, out of core.

        Parameters
        ----------
        components : {None, int or list of ints}, optional
            If None (default), rebuilds the signal from all components.
            If int, rebuilds signal from components in range 0-given
            int. If list of ints, rebuilds signal from only components
            in given list.
        dtype_learn : {np.float16, np.float32, np.float64}, optional
            Data type to set learning results to (default is float16).
        mbytes_chunk : int, optional
            Size of learning results chunks in MB, default is 100 MB as
            suggested in the Dask documentation.
        out_dir : str, optional
            Directory to place output signal in.
        out_fname : str, optional
            Name of output signal file.
        """
        # Change dtype
        target = self.learning_results
        factors = np.array(target.factors, dtype=dtype_learn)
        loadings = np.array(target.loadings, dtype=dtype_learn)

        # Extract relevant components
        if hasattr(components, '__iter__'):  # components is a list of ints
            # TODO: This should be implemented in HyperSpy
            factors = factors[:, components]
            loadings = loadings[:, components]
        else:  # components is an int
            factors = factors[:, :components]
            loadings = loadings[:, :components]

        # Write learning results to HDF5 file
        if out_dir is None:
            try:
                out_dir = self.original_metadata.General.original_filepath
            except AttributeError:
                raise AttributeError("Output directory has to be specified")

        t_str = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
        file_learn = os.path.join(out_dir, 'learn_' + t_str + '.h5')
        with File(file_learn, 'w') as f:
            f.create_dataset(name='factors', data=factors)
            f.create_dataset(name='loadings', data=loadings)

        # Matrix multiplication
        with File(file_learn, 'r') as f:
            # Read learning results from HDF5 file
            chunks = self._rechunk_learning_results(mbytes_chunk=mbytes_chunk)
            factors = da.from_array(f['factors'], chunks=chunks[0])
            loadings = da.from_array(f['loadings'], chunks=chunks[1])

            # Perform the matrix multiplication
            loadings = loadings.T
            res = factors @ loadings
            res = res.T  # Transpose

            # Create new signal from multiplied matrix
            s_model = self.deepcopy()
            s_model.learning_results = LearningResults()
            s_model.data = res.reshape(s_model.data.shape)
            s_model.data = s_model.data.rechunk(chunks=(1, 1, -1, -1))

            # Rescale intensities and revert data type
            s_model.map(rescale_pattern_intensity, ragged=False)
            s_model.data = s_model.data.astype(self.data.dtype)

            # Write signal to file (rechunking saves a little time?)
            if out_fname is None:
                out_fname = 'model_' + t_str
            file_model = os.path.join(out_dir, out_fname)
            s_model.save(file_model)

        # Delete temporary files
        os.remove(file_learn)
        gc.collect()  # Don't sink

    def _rechunk_learning_results(self, mbytes_chunk=100):
        """Return suggested data chunks for learning results. It is
        assumed that the loadings are not transposed. The last axes of
        factors and loadings are not chunked. The aims in prioritised
        order:
            1. Split into at least as many chunks as available CPUs.
            2. Limit chunks to approx. input MB (`mbytes_chunk`).
            3. Keep first axis of factors (detector pixels).

        Parameters
        ----------
        mbytes_chunk : int, optional
            Size of chunks in MB, default is 100 MB as suggested in the
            Dask documentation.

        Returns
        -------
        List of two tuples
            The first/second tuple are suggested chunks to pass to
            ``dask.array.rechunk`` for factors/loadings, respectively.
        """
        target = self.learning_results
        if target.decomposition_algorithm is None:
            raise ValueError("No learning results were found.")

        # Get dask chunks
        tshape = target.factors.shape + target.loadings.shape

        # Make sure the last factors/loading axes have the same shapes
        # TODO: Should also handle the case where the first axes are the same
        if tshape[1] != tshape[3]:
            raise ValueError("The last dimensions in factors and loadings are "
                             "not the same.")

        # Determine maximum number of (strictly necessary) chunks
        suggested_size = mbytes_chunk * 2**20
        factors_size = target.factors.nbytes
        loadings_size = target.loadings.nbytes
        total_size = factors_size + loadings_size
        num_chunks = np.ceil(total_size / suggested_size)

        # Get chunk sizes
        cpus = os.cpu_count()
        if num_chunks <= cpus:  # Return approx. as many chunks as CPUs
            chunks = [(-1, -1), (int(tshape[2]/cpus), -1)]  # -1 = don't chunk
        elif factors_size <= suggested_size:  # Chunk first axis in loadings
            chunks = [(-1, -1), (int(tshape[2]/num_chunks), -1)]
        else:  # Chunk both first axes
            sizes = [factors_size, loadings_size]
            while (sizes[0] + sizes[1]) >= suggested_size:
                i = np.argmax(sizes)
                sizes[i] = np.floor(sizes[i] / 2)
            factors_chunks = int(np.ceil(factors_size/sizes[0]))
            loadings_chunks = int(np.ceil(loadings_size/sizes[1]))
            chunks = [(int(tshape[0]/factors_chunks), -1),
                      (int(tshape[2]/loadings_chunks), -1)]

        return chunks

    def decomposition(self, normalize_poissonian_noise=False,
                      algorithm=None, output_dimension=None,
                      mbytes_chunk=100, navigation_mask=None,
                      signal_mask=None, *args, **kwargs):
        """Decomposition with a choice of algorithms.

        For a full description of parameters see
        :func:`hyperspy._signals.lazy.decomposition()`.

        This is a wrapper for HyperSpy's ``decomposition()`` function,
        except for an alternative use of scikit-image's IncrementalPCA
        algorithm.

        Parameters
        ----------
        normalize_poissonian_noise : bool
            If True (default is False), scale the patterns to normalise
            Poissonian noise.
        algorithm : {'svd', 'IPCA', 'PCA', 'ORPCA', 'ONMF'}, optional
            Default is 'svd', lazy SVD decomposition from dask. 'PCA'
            gives HyperSpy's use of scikit-learn's IncrementalPCA,
            while 'IPCA' gives our use of IncrementalPCA.
        output_dimension : int
            Number of significant components to keep. If None, keep all
            (only valid for SVD).
        mbytes_chunk : int, optional
            Size of chunks in MB, default is 100 MB as suggested in the
            Dask documentation.
        navigation_mask : boolean array_like
        signal_mask : boolean array_like
        *args :
            Arguments to be passed to ``decomposition()``.
        **kwargs :
            Keyword arguments to be passed to ``decomposition()``.

        Returns
        -------
        The results are stored in self.learning_results.
        """
        if self.data.dtype.char not in ['e', 'f', 'd']:  # If not float
            raise TypeError(
                'To perform a decomposition the data must be of the float '
                'type, but the current type is \'{}\'. '
                'No decomposition was performed.'.format(self.data.dtype))

        if algorithm == 'IPCA':
            if output_dimension is None:
                raise ValueError("With the IncrementalPCA algorithm, "
                                 "output_dimension must be specified")

            from sklearn.decomposition import IncrementalPCA

            # Normalise Poissonian noise
            original_data = self.data
            if normalize_poissonian_noise:
                rbH, raG = self._normalize_poissonian_noise(
                    navigation_mask=navigation_mask, signal_mask=signal_mask)

            # Prepare data matrix
            nx, ny, sx, sy = self.data.shape
            n, s = nx * ny, sx * sy
            X = self.data.reshape((n, s))

            # Determine number of chunks
            suggested_size = mbytes_chunk * 2 ** 20
            num_chunks = int(np.ceil(X.nbytes / suggested_size))
            cpus = os.cpu_count()
            if num_chunks <= cpus:
                num_chunks = cpus
            chunk_size = n // num_chunks

            # Get principal components (factors)
            ipca = IncrementalPCA(n_components=output_dimension)
            for i in tqdm.tqdm(iterable=range(0, num_chunks), total=num_chunks,
                               leave=True, desc='Learn'):
                start = i * chunk_size
                end = (i + 1) * chunk_size
                if i == (num_chunks - 1):  # Last iteration
                    end = None
                ipca.partial_fit(X[start:end])  # Fit
            factors = ipca.components_.T

            # Reproject data on the principal components (loadings)
            loadings = []
            for j in tqdm.tqdm(iterable=range(0, num_chunks), total=num_chunks,
                               leave=True, desc='Project'):
                start = j * chunk_size
                end = (j + 1) * chunk_size
                if j == (num_chunks - 1):  # Last iteration
                    end = None
                loadings.append(ipca.transform(X[start:end]))  # Reproject
            loadings = np.concatenate(loadings, axis=0)

            # Set signal's learning results
            target = self.learning_results
            target.decomposition_algorithm = algorithm
            target.output_dimension = output_dimension
            target.factors = factors
            target.loadings = loadings
            target.explained_variance = ipca.explained_variance_
            target.explained_variance_ratio = ipca.explained_variance_ratio_

            # Revert data
            self.data = original_data

            if normalize_poissonian_noise is True:
                target.factors = target.factors * rbH.ravel()[:, np.newaxis]
                target.loadings = target.loadings * raG.ravel()[:, np.newaxis]

        else:  # Call HyperSpy's implementation
            super(self).decomposition(*args, **kwargs)

    def _normalize_poissonian_noise(self, navigation_mask=None,
                                    signal_mask=None):
        """Scales the patterns following [1]_.

        Adapted from HyperSpy.

        Parameters
        ----------
        navigation_mask : boolean array_like
        signal_mask : boolean array_like

        Returns
        -------
        raG : array_like
            Matrix corresponding to square root of aG in referenced
            paper.
        rbH : array_like
            Matrix corresponding to square root of bH in referenced
            paper.

        References
        ----------
        .. [1] Keenan, Michael R, Kotula, Paul G: Accounting for Poisson
               noise in the multivariate analysis of ToF-SIMS spectrum
               images, Surface and Interface Analysis 36(3), Wiley
               Online Library, 203–212, 2004.
        """
        from hyperspy._signals.lazy import to_array
        data = self._data_aligned_with_axes
        ndim = self.axes_manager.navigation_dimension
        sdim = self.axes_manager.signal_dimension
        nav_chunks = data.chunks[:ndim]
        sig_chunks = data.chunks[ndim:]
        nm = da.logical_not(
            da.zeros(self.axes_manager.navigation_shape[::-1],
                     chunks=nav_chunks)
            if navigation_mask is None else to_array(
                navigation_mask, chunks=nav_chunks))
        sm = da.logical_not(
            da.zeros(
                self.axes_manager.signal_shape[::-1],
                chunks=sig_chunks)
            if signal_mask is None else to_array(
                signal_mask, chunks=sig_chunks))
        bH, aG = da.compute(
            data.sum(axis=tuple(range(ndim))),
            data.sum(axis=tuple(range(ndim, ndim + sdim))))
        bH = da.where(sm, bH, 1)
        aG = da.where(nm, aG, 1)

        raG = da.sqrt(aG)
        rbH = da.sqrt(bH)

        coeff = raG[(...,) + (None,) * rbH.ndim] * \
                rbH[(None,) * raG.ndim + (...,)]
        coeff.map_blocks(np.nan_to_num)
        coeff = da.where(coeff == 0, 1, coeff)
        data = data / coeff
        self.data = data

        return rbH, raG
