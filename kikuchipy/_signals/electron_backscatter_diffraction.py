# -*- coding: utf-8 -*-
"""Signal class for Electron Backscatter Diffraction (EBSD) data."""
import warnings
import numpy as np
import os
import numbers

from hyperspy.api import plot
from hyperspy.signals import Signal2D
from hyperspy._lazy_signals import LazySignal2D
from skimage.transform import radon
from matplotlib.pyplot import imread
from pyxem.signals.electron_diffraction import ElectronDiffraction

from dask.diagnostics import ProgressBar
from hyperspy.misc.utils import dummy_context_manager

from kikuchipy._signals.radon_transform import RadonTransform
from kikuchipy.utils.expt_utils import (correct_background, remove_dead,
                                        find_deadpixels_single_pattern,
                                        plot_markers_single_pattern,
                                        equalize_adapthist_pattern)
from kikuchipy import io


class EBSD(Signal2D):
    _signal_type = 'electron_backscatter_diffraction'
    _lazy = False

    def __init__(self, *args, **kwargs):
        if self._lazy and args:
            Signal2D.__init__(self, data=args[0].data, **kwargs)
        else:
            Signal2D.__init__(self, *args, **kwargs)
        self.set_experimental_parameters(deadpixels_corrected=False)

    def set_experimental_parameters(self, accelerating_voltage=None,
                                    condenser_aperture=None,
                                    deadpixels_corrected=None, deadvalue=None,
                                    deadpixels=None, deadthreshold=None,
                                    exposure_time=None, frame_rate=None,
                                    working_distance=None):
        """Set experimental parameters in metadata.

        Parameters
        ----------
        accelerating_voltage : float, optional
            Accelerating voltage in kV.
        condenser_aperture : float, optional
            Condenser_aperture in µm.
        deadpixels_corrected : bool, optional
            If True (default is False), deadpixels in patterns are
            corrected.
        deadpixels : list of tuple, optional
            List of tuples containing pattern indices for dead pixels.
        deadvalue : string, optional
            Specifies how dead pixels have been corrected for (average
            or nan).
        deadthreshold : int, optional
            Threshold for detecting dead pixels.
        exposure_time : float, optional
            Exposure time in µs.
        frame_rate : float, optional
            Frame rate in fps.
        working_distance : float, optional
            Working distance in mm.
        """
        md = self.metadata
        omd = self.original_metadata
        sem = 'Acquisition_instrument.SEM.'
        ebsd = sem + 'Detector.EBSD.'

        if accelerating_voltage is not None:
            md.set_item(sem + 'accelerating_voltage', accelerating_voltage)
        if condenser_aperture is not None:
            omd.set_item(sem + 'condenser_aperture', condenser_aperture)
        if deadpixels_corrected is not None:
            omd.set_item(ebsd + 'deadpixels_corrected', deadpixels_corrected)
        if deadpixels is not None:
            omd.set_item(ebsd + 'deadpixels', deadpixels)
        if deadvalue is not None:
            omd.set_item(ebsd + 'deadvalue', deadvalue)
        if deadthreshold is not None:
            omd.set_item(ebsd + 'deadthreshold', deadthreshold)
        if exposure_time is not None:
            omd.set_item(ebsd + 'exposure_time', exposure_time)
        if frame_rate is not None:
            omd.set_item(ebsd + 'frame_rate', frame_rate)
        if working_distance is not None:
            omd.set_item(ebsd + 'working_distance', working_distance)

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

    def remove_background(self, static=True, dynamic=True, bg=None,
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
        bg : file path (default:None), optional
            File path to background image for static correction. If
            None, we try to read 'Background acquisition pattern.bmp'
            from signal directory.
        relative : bool, optional
            If True (default is False), relative intensities between
            patterns are kept after static correction.
        sigma : int, float (default:None), optional
            Standard deviation for the gaussian kernel for dynamic
            correction. If None (default), a deviation of pattern
            width/30 is chosen.
        **kwargs:
            Arguments to be passed to map().
        """
        if not static and not dynamic:
            raise ValueError("No correction done")

        lazy = self._lazy
        if lazy:
            kwargs['ragged'] = False

        # Set default values for contrast stretching parameters, to be
        # overwritten if 'relative' is passed
        imin = None
        scale = None

        if static:
            if bg is None:
                try:  # Try to load from signal directory
                    bg_fname = 'Background acquisition pattern.bmp'
                    omd = self.original_metadata
                    bg = os.path.join(omd.General.original_filepath, bg_fname)
                except ValueError:
                    raise ValueError("No background image provided")

            # Read and setup background
            bg = imread(bg)
            bg = Signal2D(bg)

            # Correct dead pixels in background if they are corrected in signal
            omd = self.original_metadata.Acquisition_instrument.SEM.\
                Detector.EBSD
            if (omd.deadpixels_corrected and omd.deadpixels.any()
                    and omd.deadvalue):
                bg.data = remove_dead(bg.data, omd.deadpixels, omd.deadvalue)

            if relative and not dynamic:
                # Get lowest intensity after subtraction
                smin = self.min(self.axes_manager.navigation_axes)
                smax = self.max(self.axes_manager.navigation_axes)
                if lazy:
                    smin.compute()
                    smax.compute()
                smin.change_dtype(np.int8)
                bg.change_dtype(np.int8)
                imin = (smin.data - bg.data).min()

                # Get highest intensity after subtraction
                bg.data = bg.data.astype(np.uint8)
                imax = (smax.data - bg.data).max() + abs(imin)

                # Get global scaling factor, input dtype max. value in nominator
                scale = float(np.iinfo(self.data.dtype).max / imax)

        if dynamic and sigma is None:
            sigma = int(self.axes_manager.signal_axes[0].size/30)

        self.map(correct_background, static=static, dynamic=dynamic, bg=bg,
                 sigma=sigma, imin=imin, scale=scale, **kwargs)

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
                        pattern_coordinates=None, to_plot=False,
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
        to_plot : bool, optional
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
        if to_plot:
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
            # TODO: make sure remove_dead() stop leaking memory
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

    def save(self, filename=None, overwrite=None, extension=None, **kwargs):
        """Saves the signal in the specified format.
        The function gets the format from the extension:
            - hspy for HyperSpy's HDF5 specification
            - dat for NORDIF binary format
        If no extension is provided the default file format as defined
        in the `preferences` is used. Please note that not all the
        formats supports saving datasets of arbitrary dimensions. Each
        format accepts a different set of parameters. For details see
        the specific format documentation.

        Parameters
        ----------
        filename : str or None
            If None (default) and `tmp_parameters.filename` and
            `tmp_parameters.folder` are defined, the filename and path
            will be taken from there. A valid extension can be provided
            e.g. "Pattern.dat", see `extension`.
        overwrite : None, bool
            If None and the file exists, it will query the user. If
            True (False) it (does not) overwrite the file if it exists.
        extension : {None, 'hspy', 'hdf5', 'dat', common image
                     extensions e.g. 'tiff', 'png'}
            The extension of the file that defines the file format.
            'hspy' and 'hdf5' are equivalent. Use 'hdf5' if
            compatibility with HyperSpy versions older than 1.2 is
            required. If None, the extension is determined from the
            following list in this order:
            i) the filename
            ii)  `Signal.tmp_parameters.extension`
            iii) `hspy` (the default extension)
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
                raise ValueError("File name not defined")
        if extension is not None:
            basename, ext = os.path.splitext(filename)
            filename = basename + '.' + extension
        io.save(filename, self, overwrite=overwrite, **kwargs)


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
