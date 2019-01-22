# -*- coding: utf-8 -*-
"""Signal class for Electron Backscatter Diffraction (EBSD) data."""
import warnings
import numpy as np
import dask.array as da
import os
import datetime

from h5py import File
from hyperspy.api import plot
from hyperspy.signals import Signal2D
from hyperspy._lazy_signals import LazySignal2D
from skimage.transform import radon
from scipy.ndimage import median_filter
from matplotlib.pyplot import imread
from pyxem.signals.electron_diffraction import ElectronDiffraction

from dask.diagnostics import ProgressBar
from hyperspy.misc.utils import dummy_context_manager

from kikuchipy import io
from kikuchipy._signals.radon_transform import RadonTransform
from kikuchipy.utils.expt_utils import (correct_background, remove_dead,
                                        rescale_pattern_intensity)


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
        calibration: float
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
        calibration: float
            Diffraction pattern calibration in reciprocal Angstroms per
            pixel.
        """
        ElectronDiffraction.set_diffraction_calibration(self, calibration)
        self.axes_manager.signal_axes[0].offset = 0
        self.axes_manager.signal_axes[1].offset = 0

    def remove_background(self, static=True, dynamic=True, bg=None,
                          relative=False, sigma=None, **kwargs):
        """Perform background correction, either static, dynamic or
        both, on a stack of electron backscatter diffraction patterns.

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
            raise ValueError("No correction done, quitting")

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
            if omd.deadpixels_corrected and omd.deadpixels and omd.deadvalue:
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

    def find_deadpixels(self, pattern=(0, 0), threshold=10, to_plot=False):
        """Find dead pixels in experimentally acquired diffraction
        patterns by comparing pixel values in a blurred version of a
        selected pattern to the original pattern. If the intensity
        difference is above a threshold the pixel is labeled as dead.

        Assumes self has navigation axes, i.e. does not work on a
        single pattern.

        Parameters
        ----------
        pattern : tuple, optional
            Indices of pattern in which to search for dead pixels.
        threshold : int, optional
            Threshold for difference in pixel intensities between
            blurred and original pattern.
        to_plot : bool, optional
            If True (default is False), a pattern with the dead pixels
            highlighted is plotted.

        Returns
        -------
        deadpixels : list of tuples
            List of tuples containing pattern indices for dead pixels.
        """
        pat = self.inav[pattern].data.astype(np.int16)
        if self._lazy:
            pat = pat.compute(show_progressbar=False)
        blurred = median_filter(pat, size=2)
        difference = pat - blurred
        threshold = threshold * np.std(difference)

        # Find the dead pixels (ignoring border pixels)
        deadpixels = np.nonzero((np.abs(difference[1:-1, 1:-1]) > threshold))
        deadpixels = np.array(deadpixels) + 1
        deadpixels = list(map(tuple, deadpixels.T))  # List of tuples

        # Update original_metadata
        self.set_experimental_parameters(deadpixels=deadpixels)

        if to_plot:
            pat = self.inav[pattern]
            for (y, x) in deadpixels:
                m = plot.markers.point(x, y, color='red')
                pat.add_marker(m)
            self.inav[pattern].plot()

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

    def decomposition_model_to_file(self, components=None):
        """Get the signal generated from the principal components and
        write to a .hspy file.

        The function creates an HDF5 file to store data during
        execution, and deletes it at the end.

        Parameters
        ----------
        components : {None, int or list of ints}
            If None (default), rebuilds the signal from all components.
            If int, rebuilds signal from components in range 0-given
            int. If list of ints, rebuilds signal from only components
            in given list.
        """
        # Get learning results from signal and write to memory
        target = self.learning_results
        factors = np.array(target.factors)
        loadings = np.array(target.loadings.T)

        # Extract relevant components
        if hasattr(components, '__iter__'):  # components is a list of ints
            tfactors = np.zeros((factors.shape[0], len(components)))
            tloadings = np.zeros((len(components), loadings.shape[1]))
            for i in range(len(components)):
                tfactors[:, i] = factors[:, components[i]]
                tloadings[i, :] = loadings[components[i], :]
            factors = tfactors
            loadings = tloadings
        else:  # components is an int
            factors = factors[:, :components]
            loadings = loadings[:components, :]

        # Rescale factors and loadings to int16, keeping relative values
        # within factors and loadings, and between factors and loadings.
        # We assume that factors have higher values than loadings, and scale
        # factors to fill the whole of int16 range. Rescale to int32 the
        # following matrix multiplication (@) most likely gives values
        # higher than the int16 range.
        dtype_out = np.int32
        scale = float(np.iinfo(np.int16).max / factors.max())
        factors = np.array(factors * scale, dtype=dtype_out)
        loadings = np.array(loadings * scale, dtype=dtype_out)

        # Write learning results to HDF5 file
        datadir = self.original_metadata.General.original_filepath
        t_str = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
        file = os.path.join(datadir, 'learn_' + t_str + '.h5')
        with File(file, 'w') as f:
            f.create_dataset(name='factors', data=factors)
            f.create_dataset(name='loadings', data=loadings)

        # Matrix multiplication
        with File(file, 'r+') as d:
            # Read learning results from HDF5 file
            # TODO: Fix automatic chunking suited to file and available memory
            factors = da.from_array(d['factors'], chunks=(-1, 'auto'))
            loadings = da.from_array(d['loadings'], chunks=('auto', 500))

            # Perform the matrix multiplication
            res = factors @ loadings
            res = res.T  # Transpose
            res = res.reshape(self.data.shape)  # Reshape

            # Write rebuilt signal to HDF5 file
            da.to_hdf5(file, '/result', res)

        # Create signal from multiplication results and rescale intensities,
        # and write signal to file
        with File(file, 'r') as d:
            res = self.deepcopy()
            res.data = da.from_array(d['result'], chunks=(1, 1, -1, -1))

            # Rescale intensities
            res.map(rescale_pattern_intensity, ragged=False)

            # Write signal to .hspy file
            res.save(os.path.join(datadir, 'model_' + t_str))

        # Delete HDF5 file with learning results
        os.remove(file)


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
