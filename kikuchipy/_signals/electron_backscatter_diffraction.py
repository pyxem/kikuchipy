# -*- coding: utf-8 -*-
"""Signal class for Electron Backscatter Diffraction (EBSD) data."""
import warnings
import numpy as np
import dask.array as da
import os
import gc
import datetime

from h5py import File
from hyperspy.api import plot
from hyperspy.signals import Signal2D
from hyperspy._lazy_signals import LazySignal2D
from hyperspy.learn.mva import LearningResults
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

    def get_decomposition_model(self, components=None, dtype_out=np.float16,
                                *args, **kwargs):
        """Return the model signal generated with the selected number of
        principal components.

        This function calls HyperSpy's ``get_decomposition_model``. The
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
        dtype_out : numpy dtype, optional
            Data type of learning results. Default is float16.
            HyperSpy's ``decomposition`` returns them in float64, which
            here is assumed to be overkill.
        *args
            Passed to Hyperspy's ``get_decomposition_model``.
        **kwargs
            Passed to Hyperspy's ``get_decomposition_model``.

        Returns
        -------
        Signal instance from components
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
        if self._lazy:
            assign_class = LazyEBSD
        else:
            assign_class = EBSD
        self.__class__ = assign_class
        s_model.__class__ = assign_class

        # Remove learning results from model signal
        s_model.learning_results = LearningResults()

        return s_model

    def get_decomposition_model_write(self, components=None,
                                      dtype_learn=np.float16):
        """Write the model signal generated from the selected number of
        principal components directly to a .hspy file. The model signal
        intensities are rescaled to the original signals' data type
        range.

        Notes
        -----
        Multiplying the learning results' factors and loadings in memory
        to create the model signal can sometimes not be done due to too
        large matrices. Here, instead, learning results are written to
        file, read into dask arrays and multiplied using dask's
        ``matmul``, out of core.

        Due to memory leakage (please help) when calling the function
        ``rescale_pattern_intensity`` in ``map`` the model signal is
        written to a temporary file before rescaling and written to a
        .hspy file. Be aware that this temporary file is typically twice
        the size of the original data set file.

        Parameters
        ----------
        components : {None, int or list of ints}, optional
            If None (default), rebuilds the signal from all components.
            If int, rebuilds signal from components in range 0-given
            int. If list of ints, rebuilds signal from only components
            in given list.
        dtype_learn : data-type, optional
            Data type to set learning results to (default is float16).
        """
        if self._lazy is False:
            raise ValueError("This function assumes the model signal is too "
                             "large to compute in memory, use "
                             "get_decomposition_model() instead")

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
        datadir = self.original_metadata.General.original_filepath
        t_str = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
        file_learn = os.path.join(datadir, 'learn_' + t_str + '.h5')
        with File(file_learn, 'w') as f:
            f.create_dataset(name='factors', data=factors)
            f.create_dataset(name='loadings', data=loadings)

        # Matrix multiplication
        with File(file_learn, 'r') as f:
            # Read learning results from HDF5 file
            chunks = self._rechunk_learning_results()
            factors = da.from_array(f['factors'], chunks=chunks[0])
            loadings = da.from_array(f['loadings'], chunks=chunks[1])

            # Perform the matrix multiplication
            loadings = loadings.T
            res = factors @ loadings
            res = res.T  # Transpose
            res = res.reshape(self.data.shape)  # Reshape

            # TODO: Avoid write to file, and directly rescale with no mem. leak?
            # Write model data to file
            file_model = os.path.join(datadir, 'model_' + t_str + '.hdf5')
            res.to_hdf5(file_model, '/result')

        # TODO: Find out where we leak memory and make this line unnecessary
        # Collect garbage (so we don't sink!)
        gc.collect()

        # Write model signal to .hspy file
        with File(file_model, 'r') as f:
            # Create signal from results
            s_model = self.deepcopy()
            s_model.learning_results = LearningResults()
            s_model.data = da.from_array(f['result'], chunks=(1, 1, -1, -1))

            # Rescale intensities
            s_model.map(rescale_pattern_intensity, ragged=False)
            s_model.data = s_model.data.astype(self.data.dtype)

            # Write signal to file (rechunking saves a little time?)
            file_model2 = os.path.join(datadir, 'model2_' + t_str)
            chunks = s_model._get_dask_chunks()
            s_model.data = s_model.data.rechunk(chunks=chunks)
            gc.collect()
            s_model.save(file_model2)

        # Delete temporary files
        os.remove(file_learn)
        os.remove(file_model)

        gc.collect()  # Don't sink

    def _rechunk_learning_results(self, mbytes_chunk=100):
        """Return suggested data chunks for learning results. It is
        assumed that the loadings are not transposed. The last axes of
        factors and loadings are not chunked. The aims in prioritised
        order:
            1. Split into at least as many chunks as available CPUs.
            2. Limit chunks to around input MB (mbytes).
            3. Keep first axis of factors.

        Adapted from HyperSpy's ``_get_dask_chunks``.

        Parameters
        ----------
        mbytes_chunk : int, optional
            Size of chunks in MB, default is 100 MB as suggested in the
            Dask documentation.

        Returns
        -------
        List of two tuples
            First/second tuple is suggested chunks to pass to
            ``dask.array.rechunk`` for factors/loadings, respectively.
        """
        target = self.learning_results
        if target.decomposition_algorithm is None:
            raise ValueError("No learning results were found.")

        # Get dask chunks
        factors = target.factors
        loadings = target.loadings
        tshape = factors.shape + loadings.shape

        # Make sure the last factors/loading axes have the same shapes
        # TODO: Should also handle the case where the first axes are the same
        if tshape[1] != tshape[3]:
            raise ValueError("The last dimensions in factors and loadings are "
                             "not the same.")

        # Determine max. number of (strictly necessary) chunks
        suggested_size = mbytes_chunk * 2**20  # 100 MB default
        factors_size = factors.nbytes
        loadings_size = loadings.nbytes
        total_size = factors_size + loadings_size
        num_chunks = np.ceil(total_size / suggested_size)

        # Get chunk sizes
        cpus = os.cpu_count()
        if num_chunks <= cpus:  # Return approx. as many chunks as CPUs
            chunks = [(-1, -1), (int(tshape[2]/cpus), -1)]  # -1 = don't chunk
        elif factors.nbytes <= suggested_size:  # Chunk first axis in loadings
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
