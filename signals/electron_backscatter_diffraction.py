# -*- coding: utf-8 -*-
"""Signal class for Electron Backscatter Diffraction (EBSD) data."""
import numpy as np

from hyperspy.api import plot
from hyperspy.signals import Signal2D, BaseSignal
from skimage.exposure import rescale_intensity
from scipy.ndimage import gaussian_filter, median_filter
from matplotlib.pyplot import imread
from pyxem.signals.electron_diffraction import ElectronDiffraction


class ElectronBackscatterDiffraction(Signal2D):
    _signal_type = 'electron_backscatter_diffraction'

    def __init__(self, *args, **kwargs):
        Signal2D.__init__(self, *args, **kwargs)
        self.set_experimental_parameters()

    def set_experimental_parameters(self, deadpixels_corrected=False,
                                    deadpixels=None, deadvalue=None):
        """Set experimental parameters in metadata.

        Parameters
        ----------
        deadpixels_corrected : bool
            If True (default is False), deadpixels in patterns are corrected.
        deadpixels : list of tuple
            List of tuples containing pattern indices for dead pixels.
        deadvalue : string
            Specifies how dead pixels have been corrected for (average or nan).

        """

        md = self.metadata
        md.set_item('Acquisition_instrument.SEM.Detector.deadpixels_corrected',
                    deadpixels_corrected)
        md.set_item('Acquisition_instrument.SEM.Detector.deadpixels',
                    deadpixels)
        md.set_item('Acquisition_instrument.SEM.Detector.deadvalue',
                    deadvalue)

    def remove_background(self, static=True, dynamic=False, bg_path=None,
                          divide=False, sigma=None, *args, **kwargs):
        """Perform background correction, either static, dynamic or both. For
        the static correction, a background image is subtracted from all
        patterns. For the dynamic correction, each pattern is blurred using a
        Gaussian kernel with a standard deviation set by you. The corrections
        can either be done by subtraction or division. Relative intensities
        between patterns are lost after dynamic correction.

        Parameters
        ----------
        static : bool
            If True (default), static correction is performed.
        dynamic : bool
            If True (default is False), dynamic correction is performed.
        bg_path : file path (default:None)
            File path to background image for static correction. If None
            (default), only dynamic correction can be performed.
        divide : bool
            If True (default is False), static and dynamic correction is
            performed by division and not subtraction.
        sigma : int, float (default:None)
            Standard deviation for the gaussian kernel for dynamic correction.
            If None (default), a deviation of pattern width/20 is chosen.
        *args
            Arguments to be passed to map().
        **kwargs:
            Arguments to be passed to map().

        """

        # Change data types (increase bit depth) to avoid negative intensities
        signal_dtype = np.iinfo(self.data.dtype)
        self.data = self.data.astype('int16')

        if static:
            # Read and set up background image
            if bg_path is not None:
                bg = imread(bg_path)
                bg = ElectronBackscatterDiffraction(bg)
                bg.data = bg.data.astype('int16')
            else:
                raise ValueError("No background image provided")

            # Check if dead pixels are corrected for, and if so, correct dead
            # pixels in background pattern
            if self.metadata.Acquisition_instrument.SEM.Detector\
                    .deadpixels_corrected:
                bg.remove_deadpixels(self.metadata.Acquisition_instrument.SEM\
                                     .Detector.deadpixels, self.metadata\
                                     .Acquisition_instrument.SEM.Detector\
                                     .deadvalue)

            # Subtract by background pattern
            if divide:
                self.data = self.data / bg.data
            else:
                self.data = self.data - bg.data

            # Create new minimum and maximum intensities, keeping the ratios
            # First, get new maximums and minimums after background subtraction
            smin = self.min(self.axes_manager.signal_axes)
            smax = self.max(self.axes_manager.signal_axes)

            # Set lowest intensity to zero
            int_min = smin.data.min()
            smin = smin - int_min
            smax = smax - int_min

            # Get scaling factor and scale intensities
            scale = signal_dtype.max / smax.data.max()
            smin = smin * scale
            smax = smax * scale

            # Convert to original data type and write to memory if lazy
            smin.data = smin.data.astype(signal_dtype.dtype)
            smax.data = smax.data.astype(signal_dtype.dtype)
            if self._lazy:
                smin.compute()
                smax.compute()

            # Have to create a wrapper for rescale_intensity since values
            # passed to out_range differs for all patterns
            def rescale_pattern(pattern, signal_min, signal_max):
                return rescale_intensity(pattern, out_range=(signal_min,
                                                             signal_max))

            # Rescale each pattern according to its min. and max. intensity
            self.map(rescale_pattern, signal_min=smin, signal_max=smax, *args,
                     **kwargs)

        if dynamic:
            if sigma is None:
                sigma = self.axes_manager.signal_axes[0].size/20

            # Create signal with each pattern blurred by a Gaussian kernel
            s_blur = self.map(gaussian_filter, inplace=False, ragged=False,
                              sigma=sigma)
            s_blur.change_dtype('int16')

            if divide:
                self.data = self.data / s_blur.data
            else:
                self.data = self.data - s_blur.data

            # We don't care about relative intensities anymore since the
            # subtracted pattern is different for all patterns. We therefore
            # rescale intensities according to the original data type range
            self.map(rescale_intensity, ragged=False,
                     out_range=signal_dtype.dtype.name)

        # Revert data type
        self.data = self.data.astype(signal_dtype.dtype)

    def find_deadpixels(self, pattern=(0, 0), threshold=10, to_plot=True):
        """Find dead pixels in experimentally acquired diffraction patterns by
        comparing pixel values in a blurred version of a selected pattern to
        the original pattern. If the intensity difference is above a threshold
        the pixel is labeled as dead.

        Parameters
        ----------
        pattern : tuple
            Indices of pattern in which to search for dead pixels.
        threshold : int
            Threshold for difference in pixel intensities between blurred and
            original pattern.
        to_plot : bool
            If True (default), a pattern with the dead pixels highlighted is
            plotted.

        Returns
        -------
        deadpixels : list of tuples
            List of tuples containing pattern indices for dead pixels.

        """

        pat = self.inav[pattern].data.astype('int16')  # Avoid negative pixels
        blurred = median_filter(pat, size=2)
        difference = pat - blurred
        threshold = threshold * np.std(difference)

        # Find the dead pixels (ignoring border pixels)
        deadpixels = np.nonzero((np.abs(difference[1:-1, 1:-1]) > threshold))
        deadpixels = np.array(deadpixels) + 1
        deadpixels = list(map(tuple, deadpixels.T))  # List of tuples

        if to_plot:
            pat = self.inav[pattern]
            for (y, x) in deadpixels:
                m = plot.markers.point(x, y, color='red')
                pat.add_marker(m)
            self.inav[pattern].plot()

        return deadpixels


    def remove_deadpixels(self, deadpixels, deadvalue='average', inplace=True,
                          *args, **kwargs):
        """Remove dead pixels from experimentally acquired diffraction
        patterns, either by averaging or setting to a certain value.

        Uses pyXem's remove_deadpixels() function.

        Parameters
        ----------
        deadpixels : list of tuples
            List of tuples of indices of dead pixels.
        deadvalue : string
            Specify how deadpixels should be treated. 'average' sets the dead
            pixel value to the average of adjacent pixels. 'nan' sets the dead
            pixel to nan
        inplace : bool
            If True (default), this signal is overwritten. Otherwise, returns a
            new signal.
        *args:
            Arguments to be passed to map().
        **kwargs:
            Keyword arguments to be passed to map().

        """

        if inplace and deadpixels:
            self.set_experimental_parameters(deadpixels_corrected=True,
                                             deadpixels=deadpixels,
                                             deadvalue=deadvalue)
            return ElectronDiffraction.remove_deadpixels(self,
                                                         deadpixels=deadpixels,
                                                         deadvalue=deadvalue,
                                                         inplace=inplace, *args,
                                                         **kwargs)
        elif not inplace and deadpixels:
            s = ElectronDiffraction.remove_deadpixels(self,
                                                      deadpixels=deadpixels,
                                                      deadvalue=deadvalue,
                                                      inplace=inplace, *args,
                                                      **kwargs)
            s.set_experimental_parameters(deadpixels_corrected=True,
                                          deadpixels=deadpixels,
                                          deadvalue=deadvalue)
            return s
        else:  # Inplace is passed, but there are no dead pixels detected
            pass
