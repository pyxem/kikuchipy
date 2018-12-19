# -*- coding: utf-8 -*-
"""Signal class for Electron Backscatter Diffraction (EBSD) data."""
import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
import scipy.ndimage as scn

from hyperspy.signals import Signal2D, BaseSignal


class ElectronBackscatterDiffraction(Signal2D):
    _signal_type = 'electron_backscatter_diffraction'

    def __init__(self, *args, **kwargs):
        Signal2D.__init__(self, *args, **kwargs)
        self.decomposition.__func__.__doc__ = BaseSignal.decomposition.__doc__

    def remove_background(self, static=True, dynamic=False, bgimg_path=None,
                          divide=False, sigma=None, *args, **kwargs):
        """Perform background correction, either static, dynamic or both. For
        the static correction, a background image is subtracted from all
        patterns. For the dynamic correction, each pattern is blurred using a
        Gaussian kernel with a standard deviation set by you. The correction
        can either be done by subtraction or division. Relative intensities
        between patterns are lost after dynamic correction.

        Parameters
        ----------
        static : bool
            If True (default), static correction is performed.
        dynamic : bool
            If True (default is False), dynamic correction is performed.
        bgimg_path : file path (default:None)
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
            if bgimg_path is not None:
                bgimg = plt.imread(bgimg_path)
                bgimg = Signal2D(bgimg)
                bgimg.change_dtype('int16')
            else:
                raise ValueError("No background image provided")

            # Subtract by background pattern
            if divide:
                self.data = self.data / bgimg.data
            else:
                self.data = self.data - bgimg.data

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

            def rescale_pattern(pattern, signal_min, signal_max):
                return ski.exposure.rescale_intensity(pattern,
                                                      out_range=(signal_min,
                                                                 signal_max))

            # Rescale each pattern according to its min. and max. intensity
            self.map(rescale_pattern, signal_min=smin, signal_max=smax, *args,
                     **kwargs)

            # Finally, revert data type to save memory
            self.data = self.data.astype(signal_dtype.dtype)

        if dynamic:
            if sigma is None:
                sigma = self.axes_manager.signal_axes[0].size/20

            # Create signal with each pattern blurred by a Gaussian kernel
            s_blur = self.map(scn.gaussian_filter, inplace=False, ragged=False,
                           sigma=sigma)
            s_blur.change_dtype('int16')

            if divide:
                self.data = self.data / s_blur.data
            else:
                self.data = self.data - s_blur.data

            # We don't care about relative intensities anymore since the
            # subtracted pattern is different for all patterns. We therefore
            # rescale intensities according to the original datatype range
            self.map(ski.exposure.rescale_intensity, ragged=False,
                     out_range=signal_dtype.dtype.name)
            self.data = self.data.astype(signal_dtype.dtype)
