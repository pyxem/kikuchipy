# -*- coding: utf-8 -*-
"""Signal class for Electron Backscatter Diffraction (EBSD) data."""
import warnings
import numpy as np
import gc

from hyperspy.api import plot
from hyperspy.signals import Signal2D
from hyperspy._lazy_signals import LazySignal2D
from skimage.transform import radon
from scipy.ndimage import median_filter
from matplotlib.pyplot import imread
from pyxem.signals.electron_diffraction import ElectronDiffraction
from pyxem.utils.expt_utils import remove_dead

from kikuchipy._signals.radon_transform import RadonTransform
from kikuchipy.utils.expt_utils import correct_background


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
            If True (default is False), deadpixels in patterns are corrected.
        deadpixels : list of tuple, optional
            List of tuples containing pattern indices for dead pixels.
        deadvalue : string, optional
            Specifies how dead pixels have been corrected for (average or nan).
        deadthreshold : int, optional
            Threshold for detecting dead pixels.
        exposure_time : float, optional
            Exposure time in µs.
        frame_rate : float, optional
            Frame rate in fps.
        working_distance : float, optional
            Working distance in mm.
        """
        # TODO: Fetch directly from the settings.txt file for nordif2hdf5?
        # TODO: Not overwrite metadata stored in the file from before.

        md = self.metadata
        md_str = 'Acquisition_instrument.SEM.'

        if accelerating_voltage is not None:
            md.set_item(md_str + 'accelerating_voltage', accelerating_voltage)
        if condenser_aperture is not None:
            md.set_item(md_str + 'condenser_aperture', condenser_aperture)
        if deadpixels_corrected is not None:
            md.set_item(md_str + 'Detector.deadpixels_corrected',
                        deadpixels_corrected)
        if deadpixels is not None:
            md.set_item(md_str + 'Detector.deadpixels', deadpixels)
        if deadvalue is not None:
            md.set_item(md_str + 'Detector.deadvalue', deadvalue)
        if deadthreshold is not None:
            md.set_item(md_str + 'Detector.deadthreshold', deadthreshold)
        if exposure_time is not None:
            md.set_item(md_str + 'Detector.Diffraction.exposure_time',
                        exposure_time)
        if frame_rate is not None:
            md.set_item(md_str + 'Detector.Diffraction.frame_rate', frame_rate)
        if working_distance is not None:
            md.set_item(md_str + 'working_distance', working_distance)

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
        """Set diffraction pattern pixel size in reciprocal Angstroms. The
        offset is set to 0 for signal_axes[0] and signal_axes[1].

        Parameters
        ----------
        calibration: float
            Diffraction pattern calibration in reciprocal Angstroms per pixel.
        """
        ElectronDiffraction.set_diffraction_calibration(self, calibration)
        self.axes_manager.signal_axes[0].offset = 0
        self.axes_manager.signal_axes[1].offset = 0

    def remove_background(self, static=True, dynamic=True, bg=None,
                          relative=False, sigma=None, *args, **kwargs):
        """Perform background correction, either static, dynamic or both, on
        a stack of electron backscatter diffraction patterns.

        For the static correction, a background image is subtracted from all
        patterns. For the dynamic correction, each pattern is blurred using a
        Gaussian kernel with a standard deviation set by you.

        Contrast stretching is done either according to a global or a local
        intensity range, the former maintaining relative intensities between
        patterns after static correction. Relative intensities are lost if
        only dynamic correction is performed.

        Input data is assumed to be a two-dimensional numpy array of patterns
        of dtype uint8.

        Parameters
        ----------
        static : bool, optional
            If True (default), static correction is performed.
        dynamic : bool, optional
            If True (default), dynamic correction is performed.
        bg : file path (default:None), optional
            File path to background image for static correction.
        relative : bool, optional
            If True (default is False), relative intensities between patterns
            are kept after static correction.
        sigma : int, float (default:None), optional
            Standard deviation for the gaussian kernel for dynamic correction.
            If None (default), a deviation of pattern width/30 is chosen.
        *args
            Arguments to be passed to map().
        **kwargs:
            Arguments to be passed to map().
        """
        if not static and not dynamic:
            raise ValueError("No correction done, quitting.")

        lazy = self._lazy
        if lazy:
            kwargs['ragged'] = False

        # Set default values for contrast stretching parameters, to be
        # overwritten if 'relative' is passed
        imin = None
        scale = None

        if static:
            # Read and set up background image
            if bg is not None:
                bg = imread(bg)
                bg = Signal2D(bg)
            else:
                raise ValueError("No background image provided")

            # Correct dead pixels in background if they are corrected in signal
            md = self.metadata.Acquisition_instrument.SEM.Detector
            if md.deadpixels_corrected and md.deadpixels and md.deadvalue:
                bg.data = remove_dead(bg.data, md.deadpixels, md.deadvalue)

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
                 sigma=sigma, imin=imin, scale=scale, *args, **kwargs)

        gc.collect()

    def find_deadpixels(self, pattern=(0, 0), threshold=10, to_plot=False):
        """Find dead pixels in experimentally acquired diffraction patterns by
        comparing pixel values in a blurred version of a selected pattern to
        the original pattern. If the intensity difference is above a threshold
        the pixel is labeled as dead.

        Assumes self has navigation axes, i.e. does not work on a single
        pattern.

        Parameters
        ----------
        pattern : tuple, optional
            Indices of pattern in which to search for dead pixels.
        threshold : int, optional
            Threshold for difference in pixel intensities between blurred and
            original pattern.
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

        Assumes signal has navigation axes, i.e. does not work on a single
        pattern.

        Uses pyXem's remove_deadpixels() function.

        Parameters
        ----------
        deadpixels : list of tuples
            List of tuples of indices of dead pixels.
        deadvalue : string, optional
            Specify how deadpixels should be treated. 'average' sets the dead
            pixel value to the average of adjacent pixels. 'nan' sets the dead
            pixel to nan.
        inplace : bool, optional
            If True (default), signal is overwritten. Otherwise, returns a new
            signal.
        *args:
            Arguments to be passed to map().
        **kwargs:
            Keyword arguments to be passed to map().
        """
        if self._lazy:
            kwargs['ragged'] = False
        if inplace and deadpixels:
            # TODO: make sure remove_dead() stop leaking memory
            self.map(remove_dead, deadpixels=deadpixels, deadvalue=deadvalue,
                     inplace=inplace, *args, **kwargs)
            self.set_experimental_parameters(deadpixels_corrected=True,
                                             deadpixels=deadpixels,
                                             deadvalue=deadvalue)
        elif not inplace and deadpixels:
            s = self.map(remove_dead, deadpixels=deadpixels,
                         deadvalue=deadvalue, inplace=inplace, *args, **kwargs)
            s.set_experimental_parameters(deadpixels_corrected=True,
                                          deadpixels=deadpixels,
                                          deadvalue=deadvalue)
            return s
        else:  # No dead pixels detected
            pass

    def get_virtual_image(self, roi):
        """Method imported from
        pyXem.ElectronDiffraction.get_virtual_image(self, roi). Obtains a
        virtual image associated with a specified ROI.

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
            roi = hs.roi.RectangularROI(left=10, right=20, top=10, bottom=20)
            s.get_virtual_image(roi)
        """
        return ElectronDiffraction.get_virtual_image(self, roi)

    def plot_interactive_virtual_image(self, roi, **kwargs):
        """Method imported from
        pyXem.ElectronDiffraction.plot_interactive_virtual_image(self, roi).
        Plots an interactive virtual image formed with a specified and
        adjustable roi.

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
            roi = hs.roi.RectangularROI(left=10, right=20, top=10, bottom=20)
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
            Projection angles in degrees. If None (default), the value is set
            to np.arange(180).
        circle : bool, optional
            If True (default), assume that the image is zero outside the
            inscribed circle. The width of each projection then becomes equal
            to the smallest signal shape.
        show_progressbar : bool, optional
            If True (default), show progressbar during transformation.
        inplace : bool, optional
            If True (default is False), the ElectronBackscatterDiffraction
            signal (self) is replaced by the RadonTransform signal (return).

        Returns
        -------
        sinograms: :obj:`ebsp-pro._signals.RadonTransform`
            Corresponding RadonTransform signal (sinograms) computed from
            the ElectronBackscatterDiffraction signal. The rotation axis
            lie at index sinograms.data[0,0].shape[0]/2

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


class LazyEBSD(EBSD, LazySignal2D):

    _lazy = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
