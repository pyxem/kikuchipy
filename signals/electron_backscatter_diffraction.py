# -*- coding: utf-8 -*-
"""Signal class for Electron Backscatter Diffraction (EBSD) data."""
import numpy as np

from hyperspy.api import plot
from hyperspy.signals import Signal2D, BaseSignal
from skimage.exposure import rescale_intensity
from skimage.transform import radon
import warnings
from scipy.ndimage import gaussian_filter, median_filter
from matplotlib.pyplot import imread
from pyxem.signals.electron_diffraction import ElectronDiffraction
import radon_transform

class ElectronBackscatterDiffraction(Signal2D):
    _signal_type = 'electron_backscatter_diffraction'

    def __init__(self, *args, **kwargs):
        Signal2D.__init__(self, *args, **kwargs)
        self.set_experimental_parameters()

    def set_experimental_parameters(self, 
    								accelerating_voltage = None,
    								condenser_aperture = None,
    								deadpixels_corrected = False,
    								working_distance = None,
                                    deadpixels = None, deadvalue = None,
                                    frame_rate = None,
                                    ):
        """Set experimental parameters in metadata.

        Parameters
        ----------
        accelerating_voltage : float
        	Accelerating voltage in kV.
        condenser_aperture : float
        	Condenser_aperture in µm.
        working_distance . float
        	Working distance in mm.
        deadpixels_corrected : bool
            If True (default is False), deadpixels in patterns are corrected.
        deadpixels : list of tuple
            List of tuples containing pattern indices for dead pixels.
        deadvalue : string
            Specifies how dead pixels have been corrected for (average or nan).
		frame_rate : float
        	Frame rate in fps.
		exposure_time : float
        	Exposure time in µs.	
        """
        # TODO: Fetch some of these directly from the settings.txt file for nordif2hdf5?

        md = self.metadata
        if accelerating_voltage is not None:
            md.set_item('Acquisition_instrument.SEM.accelerating_voltage',
                        accelerating_voltage)
        if condenser_aperture is not None:
            md.set_item('Acquisition_instrument.SEM.condenser_aperture',
                        condenser_aperture)
        if working_distance is not None:
            md.set_item('Acquisition_instrument.SEM.working_distance',
                        working_distance)
        md.set_item('Acquisition_instrument.SEM.Detector.deadpixels_corrected',
                    deadpixels_corrected)
        md.set_item('Acquisition_instrument.SEM.Detector.deadpixels',
                    deadpixels)
        md.set_item('Acquisition_instrument.SEM.Detector.deadvalue',
                    deadvalue)
        if frame_rate is not None:
            md.set_item('Acquisition_instrument.SEM.Detector.Diffraction.frame_rate',
                frame_rate)
        if exposure_time is not None:
            md.set_item('Acquisition_instrument.SEM.Detector.Diffraction.exposure_time', exposure_time)
    
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
            Diffraction pattern calibration in reciprocal Angstroms per pixel.
        """
        ElectronDiffraction.set_diffraction_calibration(self, calibration)
        self.axes_manager.signal_axes[0].offset = 0
        self.axes_manager.signal_axes[1].offset = 0

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

 	def get_virtual_image(self, roi):
 		"""Method imported from pyXem.ElectronDiffraction.get_virtual_image(self, roi).
 		Obtains a virtual image associated with a specified ROI.

        Parameters
        ----------
        roi: :obj:`hyperspy.roi.BaseInteractiveROI`
            Any interactive ROI detailed in HyperSpy.

        Returns
        -------
        dark_field_sum: :obj:`hyperspy.signals.BaseSignal`
            The virtual image signal associated with the specified roi.

        Examples
        --------
        .. code-block:: python

        	import hyperspy.api as hs
            roi = hs.roi.RectangularROI(left = 10, right = 20, top= 10, bottom = 20)
            s.get_virtual_image(roi)

        """
        return ElectronDiffraction.get_virtual_image(self, roi)
    
    def plot_interactive_virtual_image(self, roi):
    	"""Method imported from pyXem.ElectronDiffraction.plot_interactive_virtual_image(self, roi).
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
            roi = hs.roi.RectangularROI(left = 10, right = 20, top= 10, bottom = 20)
            s.plot_interactive_virtual_image(roi)
		"""
        return ElectronDiffraction.plot_interactive_virtual_image(self, roi)
    
    def get_radon_transform(self, 
                        theta = None, circle = True, 
                        show_progressbar = True, inplace = False):
        '''
		Create a RadonTransform signal 

		Parameters
        ----------
        theta : :obj:`hyperspy.roi.BaseInteractiveROI`
            Projection angles in degrees. 
            If None (defualt), the value is set to np.arange(180).
		circle : bool
			If True (default), assume that the image is zero outside the inscribed circle.
			The width of each projection then becomes equal to the smallest signal shape.
        
        Returns
        -------
        sinograms: :obj:`ebsp-pro.signals.RadonTransform`
            Corresponding radon transform 2d signal (sinograms) computed from 
            the electron backscatter diffrcation 2d signal. The rotation axis
            lie at index sinograms.data[0,0].shape[0]/2

        References 
        --------
        http://scikit-image.org/docs/dev/auto_examples/transform/plot_radon_transform.html
        http://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.radon
        
        '''
        #TODO1: Remove diagonal articfact lines.
        #TODO2: Can we get it to work faster?
        
        warnings.filterwarnings("ignore", message="The default of `circle` in `skimage.transform.radon` will change to `True` in version 0.15.")
        # Ignore this warning, since this function is mapped and would otherwise slow 
        # the function down by prininting many warning messages.

        sinograms = self.map(radon, 
        					 theta = theta, circle = circle,
                             show_progressbar = show_progressbar, inplace = inplace)
        
        return RadonTransform(sinograms)