# -*- coding: utf-8 -*-
"""Signal class for radon transform of Electron Backscatter Diffraction (EBSD)
data."""
from hyperspy.signals import Signal2D
from pyxem.signals.electron_diffraction import ElectronDiffraction


class RadonTransform(Signal2D):
    _signal_type = 'radon_transform'

    def __init__(self, *args, **kwargs):
        Signal2D.__init__(self, *args, **kwargs)
        dx = self.axes_manager.signal_axes[0]
        dy = self.axes_manager.signal_axes[1]
        #Projection angle axis
        dx.name = 'theta'
        dx.units = 'degree'
        #Projection position axis
        dy.name = 'projection'
        dy.units = 'pixel'

    def get_virtual_image(self, roi):
        """Method imported from 
        pyXem.ElectronDiffraction.get_virtual_image(self, roi).
        Obtains a virtual image associated with a specified ROI.

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
            roi = hs.roi.CircleROI(cx=10.,cy=10., r_inner=0., r=10.)
            rt.get_virtual_image(roi)

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
            roi = hs.roi.CircleROI(cx=10.,cy=10., r_inner=0., r=10.)
            rt.plot_interactive_virtual_image(roi)
        """
        return ElectronDiffraction.plot_interactive_virtual_image(self, roi,
                                                                  **kwargs)

