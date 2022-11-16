"""
Pattern binning
===============

This example shows how to bin :class:`~kikuchipy.signals.EBSD` patterns using HyperSpy's
:meth:`~kikuchipy.signals.EBSD.rebin` (see :ref:`hyperspy:rebin-label` for details).
"""

import hyperspy.api as hs
import kikuchipy as kp
import numpy as np


s = kp.data.silicon_ebsd_moving_screen_in(allow_download=True, show_progressbar=False)
s.remove_static_background(show_progressbar=False)

print(s)
print(s.static_background.shape)
print(s.detector)

########################################################################################
# Rebin by passing the new shape (use ``(1, 1, 60, 60)`` if binning a 2D map). Note how
# the :attr:`~kikuchipy.signals.EBSD.static_background` and
# :attr:`~kikuchipy.signals.EBSD.detector` attributes are updated.

s2 = s.rebin(new_shape=(60, 60))

_ = hs.plot.plot_images(
    [s, s2],
    axes_decor="off",
    tight_layout=True,
    label=None,
    colorbar=False,
)
print(s2.static_background.shape)
print(s2.detector)

########################################################################################
# Rebin by passing the new:old pixel ratio (again, use ``(1, 1, 8, 8)`` for a 2D map)

s3 = s.rebin(scale=(8, 8))
print(np.allclose(s2.data, s3.data))
print(s3.static_background.shape)
print(s3.detector)

########################################################################################
# Notice how ``rebin()`` casts the data to ``uint64``, increasing the memory use by a
# factor of eight (so be careful...). Rescale intensities with
# :meth:`~kikuchipy.signals.EBSD.rescale_intensity` if desirable.

print(s.data.dtype)
print(s3.data.dtype)
