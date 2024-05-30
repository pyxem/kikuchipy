"""
Pattern binning
===============

This example shows how to bin :class:`~kikuchipy.signals.EBSD` patterns using
:meth:`~kikuchipy.signals.EBSD.downsample` or HyperSpy's
:meth:`~kikuchipy.signals.EBSD.rebin` (see :ref:`hyperspy:rebin-label` for details).

.. note::

    In general, better contrast is obtained by removing the static (and dynamic)
    background prior to binning instead of after it.
"""

import hyperspy.api as hs

import kikuchipy as kp

s = kp.data.si_ebsd_moving_screen(allow_download=True, show_progressbar=False)
s.remove_static_background(show_progressbar=False)

print(s)
print(s.static_background.shape)
print(s.detector)

# %%
# Downsample by a factor of 8 while maintaining the data type (achieved by rescaling the
# pattern intensity). Note how the :attr:`~kikuchipy.signals.EBSD.static_background` and
# :attr:`~kikuchipy.signals.EBSD.detector` attributes are updated.

s2 = s.downsample(8, inplace=False)
_ = hs.plot.plot_images([s, s2], axes_decor="off", tight_layout=True, label=None)
print(s2.static_background.shape)
print(s2.detector)

# %%
# Rebin by passing the new shape (use ``(1, 1, 60, 60)`` if binning a 2D map). Note how
# the pattern is not rescaled and the data type is cast to either ``int64`` or
# ``float64`` depending on the initial data type.

s3 = s.rebin(new_shape=(60, 60))
print(s3.data.dtype)
print(s3.data.min(), s3.data.max())

# %%
# The latter method is more flexible in that it allows for different binning factors in
# each axis, the factors are not restricted to being integers and the factors do not
# have to be divisors of the initial signal shape.

s4 = s.rebin(scale=(8, 9))
print(s4.data.shape)
