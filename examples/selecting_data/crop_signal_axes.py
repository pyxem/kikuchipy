"""
Crop signal axes
================

This example shows various ways to crop the signal axes of an
:class:`~kikuchipy.signals.EBSD` signal using HyperSpy's ``isig`` slicer and the
:meth:`~kikuchipy.signals.EBSD.crop` and
:meth:`~hyperspy._signals.signal2d.Signal2D.crop_signal` methods (see
:ref:`hyperspy:signal.indexing` for details).
"""

import hyperspy.api as hs

import kikuchipy as kp

# Import data
s = kp.data.nickel_ebsd_small()
s.remove_static_background(show_progressbar=False)

# Inspect data and attributes
plot_kwds = dict(axes_decor=None, label=None, colorbar=None, tight_layout=True)
_ = hs.plot.plot_images(s, **plot_kwds)
print(s)
print(s.static_background.shape)
print(s.detector)

# %%
# Get a new signal, removing the first and last ten rows of pixels and first and last
# five columns of pixels. Note how the :attr:`~kikuchipy.signals.EBSD.static_background`
# and :attr:`~kikuchipy.signals.EBSD.detector` attributes are updated.

s2 = s.isig[5:55, 10:50]

_ = hs.plot.plot_images(s2, **plot_kwds)
print(s2)
print(s2.static_background.shape)
print(s2.detector)

# %%
# Do the same inplace using :meth:`~kikuchipy.signals.EBSD.crop`

s3 = s.deepcopy()
s3.crop(2, start=5, end=55)
s3.crop("dy", start=10, end=50)

_ = hs.plot.plot_images(s3, **plot_kwds)
print(s3)
print(s3.static_background.shape)
print(s3.detector)

# %%
# Do the same inplace using ``crop_signal()``

s4 = s.deepcopy()
s4.crop_signal(top=10, bottom=50, left=5, right=55)

_ = hs.plot.plot_images(s4, **plot_kwds)
print(s4)
print(s4.static_background.shape)
print(s4.detector)
