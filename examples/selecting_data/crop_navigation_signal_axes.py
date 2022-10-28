"""
Crop navigation and signal axes
===============================

This example shows how to crop the navigation and signal axes of an
:class:`~kikuchipy.signals.EBSD` signal using HyperSpy's ``inav`` and ``isig``,
respectively (see the `HyperSpy documentation
<https://hyperspy.org/hyperspy-doc/current/user_guide/signal.html#indexing>`__).
"""

import hyperspy.api as hs
import kikuchipy as kp


s = kp.data.nickel_ebsd_large(lazy=True)
print(s)

########################################################################################
# Get a new signal with patterns within a rectangle defined by the upper left pattern
# with index (5, 7) (column, row) and the bottom right pattern with index (17, 23)

s2 = s.inav[5:17, 7:23]
print(s2)

########################################################################################
# Remove the ten outermost pixels of our (60, 60) pixel nickel patterns

s3 = s.isig[10:50, 10:50]

_ = hs.plot.plot_images(
    [s.inav[0, 0], s3.inav[0, 0]],
    axes_decor="off",
    tight_layout=True,
    label=None,
    colorbar=False,
)
