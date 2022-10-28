"""
=============================
Dynamic background correction
=============================

This example shows how to remove the dynamic background of an EBSD pattern using
:meth:`~kikuchipy.signals.EBSD.remove_dynamic_background`.

More details are given in the
:doc:`pattern processing tutorial </tutorials/pattern_processing>`.
"""

import matplotlib.pyplot as plt
import kikuchipy as kp


# Load low resolution Ni patterns and check that the *static* background
# pattern is stored with the signal
s = kp.data.nickel_ebsd_small()
print(s.static_background)

# Keep original for comparison and remove static and dynamic background
s.remove_static_background()
s2 = s.deepcopy()
s2.remove_dynamic_background()

# Plot pattern before and after correction and the intensity histograms
patterns = [s.inav[0, 0].data, s2.inav[0, 0].data]
fig, axes = plt.subplots(2, 2, height_ratios=[3, 1.5])
for ax, pattern, title in zip(axes[0], patterns, ["Static", "Dynamic"]):
    ax.imshow(pattern, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
for ax, pattern in zip(axes[1], patterns):
    ax.hist(pattern.ravel(), bins=100)
fig.tight_layout()
