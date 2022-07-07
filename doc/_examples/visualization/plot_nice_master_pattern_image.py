"""
==============================
Plot nice master pattern image
==============================

This example shows you how to plot a nice and clean image of an EBSD master pattern.
"""

import kikuchipy as kp
import matplotlib.pyplot as plt
import numpy as np


mp = kp.data.nickel_ebsd_master_pattern_small(hemisphere="both")
print(mp)

########################################################################
# Extract the underlying data of both hemipsheres and mask out the surrounding black
# pixels

data = mp.data.astype("float32")
mask = data[0] == 0
data[:, mask] = np.nan

########################################################################
# Plot both hemispheres with labels

fig, ax = plt.subplots(ncols=2)
ax[0].imshow(data[0], cmap="gray")
ax[1].imshow(data[1], cmap="gray")
ax[0].axis("off")
ax[1].axis("off")
ax[0].set_title("Upper")
ax[1].set_title("Lower")
fig.tight_layout()
