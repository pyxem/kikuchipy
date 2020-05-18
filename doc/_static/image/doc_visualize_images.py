import os
import logging
import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np
import skimage.exposure as ske

import kikuchipy as kp

logging.getLogger("kikuchipy").setLevel(logging.DEBUG)

data = "/home/hakon/phd/data/ni/190506_ebsd/1/nordif/Pattern.dat"
data_mp = "/home/hakon/kode/emsoft/emdata/crystal_data/ni/ni_mc_mp_20kv.h5"
imgdir = "/home/hakon/kode/kikuchipy/doc/source/_static/image"
visdir = os.path.join(imgdir, "visualizing_patterns")

datadir, fname = os.path.split(data)
fname, ext = os.path.splitext(fname)
s = kp.load(data, lazy=False)

# VBSE
roi = hs.roi.RectangularROI(left=18, top=20, right=23, bottom=25)
s.virtual_bse_imaging(roi)
s._plot.navigator_plot.figure.savefig(
    os.path.join(visdir, "standard_navigator.jpg")
)
s._plot.signal_plot.figure.savefig(os.path.join(visdir, "pattern_roi.jpg"))

# Virtual image
vbse = s.get_virtual_bse_intensity(roi)
s.plot(navigator=vbse)
s._plot.navigator_plot.figure.savefig(
    os.path.join(visdir, "vbse_navigator.jpg")
)

# Quality metric map
osm = plt.imread(os.path.join(visdir, "orientation_similarity_map.png"))
s_osm = hs.signals.Signal2D(osm)
s_osm = s_osm.rebin(s.axes_manager.navigation_shape)
s.plot(navigator=s_osm)
s._plot.navigator_plot.figure.savefig(
    os.path.join(visdir, "orientation_similarity_map_navigator.jpg")
)

# Orientation map
om = plt.imread(os.path.join(visdir, "orientation_map.jpg"))
om_scaled = ske.rescale_intensity(om, out_range=np.uint8)
s_om = hs.signals.Signal2D(om_scaled)
s_om = s_om.transpose(signal_axes=1)
s_om.change_dtype("rgb8")
s.plot(navigator=s_om)
s._plot.navigator_plot.figure.savefig(
    os.path.join(visdir, "orientation_map_navigator.jpg")
)

# Plot simulated and experimental side-by-side
s.static_background_correction()
s.dynamic_background_correction()
s_sim = kp.load(os.path.join(datadir, "../emsoft/orig/ni_emebsd2.h5"))
s.plot()
s._plot.signal_plot.figure.savefig(os.path.join(visdir, "pattern.jpg"))
s_sim.plot()
s_sim._plot.signal_plot.figure.savefig(
    os.path.join(visdir, "simulated_pattern.jpg")
)

plt.close("all")
