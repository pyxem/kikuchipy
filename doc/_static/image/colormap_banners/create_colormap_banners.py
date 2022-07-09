# Colormap banners used on the doc landing page

import matplotlib as mpl
import matplotlib.pyplot as plt


plt.rcParams["backend"] = "agg"

cmaps = [mpl.cm.plasma, mpl.cm.viridis, mpl.cm.cool]

for i, cmap in enumerate(cmaps):
    fig, ax = plt.subplots(figsize=(15, 1))
    fig.subplots_adjust(bottom=0.5)
    norm = mpl.colors.Normalize(vmin=5, vmax=10)
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation="horizontal",
    )
    cbar.ax.set_axis_off()
    cbar.ax.margins(0, 0)
    fig.savefig(
        f"/home/hakon/kode/kikuchipy/doc/_static/image/colormap_banners/banner{i}.png",
        pad_inches=0,
        bbox_inches="tight",
    )
    i += 1
