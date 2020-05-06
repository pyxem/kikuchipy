import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Getting the nice colors
extent = (0, 255)
n = 20
idx = np.linspace(extent[0], extent[1], n, dtype=int)

# winter, cividis, viridis, inferno, magma, plasma, etc
rgb_idx = plt.get_cmap("viridis")(idx)[:, :3]

# Print to sys.out
print([mcolors.to_hex(i) for i in rgb_idx])
