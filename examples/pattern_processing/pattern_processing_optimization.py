#
# Copyright 2019-2026 the kikuchipy developers
#
# This file is part of kikuchipy.
#
# kikuchipy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# kikuchipy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with kikuchipy. If not, see <http://www.gnu.org/licenses/>.
#
"""
================================
Pattern processing optimization
================================

This example shows how to search for pattern processing parameters that best
match a single experimental pattern to a simulated reference, using
:func:`kikuchipy.pattern.optimize_pattern_processing`.

The search covers dynamic background subtraction, (optional) adaptive
histogram equalization, and FFT bandpass filtering, in that order, scoring
each candidate pipeline by normalized cross-correlation (NCC) against the
reference pattern via Bayesian optimization
(:func:`skopt.gp_minimize`, from the optional dependency
:mod:`scikit-optimize`). The processing steps themselves are all existing
public kikuchipy functionality; only the search over their parameters is new.
"""

# %%
# Imports.
import hyperspy.api as hs
import matplotlib.pyplot as plt

import kikuchipy as kp

hs.preferences.General.show_progressbar = False

# %%
# Get a real experimental pattern and a matching simulated reference for it.
s = kp.data.nickel_ebsd_small()
mp = kp.data.nickel_ebsd_master_pattern_small(projection="lambert")

rotations = s.xmap.rotations.reshape(*s.xmap.shape)
sim = mp.get_patterns(
    rotations=rotations, detector=s.detector, energy=20, dtype_out="uint8", compute=True
)

# Pick one map point to optimize the processing recipe for
i, j = 1, 1
pattern = s.inav[i, j].data.copy()
reference = sim.inav[i, j].data.copy()

# %%
# Run the Bayesian optimization.
#
# ``n_calls``/``n_initial_points`` are reduced here to keep this example
# quick to run; for real work, values closer to the defaults (150/12) give
# the search more room to converge.
result = kp.pattern.optimize_pattern_processing(
    pattern,
    reference,
    n_calls=30,
    n_initial_points=8,
    random_state=0,
)

print("Best parameters:", result["best_parameters"])
print("Best NCC score:", result["best_score"])

# %%
# Plot the pattern at each processing stage, labeled with its image quality
# (IQ) and normalized cross-correlation (NCC) against the reference.
fig = kp.draw.plot_pattern_processing_result(result, reference=reference)
fig.savefig("pattern_processing_optimization_result.png", dpi=100)
plt.show()


# %%
# Apply the optimized parameters to the full pattern stack.
#
# ``optimize_pattern_processing()`` works on a single pattern. To apply the
# chosen parameters to every pattern in the map, pass them to the
# corresponding public :class:`~kikuchipy.signals.EBSD` methods.
params = result["best_parameters"]
pattern_shape = s.axes_manager.signal_shape[::-1]

s2 = s.deepcopy()
s2.remove_dynamic_background(
    operation="subtract",
    filter_domain="frequency",
    std=int(params["dynamic_background_std"]),
    truncate=int(params["dynamic_background_truncate"]),
)
if params["ahe_on"]:
    kernel_size = int(params["ahe_kernel_size"])
    s2.adaptive_histogram_equalization(
        kernel_size=(kernel_size, kernel_size),
        clip_limit=float(params["ahe_clip_limit"]),
        nbins=int(params["ahe_nbins"]),
    )
w_low = kp.filters.Window(
    window="lowpass",
    cutoff=int(params["fft_lowpass_cutoff"]),
    cutoff_width=10,
    shape=pattern_shape,
)
w_high = kp.filters.Window(
    window="highpass",
    cutoff=int(params["fft_highpass_cutoff"]),
    cutoff_width=2,
    shape=pattern_shape,
)
s2.fft_filter(transfer_function=w_low * w_high, function_domain="frequency", shift=True)
