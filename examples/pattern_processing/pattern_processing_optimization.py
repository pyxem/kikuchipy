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
===============================
Pattern processing optimization
===============================

This example shows how to search for pattern processing parameters that best match a
single experimental pattern to a simulated reference pattern, using Bayesian
optimization from the optional dependency bayesian-optimization or local optimization
with the Nelder-Mead method.

Each candidate is scored by the normalized cross-correlation (NCC) between the processed
and the reference pattern. Three processing steps can be optimized, either one at a time
or together:

* Dynamic background removal:
  :func:`~kikuchipy.pattern.optimize_remove_dynamic_background`
* Adaptive histogram equalization:
  :func:`~kikuchipy.pattern.optimize_adaptive_histogram_equalization`
* FFT bandpass filtering: :func:`~kikuchipy.pattern.optimize_bandpass_filter`
* All of the above, in any order: :func:`~kikuchipy.pattern.optimize_pattern_processing`

The Bayesian optimization of pattern processing parameters was first used in
:cite:`griesbach2026ferroelectric` and :cite:`griesbach2026global` to map ferroelectric
polarization domains.
"""

# %%
# Imports.
import hyperspy.api as hs
import matplotlib.pyplot as plt

import kikuchipy as kp

hs.preferences.General.show_progressbar = False

# %%
# Get experimental patterns and a suitable signal mask, remove the static background,
# and simulate matching reference patterns from the known orientations.
s = kp.data.nickel_ebsd_small()
print(s)

signal_mask = kp.pattern.get_signal_mask(s.data[0, 0])

s.remove_static_background()

mp = kp.data.nickel_ebsd_master_pattern_small(projection="lambert")
sim = mp.get_patterns(
    rotations=s.xmap.rotations.reshape(*s.xmap.shape),
    detector=s.detector,
    energy=20,
    dtype_out="uint8",
    compute=True,
)

# Pick one pattern to optimize the processing for
pattern = s.data[1, 1]
reference = sim.data[1, 1]

# %%
# Optimize the parameters of a single processing step, here the dynamic background
# removal. By default, the standard deviation of the Gaussian window is searched in a
# range relative to the pattern width, while the window truncation is searched in a
# fixed range.
result_dbr = kp.pattern.optimize_remove_dynamic_background(
    pattern,
    reference,
    operation="divide",
    n_calls=20,
    random_state=0,
    signal_mask=signal_mask,
)
print(result_dbr["parameters"]["remove_dynamic_background"])

# %%
# Optimize the parameters of all three steps together, in the given order. By default,
# only dynamic background removal and bandpass filtering are optimized. The
# parameters of each step are passed as a dictionary named after the step. Each
# parameter is either fixed (a single value) or searched (a range or a list of
# candidates). Parameters not given use the defaults of the single step function. Here,
# we fix the number of histogram bins in the adaptive histogram equalization and search
# a custom range of lowpass filter cutoffs.
#
# ``n_calls`` is reduced here to keep this example quick to run; for real work, values
# closer to the default of 150 give the search more room to converge. To follow the
# progress of longer optimizations, set the log level with
# ``kp.set_log_level("INFO")``.
steps = (
    "remove_dynamic_background",
    "adaptive_histogram_equalization",
    "bandpass_filter",
)
result = kp.pattern.optimize_pattern_processing(
    pattern,
    reference,
    steps=steps,
    remove_dynamic_background={"operation": "divide"},
    adaptive_histogram_equalization={"nbins": 256},
    bandpass_filter={"lowpass_cutoff": 23},
    n_calls=40,
    random_state=0,
    signal_mask=signal_mask,
)
print("Best parameters:", result["parameters"])
print(f"Best NCC: {result['score']:.4f}")

# %%
# By default, the parameters are found by Bayesian optimization. Alternatively, a local
# optimization with the Nelder-Mead method can be used, starting from the center of the
# search space and restarting twice from the best parameters so far, slightly perturbed.
# This is typically several times faster, while it might converge to a local maximum.
result_nm = kp.pattern.optimize_pattern_processing(
    pattern,
    reference,
    steps=steps,
    method="nelder-mead",
    remove_dynamic_background={"operation": "divide"},
    adaptive_histogram_equalization={"nbins": 256},
    bandpass_filter={"lowpass_cutoff": 23},
    random_state=0,
    signal_mask=signal_mask,
)
print(f"Best NCC (Nelder-Mead): {result_nm['score']:.4f}")

# %%
# Plot the pattern after each processing step, labeled with its image quality (IQ) and
# NCC against the reference.
fig = kp.draw.plot_pattern_processing_result(result, reference=reference)

# %%
# Apply the optimized parameters to all patterns. The dynamic background removal and
# adaptive histogram equalization parameters can be passed directly to the
# corresponding :class:`~kikuchipy.signals.EBSD` methods, while the bandpass filter
# parameters are passed to :func:`~kikuchipy.filters.bandpass_fft_filter` to create the
# filter for :meth:`~kikuchipy.signals.EBSD.fft_filter`.
params = result["parameters"]

s2 = s.deepcopy()

s2.remove_dynamic_background(**params["remove_dynamic_background"])

s2.adaptive_histogram_equalization(**params["adaptive_histogram_equalization"])

w = kp.filters.bandpass_fft_filter(s2.detector.shape, **params["bandpass_filter"])
s2.fft_filter(transfer_function=w, function_domain="frequency", shift=True)

# %%
# Compare the processed patterns to the simulated ones.
_ = hs.plot.plot_images(
    [
        s2.inav[:3, 0].normalize_intensity(dtype_out="float32", inplace=False)
        * ~signal_mask,
        sim.inav[:3, 0].normalize_intensity(dtype_out="float32", inplace=False)
        * ~signal_mask,
    ],
    per_row=3,
    cmap="gray",
    vmin=-3,
    vmax=3,
    axes_decor="off",
    label=None,
    colorbar=False,
    tight_layout=True,
)
plt.show()
