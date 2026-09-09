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

"""Functions for plotting the result of a pattern processing
optimization.
"""

from typing import Any

import matplotlib.figure as mfigure
import matplotlib.pyplot as plt
import numpy as np


def plot_pattern_processing_result(
    result: dict[str, Any],
    reference: np.ndarray | None = None,
    figsize: tuple[float, float] = (15, 6),
) -> mfigure.Figure:
    """Plot the pattern from each stage of a
    :func:`~kikuchipy.pattern.optimize_pattern_processing` result,
    annotated with the image quality and normalized cross-correlation
    at that stage.

    This is a diagnostic plot only; it does not save anything to disk.
    Call :meth:`matplotlib.figure.Figure.savefig` on the returned
    figure to do so.

    Parameters
    ----------
    result
        Dictionary returned by
        :func:`~kikuchipy.pattern.optimize_pattern_processing`.
    reference
        Reference pattern the optimization was scored against. If
        given, it is shown as an extra panel for comparison.
    figsize
        Figure size in inches, passed to
        :func:`matplotlib.pyplot.subplots`.

    Returns
    -------
    fig
        Figure with one column per processing stage (plus one for
        *reference* if given), showing the pattern on top and its
        intensity histogram below. Each pattern's column is titled
        with its stage name, image quality (IQ), and normalized
        cross-correlation (NCC).

    Examples
    --------
    >>> import kikuchipy as kp
    >>> s = kp.data.nickel_ebsd_small(allow_download=True).inav[0, 0]  # doctest: +SKIP
    >>> mp = kp.data.nickel_ebsd_master_pattern_small()  # doctest: +SKIP
    >>> simulated = mp.get_patterns(...)  # doctest: +SKIP
    >>> result = kp.pattern.optimize_pattern_processing(
    ...     s.data, simulated.data
    ... )  # doctest: +SKIP
    >>> fig = kp.draw.plot_pattern_processing_result(
    ...     result, reference=simulated.data
    ... )  # doctest: +SKIP
    """
    stage_titles = {
        "raw": "No processing",
        "dynamic_background": "DBS",
        "ahe": "DBS + AHE",
        "fft": "DBS + AHE + FFT",
    }
    stage_keys = list(stage_titles.keys())
    image_quality = result["image_quality"]
    ncc = result["normalized_cross_correlation"]

    patterns = [result["patterns"][key] for key in stage_keys]
    titles = [
        f"{stage_titles[key]}\nIQ={iq:.3f}, NCC={n:.3f}"
        for key, iq, n in zip(stage_keys, image_quality, ncc)
    ]

    if reference is not None:
        patterns = [reference] + patterns
        titles = ["Simulated"] + titles

    n_patterns = len(patterns)
    fig, axes = plt.subplots(
        nrows=2,
        ncols=n_patterns,
        figsize=figsize,
        gridspec_kw={"height_ratios": [3, 1.5]},
    )

    for ax, pat, title in zip(axes[0], patterns, titles):
        ax.imshow(pat, cmap="gray", vmin=pat.min(), vmax=pat.max())
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    for ax, pat in zip(axes[1], patterns):
        ax.hist(np.asarray(pat).ravel(), bins=100)

    fig.tight_layout()

    return fig
