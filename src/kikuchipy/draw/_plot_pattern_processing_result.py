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

from kikuchipy.pattern._processing_optimization import _STEPS


def plot_pattern_processing_result(
    result: dict[str, Any],
    reference: np.ndarray | None = None,
    figsize: tuple[float, float] | None = None,
) -> mfigure.Figure:
    """Plot the pattern before and after each processing step of a
    pattern processing optimization result, annotated with the image
    quality and normalized cross-correlation after that step.

    Parameters
    ----------
    result
        Dictionary returned by
        :func:`~kikuchipy.pattern.optimize_pattern_processing` or one of
        the single step optimization functions, like
        :func:`~kikuchipy.pattern.optimize_remove_dynamic_background`.
    reference
        Reference pattern the optimization was scored against. If given,
        it is shown as an extra panel for comparison.
    figsize
        Figure size in inches, passed to
        :func:`matplotlib.pyplot.subplots`. If not given, it is set
        based on the number of patterns.

    Returns
    -------
    fig
        Figure with one column per pattern, showing the pattern on top
        and its intensity histogram below. Each pattern is titled with
        the processing steps applied, its image quality (IQ), and its
        normalized cross-correlation (NCC). Pixels masked out by the
        result's signal mask are not shown and not included in the
        histograms.
    """
    labels = []
    for step in result["steps"]:
        label = _STEPS[step].label
        labels.append(label if not labels else f"{labels[-1]} + {label}")
    labels = ["Raw"] + labels

    patterns = list(result["patterns"].values())
    titles = []
    for label, iq, ncc in zip(
        labels, result["image_quality"], result["normalized_cross_correlation"]
    ):
        titles.append(f"{label}\nIQ={iq:.3f}, NCC={ncc:.4f}")
    if reference is not None:
        patterns = [reference] + patterns
        titles = ["Reference"] + titles

    signal_mask = result.get("signal_mask")
    if signal_mask is None:
        signal_mask = np.zeros(patterns[0].shape, dtype=bool)

    n_patterns = len(patterns)
    if figsize is None:
        figsize = (3 * n_patterns, 4.5)
    fig, axes = plt.subplots(
        nrows=2,
        ncols=n_patterns,
        figsize=figsize,
        squeeze=False,
        gridspec_kw={"height_ratios": [3, 1.5]},
    )

    for ax_pattern, ax_hist, pattern, title in zip(*axes, patterns, titles):
        pattern = np.ma.masked_array(pattern, mask=signal_mask)
        ax_pattern.imshow(pattern, cmap="gray")
        ax_pattern.set_title(title, fontsize=9)
        ax_pattern.axis("off")
        ax_hist.hist(pattern.compressed(), bins=100)

    fig.tight_layout()

    return fig
