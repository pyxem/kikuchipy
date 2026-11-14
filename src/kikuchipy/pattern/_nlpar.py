# Copyright 2019-2024 The kikuchipy developers
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

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import scipy.optimize as opt

from kikuchipy._constants import verify_dependency_or_raise
from kikuchipy.filters.window import Window
from kikuchipy.pattern.chunk import _rescale_neighbour_averaged_patterns


def _estimate_sigma_nlpar(
    patterns: np.ndarray,
    signal_mask: np.ndarray,
) -> np.ndarray:
    """Estimate patterns' approximate standard deviation (sigma) of the
    noise.

    If :mod:`pyebsdindex` is installed, the
    :func:`pyebsdindex.nlpar.NLPAR.sigma_numba` is run to calculate the
    patterns' standard deviation of the noise. If not, the patterns'
    standard deviation is calculated and returned.

    See docstring of :func:
    `kikuchipy.signals.ebsd.average_non_local_neighbour_patterns`.

    Parameters
    ----------
    patterns
        EBSD patterns.
    signal_mask
        A boolean mask of detector pixels to consider in the
        calculations. (equal to ``False``, i.e. pixels to *mask
        out* are ``True``.) The mask must be equal to the signal's
        signal shape. If not given all pixels are used.

    Returns
    -------
    sigma
        Patterns' standard deviation (of the noise).
    """
    if verify_dependency_or_raise("pyebsdindex", "Sigma estimation") is None:
        from pyebsdindex import nlpar

        sigma_numba = getattr(nlpar.NLPAR(), "sigma_numba")

        pats = np.asarray(patterns, dtype=np.float32)

        ndim = np.ndim(pats)
        if ndim == 3:
            nrows, ncols = pats.shape[0], 1
        else:
            nrows, ncols = pats.shape[:2]
        h, w = pats.shape[-2:]
        npix = h * w

        if signal_mask is not None:
            if signal_mask.shape != (h, w):
                raise ValueError(
                    "The shape of 'signal_mask' must match the signal shape."
                )
        elif signal_mask is None:
            signal_mask = np.zeros((h, w), dtype=bool)

        indices = np.arange(npix) * ~signal_mask.flatten()

        # Line 754: sigma, corr. accumulated squared difference, and
        # norm. distances for lamda opt.:
        sigma, _d2, _n2 = sigma_numba(
            pats.reshape((nrows * ncols,) + (-1,)).copy(),
            nn=1,  # search_radius
            nrows=nrows,
            ncols=ncols,
            rowstartcount=np.asarray((0, nrows)),
            colstartcount=np.asarray((0, ncols)),
            indices=indices,
            saturation_protect=True,
        )
    else:
        # A crude estimate:
        sigma = np.std(patterns, axis=(-2, -1), astype=np.float32)

    return sigma.astype(np.float32)


def _optimise_lambda(
    patterns: np.ndarray,
    search_radius: int = 1,
    dthresh: float = 0.0,
    signal_mask: np.ndarray | None = None,
    target_weights: tuple | list = (0.5, 0.34, 0.25),
) -> np.ndarray:
    """This function is essentially the same as
    :func:`pyebsdindex.nlpar_cpu.opt_lambda_cpu`, but slightly modified
    to enable optimisation of lambda through
    :func:`average_non_local_neighbour_patterns`.

    Parameters
    ----------
    patterns
        EBSD patterns.
    search_radius
        Pattern nearest neighbour search radius.
    dthresh
        Distance threshold used during NLPAR weighting to
        suppress neighbours that are too dissimilar.
    signal_mask
        A boolean mask of detector pixels to consider in the
        calculations. (equal to ``False``, i.e. pixels to *mask
        out* are ``True``.) The mask must be equal to the signal's
        signal shape. If not given all pixels are used.'
    target_weights
        Target average weights (over all points in the EBSD scan)
        that corresponds to an optimised lambda.

    Returns
    -------
    lamopt_values
        Array of optimised lamda corresponding to the target weights.
    """
    # C.f. line 94:
    # will accept all keywords to calcsigma_cpu.
    # See NLPAR __init__ for default values
    from pyebsdindex import nlpar

    patterns = np.asarray(patterns, dtype=np.float32)

    ny, nx, sy, sx = patterns.shape

    target_weights = np.asarray(target_weights)
    dthresh = np.float64(dthresh)

    def loptfunc(lam, d2, tw, dthresh):
        temp = np.maximum(d2, dthresh)
        dw = np.exp(-(temp) / lam**2)
        w = np.sum(dw, axis=2) + 1e-12
        metric = np.mean(np.abs(tw - 1.0 / w))
        return metric

    nrows = np.uint64(nx)
    ncols = np.uint64(ny)

    nn = search_radius
    nn = np.uint64(nn)

    sigma_numba = getattr(nlpar.NLPAR(), "sigma_numba")

    if signal_mask is None:
        signal_mask = np.zeros((sy, sx), bool)

    indices = np.arange(sy * sx) * ~signal_mask.flatten()

    # C.f. lines 266 & 486:
    # https://github.com/USNavalResearchLaboratory/PyEBSDIndex/blob/main/pyebsdindex/nlpar_cpu.py
    sigma, d2, n2 = sigma_numba(
        patterns.reshape((ny * nx, sy * sx)).copy(),
        nn=search_radius,
        nrows=nrows,
        ncols=ncols,
        rowstartcount=np.asarray((0, ny)),
        colstartcount=np.asarray((0, nx)),
        indices=indices,
    )

    lamopt_values = []
    stride = (
        1 if sigma.size < 1e6 else 2
    )  # for large scans cut down on the optimization time.
    for tw in target_weights:
        lam = 1.0
        lambopt1 = opt.minimize(
            loptfunc,
            lam,
            args=(d2[0::stride, 0::stride, :], tw, dthresh),
            method="Nelder-Mead",
            bounds=[[0.001, 10.0]],
            options={"fatol": 0.0001},
        )

        lamopt_values.append(lambopt1["x"])

    print("", end="")
    lamopt_values = np.asarray(lamopt_values)

    print("With target weights", target_weights)
    print("The optimised lambda values are", lamopt_values.flatten())
    print("Optimal choice (median): ", np.median(lamopt_values))

    return lamopt_values.flatten()


def _average_non_local_neighbour_patterns(
    patterns: np.ndarray,
    sigma: np.ndarray,
    lamda: int | float,
    window: np.ndarray | Window,
    signal_mask: np.ndarray,
    dtype_out: np.dtype,
    omin: int | float,
    omax: int | float,
) -> np.ndarray:
    """See docstring of :func:`average_non_local_neighbour_patterns`."""

    pats = patterns.astype("float32")
    nrows, ncols, h, w = pats.shape
    win_shape = window.shape[:2]

    sr = tuple(  # ((-row,+row), (-col,+col))
        (
            size // 2,
            size - 1 - (size // 2),
        )
        for size in win_shape
    )

    # Patterns of interest within the search window:
    win_mask = np.asarray(window[:, :, 0, 0], dtype=bool).ravel()

    # Center patterns | (nrows, ncols, h, w) -> (nrows, ncols, h*w[mask])
    center = pats.reshape(nrows, ncols, h * w)[:, :, ~signal_mask.ravel()]

    # Tot. num. of pixels of interest per pattern
    npix = center.shape[-1]

    # Mapping neighbour patterns and sigmas:
    padded_pat = np.pad(
        array=center,
        pad_width=(sr[0], sr[1], (0, 0)),
        mode="edge",
    )

    # Mapping neighbours
    neigh = sliding_window_view(
        padded_pat,
        win_shape,
        axis=(0, 1),
    )

    neigh = np.moveaxis(
        a=neigh, source=(-2, -1), destination=(2, 3)
    ).reshape(  # (..., sr[col], sr[row], npix) -> (..., sr[col]*sr[row][window], npix)
        (nrows, ncols, np.prod(win_shape), npix)
    )

    neigh = neigh[:, :, win_mask, :]

    padded_sigma = np.pad(
        array=sigma,
        pad_width=(sr[0], sr[1], (0, 0), (0, 0)),
        mode="edge",
    )

    neigh_sigma = sliding_window_view(
        padded_sigma,
        win_shape,
        axis=(0, 1),
    )

    neigh_sigma = neigh_sigma.reshape(
        nrows,
        ncols,
        np.prod(win_shape),
    )[:, :, win_mask]

    # Expand center pattern dimensions: (nrows, ncols, 1, npix)
    center = center[:, :, None, :]

    # Euclidean distance | Eq. 3
    diff = center - neigh

    d2 = np.sum(diff * diff, axis=-1, dtype=np.float32)

    # Sum of sigmas squared | c.f. Eq. 7
    sig2 = (sigma[:, :, 0, 0, None] ** 2) + (neigh_sigma**2)

    # Normalised distance | Eq. 7
    num = d2 - (npix * sig2)
    denom = sig2 * np.sqrt(2.0 * npix)

    dist = np.zeros_like(d2, dtype=np.float32)

    valid = denom > 1e-8
    np.divide(num, denom, out=dist, where=valid)

    dist[~valid] = np.float32(1.0e6) * npix

    center_flat = (  # int
        (win_shape[0] // 2) * win_shape[1] + (win_shape[1] // 2)
    )

    mask_indices = np.flatnonzero(win_mask)
    center_matches = np.where(mask_indices == center_flat)[0]
    if center_matches.size:
        center_idx = center_matches[0]
        dist[:, :, center_idx] = np.float32(-1.0e6)

    # Weights | Eq. 8
    weights = np.maximum(dist, 0.0, dtype=np.float32)
    weights = np.exp(-weights / (np.float32(lamda) ** 2)).astype(np.float32)

    weights /= weights.sum(
        axis=2,
        keepdims=True,
        dtype=np.float32,
    )

    pats[:, :, ~signal_mask] = np.sum(
        neigh * weights[..., None],
        axis=2,
        dtype=np.float32,
    )

    rescaled_patterns = _rescale_neighbour_averaged_patterns(
        patterns=pats,
        window_sums=np.ones((nrows, ncols)),
        dtype_out=dtype_out,
        omin=omin,
        omax=omax,
    )

    return rescaled_patterns
