'''
Noise handling: load Mojito XYZ covariance estimates and build an
inverse-covariance matrix on the analysis frequency grid.

Also supports AET recombination via a unitary transform.
'''

from __future__ import annotations

from typing import Literal, Tuple

import cupy as cp
import h5py
import numpy as np
from scipy.interpolate import CubicSpline


# XYZ -> AET orthonormal transform.
_TO_AET = np.array([
    [-1.0 / np.sqrt(2),  0.0,              1.0 / np.sqrt(2)],
    [ 1.0 / np.sqrt(6), -2.0 / np.sqrt(6), 1.0 / np.sqrt(6)],
    [ 1.0 / np.sqrt(3),  1.0 / np.sqrt(3), 1.0 / np.sqrt(3)],
])


def load_mojito_xyz_covariance(
    noise_file: str,
    central_freq: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Opens and reads mojito noise file, with mean forXYZ noise estimates and frequency sampling information
    returns two arrays:
    noise_freqs : (n_noise,) frequency grid
    covariance  : (n_noise, 3, 3) complex covariance for XYZ channels (mean over all time bins)
    """
    with h5py.File(noise_file, "r") as f:
        xyz = np.mean(f["noise_estimates/XYZ"][:], axis=0) / central_freq**2
        attrs = f["noise_estimates/log_frequency_sampling"].attrs
        fmin, fmax, size = attrs["fmin"], attrs["fmax"], attrs["size"]
    noise_freqs = np.logspace(np.log10(fmin), np.log10(fmax), size)
    return noise_freqs, xyz


def _interp_covariance(
    noise_freqs: np.ndarray,
    xyz_cov: np.ndarray,
    freqs_inband: np.ndarray,
) -> np.ndarray:
    """Cubic-spline interpolate the (f, 3, 3) covariance onto freqs_inband."""
    n_f = len(freqs_inband)
    cov = np.zeros((n_f, 3, 3), dtype=complex)

    # now rebuild the covariance matrix on a denser frequency grid with Cubic Splines.
    for i in range(3):
        # build spline
        cov[:, i, i] = CubicSpline(noise_freqs, xyz_cov[:, i, i].real)(freqs_inband)
        for j in range(i + 1, 3):
            # cover imaginary part for CSDs as well
            re = CubicSpline(noise_freqs, xyz_cov[:, i, j].real)(freqs_inband)
            im = CubicSpline(noise_freqs, xyz_cov[:, i, j].imag)(freqs_inband)
            cov[:, i, j] = re + 1j * im

            # exploit hermitian symmetry
            cov[:, j, i] = np.conj(cov[:, i, j])
    return cov


def build_inv_covariance(
    noise_file: str,
    central_freq: float,
    freqs_inband: np.ndarray,
    dt: float,
    n_samples: int,
    channels: Literal["XYZ", "AET"] = "XYZ",
) -> Tuple[cp.ndarray, np.ndarray]:
    """
    Build the inverse-covariance matrix on `freqs_inband` with the standard
    2 dt / N likelihood normalization applied.

    Returns
    -------
    inv_cov : cupy (n_f, 3, 3) complex inverse covariance (likelihood-normalized)
    psd_diag : numpy (3, n_f) real diagonal PSDs (for plotting / SNR sanity)
    """
    noise_freqs, xyz_cov = load_mojito_xyz_covariance(noise_file, central_freq)
    cov_xyz = _interp_covariance(noise_freqs, xyz_cov, np.asarray(freqs_inband))

    if channels == "AET":
        cov = np.einsum("ij,fjk,lk->fil", U, cov_xyz, _TO_AET)
    else:
        cov = cov_xyz

    # invert and apply correct normalization from FFT conventions
    inv = np.linalg.inv(cov)
    inv *= 2.0 * dt / n_samples  # likelihood normalization: 2 df with df = 1/(N dt)

    psd_diag = np.stack([cov[:, i, i].real for i in range(3)], axis=0)
    return cp.asarray(inv), psd_diag
