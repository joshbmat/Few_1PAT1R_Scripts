'''
Utility functions for PE validation: zero padding, inner products,
mismatches, and band-limited FFT.
'''

from typing import Tuple, Union

import cupy as cp
import numpy as np

ArrayLike = Union[np.ndarray, cp.ndarray]


def zero_pad(data: ArrayLike) -> ArrayLike:
    """Zero-pad a 1D signal to the next power of two."""
    xp = cp.get_array_module(data)
    N = len(data)
    pow_2 = xp.ceil(xp.log2(N))
    return xp.pad(data, (0, int((2**pow_2) - N)), "constant")


def fft_inband(
    x_td: cp.ndarray,
    window: cp.ndarray,
    mask: cp.ndarray,
    axis: int = -1,
) -> cp.ndarray:
    """Window + rFFT + band-limit along `axis`."""
    return cp.fft.rfft(x_td * window, axis=axis).take(cp.where(mask)[0], axis=axis)


def inband_freqs(
    n_samples: int,
    dt: float,
    f_min: float = 1e-5,
    f_max: float | None = None,
    filter_freq: bool = True,
) -> Tuple[cp.ndarray, cp.ndarray]:
    """Return (freqs_inband, mask) for an rFFT of length n_samples."""
    freqs = cp.fft.rfftfreq(n_samples, d=dt)
    if filter_freq:
        fmax = 1.0 / (2.0 * dt) if f_max is None else f_max
        mask = (freqs >= f_min) & (freqs <= fmax)
    else:
        mask = cp.ones_like(freqs, dtype=bool)
    return freqs[mask], mask


def inner_prod_tdi(
    a_fft: cp.ndarray,
    b_fft: cp.ndarray,
    inv_cov: cp.ndarray,
) -> cp.ndarray:
    """
    Covariance-weighted inner product for multi-channel TDI data.

    Parameters
    ----------
    a_fft, b_fft : (n_chan, n_f) complex
    inv_cov      : (n_f, n_chan, n_chan) complex, already carrying the
                   2 df = 2 / (N dt) normalization.
    """
    per_freq = cp.einsum("fj,fjk,fk->f", cp.conj(a_fft.T), inv_cov, b_fft.T)
    return 2.0 * cp.real(cp.sum(per_freq))


def mismatch_tdi(
    a_fft: cp.ndarray,
    b_fft: cp.ndarray,
    inv_cov: cp.ndarray,
) -> float:
    """Standard overlap-based mismatch 1 - <a|b> / sqrt(<a|a><b|b>)."""
    ab = inner_prod_tdi(a_fft, b_fft, inv_cov)
    aa = inner_prod_tdi(a_fft, a_fft, inv_cov)
    bb = inner_prod_tdi(b_fft, b_fft, inv_cov)
    return float(1.0 - cp.abs(ab) / cp.sqrt(aa * bb))


def inner_product(
    h: ArrayLike,
    d: ArrayLike,
    psd: ArrayLike,
    df: float,
    dt: float,
) -> Union[float, cp.ndarray]:
    """Whittle inner product for a single channel with diagonal PSD."""
    xp = cp.get_array_module(h)
    assert len(h) == len(d) == len(psd), "Signals and PSD must match in length"
    result = 4.0 * dt**2 * xp.real(xp.sum((h * d.conj()) / psd * df))
    if xp is cp:
        return result.get() if hasattr(result, "get") else result
    return result
