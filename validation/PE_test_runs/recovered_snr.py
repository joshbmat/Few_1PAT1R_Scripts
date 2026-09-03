'''
Recovered SNR from a finished PE run.

Rebuilds the recovery response exactly as `PE_response.py` sets it up
(same waveform config, TDI settings, orbits, timing and Mojito noise
covariance), evaluates the template at the maximum-likelihood sample of the
chain, and returns its optimal SNR

    SNR = sqrt( <h_ML | h_ML> )

with the same covariance-weighted TDI inner product used in the likelihood.
Parameters that were held fixed during sampling (`Sampler.fixed_params`,
plus `x_I0`) are filled in at their injected/true values.

Usage
-----
    from recovered_snr import recovered_snr

    res = recovered_snr(run_dir, discard=5000)
    print(res['snr'])

or from the command line:

    python recovered_snr.py <run_dir> --discard 5000
'''

from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional, Sequence, Tuple

import cupy as cp
import numpy as np
from eryn.backends import HDFBackend
from lisaconstants import ASTRONOMICAL_YEAR
from lisaorbits import OEMOrbits
from mojito import MojitoL1File
from scipy.signal.windows import tukey

from src.io import param_load
from src.noise import build_inv_covariance
from src.utils import inband_freqs, inner_prod_tdi
from src.waveform import (
    ResponseConfig,
    WaveformConfig,
    build_response,
    param_names_for,
)


def _waveform_cfg(block: dict) -> WaveformConfig:
    """Same conversion as PE_response.py so the template is bit-for-bit the same."""
    lmax_raw = block.get("lmax", None)
    return WaveformConfig(
        model=block["model"],
        dt=float(block["dt"]),
        T=float(block["T"]),
        mode_selection_threshold=float(block.get("mode_selection_threshold", 0.0)),
        evolve_chi1=bool(block.get("evolve_chi1", True)),
        include_1PA_amps=bool(block.get("include_1PA_amps", True)),
        inspiral_kwargs=dict(block.get("inspiral_kwargs") or {}),
        amplitude_kwargs=dict(block.get("amplitude_kwargs") or {}),
        summation_kwargs=dict(block.get("summation_kwargs") or {}),
        lmax=int(lmax_raw) if lmax_raw is not None else None,
    )


def _response_cfg(block: dict, orbit_file: str) -> ResponseConfig:
    return ResponseConfig(
        orbit_file=orbit_file,
        tdi_gen=block["tdi_gen"],
        tdi_chan=block["tdi_chan"],
        order=int(block["order"]),
        offset=float(block["offset"]),
        n_samples_delay=int(block["n_samples_delay"]),
        t_buffer=float(block["t_buffer"]),
        flip_hx=bool(block.get("flip_hx", True)),
        is_ecliptic_latitude=bool(block.get("is_ecliptic_latitude", False)),
    )


def _emri_vector(emri_block: dict, model: str) -> List[float]:
    """True parameter vector in the layout of `model`, redshifting M and mu."""
    z = float(emri_block.get("z", 0.0))
    params = []
    for n in param_names_for(model):
        val = float(emri_block[n])
        if n in ("M", "mu"):
            val *= (1.0 + z)
        params.append(val)
    return params


def sampled_indices(param_names: Sequence[str], fixed_names: Sequence[str]) -> List[int]:
    """Indices into `param_names` that were actually sampled (x_I0 is always fixed)."""
    fixed = set(fixed_names) | {"x_I0"}
    return [i for i, n in enumerate(param_names) if n not in fixed]


def max_likelihood_sample(
    h5_path: str,
    discard: int = 0,
    temp: int = 0,
) -> Tuple[np.ndarray, float]:
    """
    Return the (sampled) parameter vector with the highest log-likelihood.

    Parameters
    ----------
    h5_path : Eryn HDF backend file
    discard : burn-in iterations to drop
    temp    : temperature index (0 = cold chain)

    Returns
    -------
    params_ml : ndarray (N_sampled,)
    ll_max    : float, log-likelihood at that sample
    """
    reader = HDFBackend(h5_path, read_only=True)
    chain = reader.get_chain(discard=discard)["model_0"]      # (it, temp, walker, 1, p)
    log_like = reader.get_log_like(discard=discard)           # (it, temp, walker)

    ll = log_like[:, temp, :]
    it, walker = np.unravel_index(np.nanargmax(ll), ll.shape)
    return np.asarray(chain[it, temp, walker, 0, :], dtype=float), float(ll[it, walker])


def recovered_snr(
    run_dir: Optional[str] = None,
    config_path: Optional[str] = None,
    h5_path: Optional[str] = None,
    discard: int = 0,
    temp: int = 0,
    params_ml: Optional[Sequence[float]] = None,
    window_source: str = "template",
    use_gpu: bool = True,
) -> Dict[str, object]:
    """
    Recreate the recovery response and evaluate the SNR of the max-likelihood template.

    Parameters
    ----------
    run_dir       : sampling directory containing the copied `*.yaml` and
                    `SamplingResults_*.h5`.  `config_path` / `h5_path` override
                    the files found there.
    discard       : burn-in iterations to drop before locating the max-likelihood point
    temp          : temperature index of the chain to search (0 = cold)
    params_ml     : optional explicit sampled-parameter vector; when given the
                    chain is not read at all (`h5_path` may then be omitted)
    window_source : how the Tukey window entering the FFT is built --
                    'template'  : from the non-zero extent of the ML template itself
                                  (recovery settings only, no injection needed)
                    'injection' : rebuild the injection response and use its
                                  non-zero extent, exactly as in the PE run
                    'none'      : flat window
    use_gpu       : passed through to `build_response`

    Returns
    -------
    dict with keys
        snr           : float, sqrt(<h_ML|h_ML>)
        params_ml     : full recovery parameter vector (fixed params at true values)
        params_sampled: the sampled subset that was read from the chain
        param_names   : full recovery parameter names
        sampled_names : names of the sampled subset
        log_like_max  : log-likelihood reported by the sampler at that point (or None)
        snr_true      : SNR of the recovery model at the true parameters
        n_samples     : number of time samples in the TDI series
    """
    if window_source not in ("template", "injection", "none"):
        raise ValueError(f"window_source must be template/injection/none, got {window_source}")

    # Locate config and backend
    if config_path is None:
        if run_dir is None:
            raise ValueError("Provide either run_dir or config_path.")
        matches = glob.glob(os.path.join(run_dir, "*.yaml"))
        if not matches:
            raise FileNotFoundError(f"No YAML config found in {run_dir}")
        config_path = matches[0]
    cfg = param_load(config_path)

    if params_ml is None:
        if h5_path is None:
            if run_dir is None:
                raise ValueError("Provide h5_path, run_dir, or params_ml.")
            matches = glob.glob(os.path.join(run_dir, "SamplingResults_*.h5"))
            if not matches:
                raise FileNotFoundError(f"No HDF5 backend found in {run_dir}")
            h5_path = matches[0]
        params_sampled, ll_max = max_likelihood_sample(h5_path, discard=discard, temp=temp)
    else:
        params_sampled, ll_max = np.asarray(params_ml, dtype=float), None

    # Waveform / response settings (identical to PE_response.py)
    inj_wcfg = _waveform_cfg(cfg["Injection"]["Waveform"])
    rec_wcfg = _waveform_cfg(cfg["Recovery"]["Waveform"])
    resp_cfg = _response_cfg(cfg["Response"], cfg["Data"]["orbit_file"])

    param_names = rec_wcfg.param_names()
    fixed_names = list(cfg["Sampler"].get("fixed_params", []) or [])
    idx_sampled = sampled_indices(param_names, fixed_names)
    sampled_names = [param_names[i] for i in idx_sampled]

    if len(params_sampled) != len(idx_sampled):
        raise ValueError(
            f"Chain has {len(params_sampled)} sampled parameters but the config "
            f"implies {len(idx_sampled)} ({sampled_names})."
        )

    # Full vector: start from the truth (this fills x_I0 and every fixed
    # parameter at its injected value) and overwrite the sampled entries.
    inj_truth = _emri_vector(cfg["Injection"]["EMRI"], inj_wcfg.model)
    rec_truth = _emri_vector(cfg["Injection"]["EMRI"], rec_wcfg.model)
    full_params = list(rec_truth)
    for k, i in enumerate(idx_sampled):
        full_params[i] = float(params_sampled[k])
        
    # do the same for the injection parameters
    full_params_inj = list(inj_truth)
    for k, i in enumerate(idx_sampled):
        full_params_inj[i] = float(params_sampled[k])

    # Timing, exactly as in the PE run
    with MojitoL1File(cfg["Data"]["mojito_l1_file"]) as l1:
        ts = l1.tdis.time_sampling
        t0_l1 = float(ts.t0)
        mojito_dt = float(ts.dt)
        central_freq = float(l1.laser_frequency)

    t0_orbits = float(OEMOrbits.from_included("esa-trailing").t_start) + 10.0
    DT = inj_wcfg.dt
    T_response = (inj_wcfg.T
                  + (2 * resp_cfg.offset
                     + 2 * resp_cfg.n_samples_delay * DT) / ASTRONOMICAL_YEAR)
    t_init = t0_l1 - resp_cfg.n_samples_delay * mojito_dt - resp_cfg.offset

    # Recovery response, evaluated at the ML point and at the truth
    inj_response = build_response(inj_wcfg, resp_cfg, t_init, t0_orbits,
                                  T_response, use_gpu=use_gpu)
    rec_response = build_response(rec_wcfg, resp_cfg, t_init, t0_orbits,
                                  T_response, use_gpu=use_gpu)
    xyz_ml = rec_response(*full_params)
    xyz_true = inj_response(*full_params_inj)
    N_t = xyz_ml.shape[1]

    # Window: Tukey over the non-zero extent of the signal, zero beyond it
    if window_source == "injection":
        inj_response = build_response(inj_wcfg, resp_cfg, t_init, t0_orbits,
                                      T_response, use_gpu=use_gpu)
        xyz_win_src = inj_response(*_emri_vector(cfg["Injection"]["EMRI"], inj_wcfg.model))
    else:
        xyz_win_src = xyz_ml

    if bool(cfg["Sampler"]["windowing"]) and window_source != "none":
        nonzero = cp.where(cp.any(xyz_win_src != 0.0, axis=0))[0]
        n_signal = int(nonzero[-1]) + 1 if nonzero.size > 0 else N_t
        window = cp.zeros(N_t)
        window[:n_signal] = cp.asarray(tukey(n_signal, alpha=0.01))
    else:
        n_signal = N_t
        window = cp.ones(N_t)

    # Band-limited FFT and Mojito inverse covariance
    freqs_inband, mask = inband_freqs(N_t, DT, filter_freq=bool(cfg["Sampler"]["filter_freq"]))
    inv_cov, _ = build_inv_covariance(
        cfg["Data"]["noise_file"], central_freq,
        cp.asnumpy(freqs_inband), DT, N_t,
        channels=resp_cfg.tdi_chan,
    )

    ml_fft = cp.fft.rfft(xyz_ml * window, axis=1)[:, mask]
    true_fft = cp.fft.rfft(xyz_true * window, axis=1)[:, mask]

    snr = float(cp.sqrt(inner_prod_tdi(ml_fft, ml_fft, inv_cov)))
    snr_true = float(cp.sqrt(inner_prod_tdi(true_fft, true_fft, inv_cov)))

    return dict(
        snr=snr,
        snr_true=snr_true,
        params_ml=np.array(full_params),
        params_sampled=params_sampled,
        param_names=param_names,
        sampled_names=sampled_names,
        log_like_max=ll_max,
        n_samples=N_t,
        n_signal=n_signal,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=str, help="Sampling run directory.")
    parser.add_argument("--discard", type=int, default=0, help="Burn-in iterations.")
    parser.add_argument("--temp", type=int, default=0, help="Temperature index.")
    parser.add_argument("--window-source", type=str, default="template",
                        choices=["template", "injection", "none"])
    args = parser.parse_args()

    res = recovered_snr(run_dir=args.run_dir, discard=args.discard, temp=args.temp,
                        window_source=args.window_source)

    print("Max-likelihood parameters (fixed params at true values):")
    for name, val in zip(res["param_names"], res["params_ml"]):
        tag = "" if name in res["sampled_names"] else "   (fixed)"
        print(f"  {name:<12} {val:>16.8g}{tag}")
    if res["log_like_max"] is not None:
        print(f"\nloglike at ML point (sampler) : {res['log_like_max']:.6e}")
    print(f"SNR at true parameters        : {res['snr_true']:.2f}")
    print(f"Recovered SNR (ML template)   : {res['snr']:.2f}")
