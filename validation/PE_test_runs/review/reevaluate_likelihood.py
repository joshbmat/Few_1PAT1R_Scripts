#!/usr/bin/env python
"""
Re-evaluate the q = 1e-4 (eps4) posterior samples with the current FEW, after the fix of the
delta-m contribution in the 1PAT1R trajectory.

For every run the likelihood is rebuilt exactly as ``PE_response.py`` builds it, from the config
that was copied into the run directory and with the same ``src`` modules (``src/waveform.py``,
not ``src/waveform_updated.py``): Mojito L1 timing, OEM orbit t0, response buffer, Tukey window up
to the end of the injected signal, in-band mask, Mojito XYZ inverse covariance, and fixed
parameters.  ``build_response`` in ``src/waveform.py`` is unchanged since the runs were made
(May 2026), so the response gets the same settings; ``check_settings`` compares them against the
run's ``RunLog`` as well.

Note that the injection is generated on the fly from the ``Injection`` block, so the fix changes
the data as well as the template.

The effective samples are the cold chain after the burn-in ``discard``, thinned by the maximum
integrated auto-correlation time along the iteration axis.

Run on a GPU node from this directory, e.g.

    python reevaluate_likelihood.py --runs full evol_off --max-samples 2000

or from a notebook:

    import reevaluate_likelihood as rl
    res = rl.reevaluate('evol_off', max_samples=500)
"""

import argparse
import glob
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

PE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PE_DIR))

import cupy as cp
from eryn.backends import HDFBackend
from lisaconstants import ASTRONOMICAL_YEAR
from lisaorbits import OEMOrbits
from mojito import MojitoL1File
from scipy.signal.windows import tukey

from src.io import param_load
from src.likelihood import LogLikelihood, RecoveryConfig
from src.noise import build_inv_covariance
from src.priors import build_priors
from src.utils import inband_freqs, inner_prod_tdi, mismatch_tdi
from src.waveform import (
    ResponseConfig,
    WaveformConfig,
    build_response,
    param_names_for,
)

SAMPLING_DIR = PE_DIR / "sampling_data"
OUT_DIR = Path("reeval_out").resolve()

# q = 1e-4 runs with the primary spin sampled (the later '_nospin' runs fix it), and the burn-in
# used for them in analyse_samples.ipynb.  The 'full' run is the continuation of
# test_inj_1PA_rec_1PA_eps4_Ollie_2026-05-11_10-52-27, so its backend holds only the extension.
RUNS = {
    "full":     ("test_inj_1PA_rec_1PA_eps4_Ollie_extended_2026-05-12_18-14-15", 1800),
    "evol_off": ("test_inj_1PA_rec_1PA_eps4_Ollie_evol_2026-05-14_22-34-43", 1100),
    "amps_off": ("test_inj_1PA_rec_1PA_eps4_Ollie_amps_2026-05-14_21-37-31", 1900),
    "0PA":      ("test_inj_1PA_rec_0PA_eps4_Ollie_2026-05-15_02-33-23", 3140),
}


# ---- config -> dataclasses, copied from PE_response.py ------------------------------------

def _waveform_cfg(block: dict) -> WaveformConfig:
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


def _emri_vector(emri_block: dict, model: str) -> list[float]:
    z = float(emri_block.get("z", 0.0))
    params = []
    for n in param_names_for(model):
        val = float(emri_block[n])
        if n in ("M", "mu"):
            val *= (1.0 + z)
        params.append(val)
    return params


# ---- run directory ------------------------------------------------------------------------

def load_run(label: str, sampling_dir: Path = SAMPLING_DIR) -> dict:
    """Config, backend and RunLog of one of the RUNS."""
    dirname, discard = RUNS[label]
    run_dir = Path(sampling_dir) / dirname

    def _one(pattern):
        hits = glob.glob(str(run_dir / pattern))
        if len(hits) != 1:
            raise FileNotFoundError(f"expected one {pattern} in {run_dir}, found {hits}")
        return hits[0]

    config_path = _one("*.yaml")
    return dict(
        label=label,
        run_dir=run_dir,
        discard=discard,
        config_path=config_path,
        config=param_load(config_path),
        backend=_one("SamplingResults_*.h5"),
        runlog=_one("RunLog_*.txt"),
    )


def parse_runlog(path: str) -> dict:
    text = Path(path).read_text()

    def grab(pattern, cast=str):
        m = re.search(pattern, text)
        return cast(m.group(1).strip()) if m else None

    return dict(
        injection=grab(r"Injection model: (.*)"),
        recovery=grab(r"Recovery  model: (.*)"),
        tdi=grab(r"TDI channels: (.*)"),
        orbit_file=grab(r"Orbit file: (.*)"),
        noise_file=grab(r"Noise file: (.*)"),
        mojito=grab(r"Mojito L1: (.*)"),
        fixed=grab(r"Fixed params: (.*)"),
        snr=grab(r"SNR \(injection\): (.*)", float),
        mismatch=grab(r"Mismatch \(inj vs recov at truth\): (.*)", float),
        ll_truth=grab(r"loglike at truth: (.*)", float),
    )


def check_settings(cfg: dict, runlog: dict, timing: dict) -> None:
    """Assert that the settings handed to the response now are those the run logged."""
    inj, rec = _waveform_cfg(cfg["Injection"]["Waveform"]), _waveform_cfg(cfg["Recovery"]["Waveform"])
    resp = cfg["Response"]
    expected = dict(
        injection=f"{inj.model} (evolve_chi1={inj.evolve_chi1}, "
                  f"include_1PA_amps={inj.include_1PA_amps})",
        recovery=f"{rec.model} (evolve_chi1={rec.evolve_chi1}, "
                 f"include_1PA_amps={rec.include_1PA_amps})",
        tdi=f"{resp['tdi_chan']}, gen: {resp['tdi_gen']}",
        orbit_file=str(cfg["Data"]["orbit_file"]).strip(),
        noise_file=str(cfg["Data"]["noise_file"]).strip(),
        mojito=f"{str(cfg['Data']['mojito_l1_file']).strip()} "
               f"(t0={timing['t0_l1']}, dt={timing['mojito_dt']})",
        fixed=str(list(cfg["Sampler"]["fixed_params"])),
    )
    mismatched = {k: (runlog[k], v) for k, v in expected.items() if runlog[k] != v}
    if mismatched:
        msg = "\n".join(f"  {k}:\n    RunLog: {a}\n    now:    {b}" for k, (a, b) in mismatched.items())
        raise RuntimeError(f"settings differ from the RunLog:\n{msg}")
    print("Settings match the RunLog:")
    for k, v in expected.items():
        print(f"  {k:<11} {v}")


# ---- likelihood, as in PE_response.py -----------------------------------------------------

def build_likelihood(cfg: dict, use_gpu: bool = True) -> dict:
    with MojitoL1File(cfg["Data"]["mojito_l1_file"]) as l1:
        ts = l1.tdis.time_sampling
        t0_l1 = float(ts.t0)
        mojito_dt = float(ts.dt)
        central_freq = float(l1.laser_frequency)

    inj_wcfg = _waveform_cfg(cfg["Injection"]["Waveform"])
    rec_wcfg = _waveform_cfg(cfg["Recovery"]["Waveform"])
    param_names = rec_wcfg.param_names()
    x_I0_index = param_names.index("x_I0") if "x_I0" in param_names else None
    resp_cfg = _response_cfg(cfg["Response"], cfg["Data"]["orbit_file"])

    oem_orbits = OEMOrbits.from_included("esa-trailing")
    t0_orbits = float(oem_orbits.t_start) + 10.0
    DT = inj_wcfg.dt
    T_response = (inj_wcfg.T
                  + (2 * resp_cfg.offset
                     + 2 * resp_cfg.n_samples_delay * DT) / ASTRONOMICAL_YEAR)
    t0_l0 = t0_l1 - resp_cfg.n_samples_delay * mojito_dt
    t_init = t0_l0 - resp_cfg.offset

    inj_params = _emri_vector(cfg["Injection"]["EMRI"], inj_wcfg.model)
    inj_response = build_response(inj_wcfg, resp_cfg, t_init, t0_orbits,
                                  T_response, use_gpu=use_gpu)
    xyz_data = inj_response(*inj_params)
    N_t = xyz_data.shape[1]

    nonzero_cols = cp.where(cp.any(xyz_data != 0.0, axis=0))[0]
    Nt_injection = int(nonzero_cols[-1]) + 1 if nonzero_cols.size > 0 else N_t
    if bool(cfg["Sampler"]["windowing"]):
        window = cp.zeros(N_t)
        window[:Nt_injection] = cp.asarray(tukey(Nt_injection, alpha=0.01))
    else:
        window = cp.ones(N_t)

    freqs_inband, mask = inband_freqs(N_t, DT, filter_freq=bool(cfg["Sampler"]["filter_freq"]))
    xyz_data_fft = cp.fft.rfft(xyz_data * window, axis=1)[:, mask]
    del inj_response, xyz_data
    cp.get_default_memory_pool().free_all_blocks()

    inv_cov, _ = build_inv_covariance(
        cfg["Data"]["noise_file"], central_freq,
        cp.asnumpy(freqs_inband), DT, N_t,
        channels=resp_cfg.tdi_chan,
    )

    rec_response = build_response(rec_wcfg, resp_cfg, t_init, t0_orbits,
                                  T_response, use_gpu=use_gpu)
    rec_truth_params = _emri_vector(cfg["Injection"]["EMRI"], rec_wcfg.model)

    fixed_names = list(cfg["Sampler"]["fixed_params"])
    _, _, sampled_idx = build_priors(
        param_names, rec_truth_params, fixed_names,
        n=float(cfg["Sampler"]["d"]), use_cupy=use_gpu,
    )
    fixed_idx = {param_names.index(n): rec_truth_params[param_names.index(n)]
                 for n in fixed_names if n in param_names and n != "x_I0"}

    llike = LogLikelihood(
        data_fft=xyz_data_fft,
        inv_cov=inv_cov,
        recovery_response=rec_response,
        cfg=RecoveryConfig(param_names=param_names, fixed_params=fixed_idx,
                           x_I0_index=x_I0_index),
        window=window,
        mask=mask,
    )

    truth = np.array([rec_truth_params[i] for i in sampled_idx])
    xyz_rec_true_fft = cp.fft.rfft(rec_response(*rec_truth_params) * window, axis=1)[:, mask]
    return dict(
        llike=llike,
        sampled_names=[param_names[i] for i in sampled_idx],
        truth=truth,
        snr=float(cp.sqrt(inner_prod_tdi(xyz_data_fft, xyz_data_fft, inv_cov))),
        mismatch=mismatch_tdi(xyz_data_fft, xyz_rec_true_fft, inv_cov),
        ll_truth=float(llike(truth)),
        timing=dict(t0_l1=t0_l1, mojito_dt=mojito_dt, t_init=t_init,
                    T_response=T_response, N_t=N_t, Nt_injection=Nt_injection),
    )


# ---- effective samples --------------------------------------------------------------------

def effective_samples(backend: str, discard: int, ll_floor: float | None = None,
                      max_samples: int | None = None, seed: int = 0) -> dict:
    """
    Cold-chain samples after burn-in, thinned along the iteration axis by ceil(max tau).

    ``ll_floor`` drops walkers whose cold-chain log-likelihood ever falls below it (lost
    walkers); ``max_samples`` draws a random subset of the effective samples.
    """
    reader = HDFBackend(backend, read_only=True)
    chain = reader.get_chain(discard=discard)["model_0"][:, 0, :, 0, :]  # (iters, walkers, ndim)
    log_like = reader.get_log_like(discard=discard)[:, 0, :]             # (iters, walkers)

    tau = np.atleast_2d(reader.get_autocorr_time(discard=discard)["model_0"])[0]
    if not np.all(np.isfinite(tau)):
        raise ValueError(f"auto-correlation time not finite: {tau}")
    thin = int(np.ceil(np.max(tau)))

    walkers = np.arange(chain.shape[1])
    if ll_floor is not None:
        walkers = walkers[np.min(log_like, axis=0) > ll_floor]
        print(f"Kept {len(walkers)}/{chain.shape[1]} walkers above ll = {ll_floor}")

    iters = np.arange(0, chain.shape[0], thin)
    it, wk = np.meshgrid(iters, walkers, indexing="ij")
    it, wk = it.ravel(), wk.ravel()
    n_eff = len(it)
    if max_samples is not None and max_samples < n_eff:
        pick = np.sort(np.random.default_rng(seed).choice(n_eff, max_samples, replace=False))
        it, wk = it[pick], wk[pick]

    print(f"{chain.shape[0]} iterations after discard={discard}, tau = {np.round(tau, 1)}, "
          f"thin = {thin}: {n_eff} effective samples, evaluating {len(it)}")
    return dict(samples=chain[it, wk], ll_old=log_like[it, wk],
                iteration=it + discard, walker=wk, tau=tau, thin=thin, n_eff=n_eff)


# ---- driver -------------------------------------------------------------------------------

def reevaluate(label: str, sampling_dir: Path = SAMPLING_DIR, ll_floor: float | None = None,
               max_samples: int | None = None, seed: int = 0, use_gpu: bool = True,
               save: bool = True) -> dict:
    run = load_run(label, sampling_dir)
    cfg = run["config"]
    print(f"\n===== {label}: {run['run_dir'].name}")
    print(f"Config: {run['config_path']}")

    lk = build_likelihood(cfg, use_gpu=use_gpu)
    old = parse_runlog(run["runlog"])
    check_settings(cfg, old, lk["timing"])

    print(f"{'':<20}{'RunLog':>14}{'now':>14}")
    for key in ("snr", "mismatch", "ll_truth"):
        print(f"{key:<20}{old[key]:>14.6e}{lk[key]:>14.6e}")

    eff = effective_samples(run["backend"], run["discard"], ll_floor, max_samples, seed)
    if eff["samples"].shape[1] != len(lk["sampled_names"]):
        raise ValueError(f"backend has {eff['samples'].shape[1]} parameters, config samples "
                         f"{lk['sampled_names']}")

    t_start = time.time()
    ll_new = np.array([lk["llike"](s) for s in eff["samples"]])
    print(f"Evaluated {len(ll_new)} samples in {time.time() - t_start:.0f} s, "
          f"{np.sum(ll_new <= -1e9)} failed")

    d = ll_new - eff["ll_old"]
    print(f"ll_old: median {np.median(eff['ll_old']):.3f}, max {np.max(eff['ll_old']):.3f}")
    print(f"ll_new: median {np.median(ll_new):.3f}, max {np.max(ll_new):.3f}")
    print(f"ll_new - ll_old: median {np.median(d):.3f}, "
          f"16-84% [{np.percentile(d, 16):.3f}, {np.percentile(d, 84):.3f}]")

    res = dict(
        label=label, run_dir=str(run["run_dir"]), config_path=run["config_path"],
        param_names=lk["sampled_names"], truth=lk["truth"], discard=run["discard"],
        **eff, ll_new=ll_new,
        snr_old=old["snr"], snr_new=lk["snr"],
        mismatch_old=old["mismatch"], mismatch_new=lk["mismatch"],
        ll_truth_old=old["ll_truth"], ll_truth_new=lk["ll_truth"],
    )
    if save:
        OUT_DIR.mkdir(exist_ok=True)
        out = OUT_DIR / f"reeval_eps4_{label}.npz"
        np.savez(out, **res)
        print(f"Saved {out}")

    del lk
    cp.get_default_memory_pool().free_all_blocks()
    return res


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", nargs="+", default=["full", "evol_off"], choices=list(RUNS))
    parser.add_argument("--sampling-dir", type=Path, default=SAMPLING_DIR)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="random subset of the effective samples to evaluate")
    parser.add_argument("--ll-floor", type=float, default=None,
                        help="drop walkers whose cold-chain log-likelihood falls below this")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for label in args.runs:
        reevaluate(label, args.sampling_dir, args.ll_floor, args.max_samples, args.seed)


if __name__ == "__main__":
    main()
