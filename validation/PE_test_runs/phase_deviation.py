'''
Phase deviation between an injected EMRI and its recovered max-likelihood point.

Rebuilds the *trajectory* modules only (no response, no GPU work) for the
injection and the recovery waveform models exactly as `src/waveform.py`
configures them from the YAML:

    1PAT1R                  -> few.trajectory.ode.TrajectoryCirc1PAT1R
                               (evolve_primary from `evolve_chi1`)
    0PA_Kerr                -> few.trajectory.ode.KerrEccEqFlux
    FastSchwarzschild*      -> few.trajectory.ode.SchwarzEccFlux

`TrajectoryCirc1PAT1R` integrates the 8-component state

    y = [p, e, xI, Phi_phi, Phi_theta, Phi_r, deltaM, delta_chit1]

so its trajectory carries the drift of the primary's mass and spin alongside
the orbital phases; the adiabatic models return the first six only.

The injection trajectory is integrated at the injected (redshifted) parameters,
the recovery trajectory at the maximum-likelihood sample of the chain (fixed
parameters filled in at their injected values, as in `recovered_snr.py`).  Both
are interpolated onto a common coordinate-time grid and compared through

    dPhi_phi(t) = Phi_phi^inj(t) - Phi_phi^ML(t)

Outputs a short dephasing report plus three figures: the dephasing over the
full span, the azimuthal phase at the start and at the end of the observation,
and the (a, p) phase-space evolution of both systems.

Usage
-----
    from phase_deviation import phase_deviation

    res = phase_deviation(run_dir, discard=5000)
    print(res['report'])

or from the command line:

    python phase_deviation.py <run_dir> --discard 5000 --outdir ./Plots
'''

from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

from recovered_snr import max_likelihood_sample, sampled_indices
from src.io import param_load
from src.waveform import param_names_for


# Trajectory setup mirroring src/waveform.py

TRAJ_ODE_FOR_MODEL: Dict[str, str] = {
    "1PAT1R": "TrajectoryCirc1PAT1R",
    "0PA_Kerr": "KerrEccEqFlux",
    "FastSchwarzschildEccentricFlux": "SchwarzEccFlux",
    "FastSchwarzschildEccentricFluxBicubic": "SchwarzEccFlux",
}

# Initial phases each waveform actually forwards to its trajectory: the rest are
# dropped by GenerateEMRIWaveform and start from zero.  Mirrors the
# `phases_needed` branches of GenerateEMRIWaveform.__init__ -- Circ1PAT1R keeps
# Phi_phi0 only, the Schwarzschild models keep Phi_phi0/Phi_r0 (descriptor
# "eccentric"), and KerrEccEq keeps all three (descriptor "eccentric equatorial",
# which does *not* match that branch).
PHASES_FOR_MODEL: Dict[str, Tuple[str, ...]] = {
    "1PAT1R": ("Phi_phi0",),
    "0PA_Kerr": ("Phi_phi0", "Phi_theta0", "Phi_r0"),
    "FastSchwarzschildEccentricFlux": ("Phi_phi0", "Phi_r0"),
    "FastSchwarzschildEccentricFluxBicubic": ("Phi_phi0", "Phi_r0"),
}

# Keywords EMRIInspiral consumes per call; everything else in `inspiral_kwargs`
# belongs on the integrator/ODE constructor (see EMRIInspiral.specific_kwarg_keys).
_TRAJ_CALL_KEYS = frozenset({
    "T", "dt", "err", "DENSE_STEPPING", "buffer_length",
    "integrate_backwards", "max_step_size",
    # consumed by TrajectoryBase.__call__ itself rather than forwarded
    "in_coordinate_time", "new_t", "spline_kwargs", "upsample", "fix_t",
})
_TRAJ_KEY_ALIASES = {"max_init_len": "buffer_length"}

# SphericalHarmonicWaveformBase.__call__ pins this before calling the trajectory
DEFAULT_ERR = 1e-11

INJ_COLOR = "C0"
ML_COLOR = "C3"


def split_inspiral_kwargs(block: dict) -> Tuple[Dict, Dict]:
    """
    Split a Waveform block's `inspiral_kwargs` into (constructor, call) kwargs.

    The 1PA `evolve_chi1` toggle is wired into the ODE constructor as
    `evolve_primary`, exactly as `EMRIWave.__init__` does for the waveform.
    """
    ctor: Dict = {}
    call: Dict = {}
    for key, val in dict(block.get("inspiral_kwargs") or {}).items():
        key = _TRAJ_KEY_ALIASES.get(key, key)
        (call if key in _TRAJ_CALL_KEYS else ctor)[key] = val
    if block["model"] == "1PAT1R":
        ctor["evolve_primary"] = bool(block.get("evolve_chi1", True))
    return ctor, call


def build_trajectory(block: dict):
    """
    Return (traj, model, call_kwargs) for a Waveform block of the config.

    `traj` is an `EMRIInspiral` around the ODE that the corresponding waveform
    model integrates internally.
    """
    from few.trajectory import ode as few_ode
    from few.trajectory.inspiral import EMRIInspiral

    model = block["model"]
    try:
        ode_name = TRAJ_ODE_FOR_MODEL[model]
    except KeyError:
        raise ValueError(f"No trajectory module known for waveform model {model!r}.")
    ode = getattr(few_ode, ode_name)

    ctor_kwargs, call_kwargs = split_inspiral_kwargs(block)
    call_kwargs.setdefault("dt", float(block["dt"]))
    call_kwargs.setdefault("T", float(block["T"]))
    call_kwargs.setdefault("err", DEFAULT_ERR)
    return EMRIInspiral(func=ode, **ctor_kwargs), model, call_kwargs


def run_trajectory(traj, model: str, params: Dict[str, float], call_kwargs: Dict) -> Dict:
    """
    Integrate one inspiral and return its evolution in coordinate time.

    `params` is a name -> value mapping in the layout of `model` (masses already
    redshifted).  The returned `a` is the *evolving* primary spin for 1PAT1R
    (a + delta_chit1) and a constant array otherwise; `dM` is the fractional
    total-mass drift deltaM (1PA only, None elsewhere).

    Only the initial phases the waveform actually forwards are passed on, and a
    retrograde x_I0 is folded into (a, Phi_phi0) the way GenerateEMRIWaveform
    does, so the phases match the ones the PE template was built from.
    """
    a, x0 = params["a"], params["x_I0"]
    phases = {n: params[n] for n in PHASES_FOR_MODEL[model]}
    if x0 < 0.0:
        a, x0 = -a, -x0
        phases["Phi_phi0"] = phases["Phi_phi0"] + np.pi
    if traj.func.background == "Schwarzschild":
        a = 0.0     # get_inspiral zeroes the spin for a Schwarzschild background

    args = [params["M"], params["mu"], a, params["p0"], params["e0"], x0]
    if model == "1PAT1R":
        args.append(params["chi2"])

    out = traj(*args, **phases, **call_kwargs)
    t, p, e, x, Phi_phi, Phi_theta, Phi_r = out[:7]
    dM = np.asarray(out[7]) if len(out) > 7 else None
    a_t = a + np.asarray(out[8]) if len(out) > 8 else np.full_like(t, a)

    return dict(t=np.asarray(t), p=np.asarray(p), e=np.asarray(e), x=np.asarray(x),
                Phi_phi=np.asarray(Phi_phi), Phi_theta=np.asarray(Phi_theta),
                Phi_r=np.asarray(Phi_r), a=np.asarray(a_t), dM=dM, model=model)


# Comparison helpers

def _emri_named(emri_block: dict, model: str) -> Dict[str, float]:
    """Injected parameters in the layout of `model`, redshifting M and mu."""
    z = float(emri_block.get("z", 0.0))
    out = {}
    for n in param_names_for(model):
        val = float(emri_block[n])
        if n in ("M", "mu"):
            val *= (1.0 + z)
        out[n] = val
    return out


def _resample(traj_out: Dict, t_grid: np.ndarray, keys: Sequence[str]) -> Dict[str, np.ndarray]:
    """Cubic-spline resampling of trajectory quantities onto `t_grid`."""
    t = traj_out["t"]
    return {k: CubicSpline(t, traj_out[k])(t_grid) for k in keys}


def _first_crossing(t: np.ndarray, dphi: np.ndarray, level: float) -> Optional[float]:
    """First time |dphi| exceeds `level` radians, or None if it never does."""
    idx = np.nonzero(np.abs(dphi) >= level)[0]
    return float(t[idx[0]]) if idx.size else None


def phase_deviation(
    run_dir: Optional[str] = None,
    config_path: Optional[str] = None,
    h5_path: Optional[str] = None,
    discard: int = 0,
    temp: int = 0,
    params_ml: Optional[Sequence[float]] = None,
    n_out: int = 20000,
    zoom: float = 4 * 3600.0,
) -> Dict[str, object]:
    """
    Compare the injected inspiral with the max-likelihood recovered inspiral.

    Parameters
    ----------
    run_dir    : sampling directory holding the copied `*.yaml` and
                 `SamplingResults_*.h5`; `config_path` / `h5_path` override it
    discard    : burn-in iterations dropped before locating the ML point
    temp       : temperature index of the chain to search (0 = cold)
    params_ml  : optional explicit sampled-parameter vector (skips reading the chain)
    n_out      : points on the common comparison grid
    zoom       : length [s] of the start/end windows used for the phase plots

    Returns
    -------
    dict with keys
        t              : common coordinate-time grid [s]
        dPhi_phi       : Phi_phi^inj - Phi_phi^ML on that grid [rad]
        dPhi_r         : same for the radial phase [rad]
        inj, ml        : full trajectory outputs (see `run_trajectory`)
        inj_params     : injected parameters, injection-model layout
        ml_params      : ML parameters, recovery-model layout (fixed at truth)
        summary        : scalar dephasing diagnostics
        report         : formatted text report
    """
    # Locate config and backend (same discovery rules as recovered_snr.py)
    if config_path is None:
        if run_dir is None:
            raise ValueError("Provide either run_dir or config_path.")
        matches = glob.glob(os.path.join(run_dir, "*.yaml"))
        if not matches:
            raise FileNotFoundError(f"No YAML config found in {run_dir}")
        config_path = matches[0]
    cfg = param_load(config_path)

    inj_block = cfg["Injection"]["Waveform"]
    rec_block = cfg["Recovery"]["Waveform"]
    inj_model = inj_block["model"]
    rec_model = rec_block["model"]

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

    # Full recovery vector: injected truth with the sampled entries overwritten
    rec_names = param_names_for(rec_model)
    idx_sampled = sampled_indices(rec_names, list(cfg["Sampler"].get("fixed_params", []) or []))
    sampled_names = [rec_names[i] for i in idx_sampled]
    if len(params_sampled) != len(idx_sampled):
        raise ValueError(
            f"Chain has {len(params_sampled)} sampled parameters but the config "
            f"implies {len(idx_sampled)} ({sampled_names})."
        )

    inj_params = _emri_named(cfg["Injection"]["EMRI"], inj_model)
    ml_params = _emri_named(cfg["Injection"]["EMRI"], rec_model)
    for k, i in enumerate(idx_sampled):
        ml_params[rec_names[i]] = float(params_sampled[k])

    # Integrate both inspirals
    inj_traj, _, inj_call = build_trajectory(inj_block)
    rec_traj, _, rec_call = build_trajectory(rec_block)
    inj = run_trajectory(inj_traj, inj_model, inj_params, inj_call)
    ml = run_trajectory(rec_traj, rec_model, ml_params, rec_call)

    # Common grid over the overlap of the two inspirals
    t_end = min(float(inj["t"][-1]), float(ml["t"][-1]))
    t_grid = np.linspace(0.0, t_end, int(n_out))
    keys = ("p", "e", "Phi_phi", "Phi_r", "a")
    inj_i = _resample(inj, t_grid, keys)
    ml_i = _resample(ml, t_grid, keys)

    dPhi_phi = inj_i["Phi_phi"] - ml_i["Phi_phi"]
    dPhi_r = inj_i["Phi_r"] - ml_i["Phi_r"]

    summary = dict(
        t_plunge_inj=float(inj["t"][-1]),
        t_plunge_ml=float(ml["t"][-1]),
        t_common=t_end,
        dPhi_phi_end=float(dPhi_phi[-1]),
        dPhi_phi_max=float(np.max(np.abs(dPhi_phi))),
        dPhi_phi_rms=float(np.sqrt(np.mean(dPhi_phi ** 2))),
        dPhi_r_end=float(dPhi_r[-1]),
        dPhi_r_max=float(np.max(np.abs(dPhi_r))),
        n_cycles_inj=float(inj_i["Phi_phi"][-1] - inj_i["Phi_phi"][0]) / (2 * np.pi),
        n_cycles_ml=float(ml_i["Phi_phi"][-1] - ml_i["Phi_phi"][0]) / (2 * np.pi),
        t_dephase_0p1=_first_crossing(t_grid, dPhi_phi, 0.1),
        t_dephase_1=_first_crossing(t_grid, dPhi_phi, 1.0),
        t_dephase_10=_first_crossing(t_grid, dPhi_phi, 10.0),
        p_end_inj=float(inj_i["p"][-1]),
        p_end_ml=float(ml_i["p"][-1]),
        a_end_inj=float(inj_i["a"][-1]),
        a_end_ml=float(ml_i["a"][-1]),
        log_like_max=ll_max,
    )

    res = dict(
        t=t_grid, dPhi_phi=dPhi_phi, dPhi_r=dPhi_r,
        inj=inj, ml=ml, inj_interp=inj_i, ml_interp=ml_i,
        inj_params=inj_params, ml_params=ml_params,
        inj_model=inj_model, rec_model=rec_model,
        sampled_names=sampled_names, summary=summary,
        name=str(cfg["Sampler"].get("name", "run")), zoom=float(zoom),
        config_path=config_path,
    )
    res["report"] = format_report(res)
    return res


# Reporting

def format_report(res: Dict[str, object]) -> str:
    s = res["summary"]
    yr = 365.25 * 24 * 3600.0
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append(f"  Phase deviation report -- {res['name']}")
    lines.append(f"  injection: {res['inj_model']}   recovery: {res['rec_model']}")
    lines.append("=" * 72)

    lines.append("")
    lines.append("Parameters (injected -> max-likelihood):")
    lines.append(f"  {'param':<12}{'injected':>18}{'max-like':>18}{'delta':>16}")
    for name in res["ml_params"]:
        inj_val = res["inj_params"].get(name, float("nan"))
        ml_val = res["ml_params"][name]
        tag = "" if name in res["sampled_names"] else "  (fixed)"
        lines.append(f"  {name:<12}{inj_val:>18.10g}{ml_val:>18.10g}"
                     f"{ml_val - inj_val:>16.4g}{tag}")
    for name in res["inj_params"]:
        if name not in res["ml_params"]:
            lines.append(f"  {name:<12}{res['inj_params'][name]:>18.10g}"
                         f"{'--':>18}{'--':>16}  (injection only)")
    if s["log_like_max"] is not None:
        lines.append(f"\n  loglike at ML point (sampler) : {s['log_like_max']:.6e}")

    lines.append("")
    lines.append("Inspiral duration:")
    lines.append(f"  plunge time, injection      : {s['t_plunge_inj']:.4g} s "
                 f"({s['t_plunge_inj'] / yr:.4f} yr)")
    lines.append(f"  plunge time, max-likelihood : {s['t_plunge_ml']:.4g} s "
                 f"({s['t_plunge_ml'] / yr:.4f} yr)")
    lines.append(f"  compared over               : {s['t_common']:.4g} s "
                 f"({s['t_common'] / yr:.4f} yr)")
    lines.append(f"  cycles (inj / ML)           : {s['n_cycles_inj']:.2f} / "
                 f"{s['n_cycles_ml']:.2f}")

    lines.append("")
    lines.append("Dephasing (injection - max-likelihood):")
    lines.append(f"  dPhi_phi at end             : {s['dPhi_phi_end']:+.4f} rad "
                 f"({s['dPhi_phi_end'] / (2 * np.pi):+.4f} cycles)")
    lines.append(f"  max |dPhi_phi|              : {s['dPhi_phi_max']:.4f} rad")
    lines.append(f"  rms dPhi_phi                : {s['dPhi_phi_rms']:.4f} rad")
    lines.append(f"  dPhi_r at end               : {s['dPhi_r_end']:+.4f} rad")
    for level, key in ((0.1, "t_dephase_0p1"), (1.0, "t_dephase_1"), (10.0, "t_dephase_10")):
        t_x = s[key]
        val = f"{t_x:.4g} s ({t_x / yr:.4f} yr)" if t_x is not None else "never"
        lines.append(f"  |dPhi_phi| > {level:<5g} rad reached : {val}")

    lines.append("")
    lines.append("Orbital evolution at the end of the compared span:")
    lines.append(f"  p   (inj / ML)              : {s['p_end_inj']:.6f} / {s['p_end_ml']:.6f}"
                 f"   (delta {s['p_end_ml'] - s['p_end_inj']:+.3e})")
    lines.append(f"  a   (inj / ML)              : {s['a_end_inj']:.6e} / {s['a_end_ml']:.6e}"
                 f"   (delta {s['a_end_ml'] - s['a_end_inj']:+.3e})")
    for label, traj in (("injection", res["inj"]), ("max-like", res["ml"])):
        da = float(traj["a"][-1] - traj["a"][0])
        line = f"  spin drift, {label:<16}: {da:+.4e}"
        if traj["dM"] is not None:
            line += f"   mass drift dM = {float(traj['dM'][-1]):+.4e}"
        lines.append(line)
    lines.append("")
    return "\n".join(lines)


# Plots

def plot_dephasing(res: Dict[str, object], outdir: str) -> str:
    """dPhi_phi over the full compared span, linear and logarithmic."""
    t = res["t"] / (365.25 * 24 * 3600.0)
    dphi = res["dPhi_phi"]

    fig, axs = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axs[0].plot(t, dphi, color="k", lw=1.2)
    axs[0].axhline(0.0, color="grey", lw=0.6)
    axs[0].set_ylabel(r"$\Delta\Phi_\phi$ [rad]")
    axs[0].set_title(f"Dephasing, injection $-$ max-likelihood ({res['name']})")

    axs[1].semilogy(t, np.abs(dphi), color="k", lw=1.2)
    for level, style in ((0.1, ":"), (1.0, "--"), (10.0, "-.")):
        axs[1].axhline(level, color="grey", lw=0.8, ls=style, label=f"{level:g} rad")
    axs[1].set_ylabel(r"$|\Delta\Phi_\phi|$ [rad]")
    axs[1].set_xlabel("time [yr]")
    axs[1].legend(fontsize=8)
    for ax in axs:
        ax.grid(alpha=0.3)

    path = os.path.join(outdir, f"phase_deviation_{res['name']}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_phase_start_end(res: Dict[str, object], outdir: str, n_cycles: float = 3.0) -> str:
    """
    Azimuthal phase at the start and at the end of the inspiral.

    Top row shows cos(2 Phi_phi), i.e. the phasing of the dominant l=m=2
    harmonic over the first `n_cycles` orbits of each window -- the visual
    dephasing between the two systems.  Middle and bottom rows show the
    accumulated phase and the dephasing over the whole window.
    """
    zoom = res["zoom"]
    t_end = res["summary"]["t_common"]
    windows = [("start", 0.0, min(zoom, t_end)),
               ("end", max(0.0, t_end - zoom), t_end)]
    spl_inj = CubicSpline(res["inj"]["t"], res["inj"]["Phi_phi"])
    spl_ml = CubicSpline(res["ml"]["t"], res["ml"]["Phi_phi"])
    inj_label = f"injection ({res['inj_model']})"
    ml_label = f"max-likelihood ({res['rec_model']})"

    fig, axs = plt.subplots(3, 2, figsize=(11, 9))
    for col, (label, t0, t1) in enumerate(windows):
        t_w = np.linspace(t0, t1, 4000)
        phi_inj, phi_ml = spl_inj(t_w), spl_ml(t_w)

        # A few orbits from the start of the window, resolved in the waveform phase
        t_osc_end = float(np.interp(phi_inj[0] + n_cycles * 2 * np.pi, phi_inj, t_w))
        t_osc = np.linspace(t0, t_osc_end, 2000)
        ax = axs[0, col]
        ax.plot(t_osc - t0, np.cos(2 * spl_inj(t_osc)), color=INJ_COLOR, lw=1.2,
                label=inj_label)
        ax.plot(t_osc - t0, np.cos(2 * spl_ml(t_osc)), color=ML_COLOR, lw=1.2, ls="--",
                label=ml_label)
        ax.set_ylabel(r"$\cos(2\Phi_\phi)$")
        ax.set_title(f"{label} of observation  ($t_0$ = {t0 / 86400.0:.2f} d)")
        ax.set_xlabel(r"$t - t_0$ [s]")
        ax.legend(fontsize=8, loc="lower left")

        ax = axs[1, col]
        ax.plot(t_w - t0, phi_inj - phi_inj[0], color=INJ_COLOR, lw=1.2, label=inj_label)
        ax.plot(t_w - t0, phi_ml - phi_ml[0], color=ML_COLOR, lw=1.2, ls="--", label=ml_label)
        ax.set_ylabel(r"$\Phi_\phi - \Phi_\phi(t_0)$ [rad]")

        ax = axs[2, col]
        ax.plot(t_w - t0, phi_inj - phi_ml, color="k", lw=1.2)
        ax.axhline(0.0, color="grey", lw=0.6)
        ax.set_ylabel(r"$\Delta\Phi_\phi$ [rad]")
        ax.set_xlabel(r"$t - t_0$ [s]")

        for ax in axs[:, col]:
            ax.grid(alpha=0.3)

    fig.suptitle(f"Azimuthal phase at the start and end of the inspiral ({res['name']})")
    path = os.path.join(outdir, f"phase_start_end_{res['name']}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_phase_space(res: Dict[str, object], outdir: str) -> str:
    """(a, p) phase-space evolution plus the individual p(t) and a(t) tracks."""
    yr = 365.25 * 24 * 3600.0
    fig = plt.figure(figsize=(12, 6.5))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.2, 1.0])
    ax_ps = fig.add_subplot(gs[:, 0])
    ax_p = fig.add_subplot(gs[0, 1])
    ax_a = fig.add_subplot(gs[1, 1], sharex=ax_p)
    ax_dp = fig.add_subplot(gs[2, 1], sharex=ax_p)

    for traj, color, ls, label in (
        (res["inj"], INJ_COLOR, "-", f"injection ({res['inj_model']})"),
        (res["ml"], ML_COLOR, "--", f"max-likelihood ({res['rec_model']})"),
    ):
        ax_ps.plot(traj["a"], traj["p"], color=color, ls=ls, lw=1.3, label=label)
        ax_ps.plot(traj["a"][0], traj["p"][0], "o", color=color, ms=5)
        ax_ps.plot(traj["a"][-1], traj["p"][-1], "s", color=color, ms=5)
        ax_p.plot(traj["t"] / yr, traj["p"], color=color, ls=ls, lw=1.2, label=label)
        ax_a.plot(traj["t"] / yr, traj["a"], color=color, ls=ls, lw=1.2)

    ax_ps.set_xlabel(r"$a$ (primary spin)")
    ax_ps.set_ylabel(r"$p$")
    ax_ps.set_title(r"$(a, p)$ phase space (circles: start, squares: plunge)")
    ax_ps.legend(fontsize=8)

    ax_dp.plot(res["t"] / yr, res["inj_interp"]["p"] - res["ml_interp"]["p"],
               color="k", lw=1.2)
    ax_dp.axhline(0.0, color="grey", lw=0.6)

    ax_p.set_ylabel(r"$p$")
    ax_a.set_ylabel(r"$a$")
    ax_dp.set_ylabel(r"$\Delta p$")
    ax_dp.set_xlabel("time [yr]")
    for ax in (ax_ps, ax_p, ax_a, ax_dp):
        ax.grid(alpha=0.3)

    fig.suptitle(f"Phase-space evolution ({res['name']})")
    path = os.path.join(outdir, f"phase_space_{res['name']}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def make_plots(res: Dict[str, object], outdir: str) -> List[str]:
    os.makedirs(outdir, exist_ok=True)
    return [plot_dephasing(res, outdir),
            plot_phase_start_end(res, outdir),
            plot_phase_space(res, outdir)]


if __name__ == "__main__":
    import argparse

    import matplotlib
    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=str, help="Sampling run directory.")
    parser.add_argument("--discard", type=int, default=0, help="Burn-in iterations.")
    parser.add_argument("--temp", type=int, default=0, help="Temperature index.")
    parser.add_argument("--n-out", type=int, default=20000,
                        help="Points on the common comparison grid.")
    parser.add_argument("--zoom", type=float, default=4 * 3600.0,
                        help="Length [s] of the start/end phase windows.")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Where to write the figures and report (default: run_dir).")
    args = parser.parse_args()

    res = phase_deviation(run_dir=args.run_dir, discard=args.discard, temp=args.temp,
                          n_out=args.n_out, zoom=args.zoom)
    print(res["report"])

    outdir = args.outdir or args.run_dir
    paths = make_plots(res, outdir)
    report_path = os.path.join(outdir, f"phase_deviation_{res['name']}.txt")
    with open(report_path, "w") as f:
        f.write(res["report"])

    print("Written:")
    for p in [report_path] + paths:
        print(f"  {p}")
