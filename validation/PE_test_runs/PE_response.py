'''
PE validation with fastlisaresponse.

The injected dataset is generated on the fly from the injection waveform
model. recovery may use a different waveform model so we can probe
systematics (1PA amplitude corrections, primary spin evolution, secondary
spin). All paths come from the YAML config. That is the only file one in 
principle needs to edit before running this script on a new test case. 
'''

import argparse
import logging
import os
import time

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from eryn.backends import HDFBackend
from eryn.ensemble import EnsembleSampler
from eryn.moves import StretchMove
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

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# reuseable functions for setting up waveform and response. 
# We do this separately for injection and recovery since we want to be able to use different waveform settings for each
# These return a dataclass config object that can be use to build a response wrapper callable with tuned settings 
def _waveform_cfg(block: dict) -> WaveformConfig:
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
    names = param_names_for(model)
    return [float(emri_block[n]) for n in names]


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True,
                    help="YAML configuration file.")
args = parser.parse_args()

if not os.path.exists(args.config):
    raise FileNotFoundError(args.config)

cfg = param_load(args.config)
logger.info(f"Loaded config from {args.config}")

# Read timing from Mojito L1 file 
mojito_l1 = cfg["Data"]["mojito_l1_file"]
with MojitoL1File(mojito_l1) as l1:
    ts = l1.tdis.time_sampling
    t0_l1 = float(ts.t0)
    mojito_dt = float(ts.dt)
    central_freq = float(l1.laser_frequency)

logger.info(f"Mojito L1: t0={t0_l1}, dt={mojito_dt}")

# define Waveform and response configs. These are dataclasses containing all settings to set up callables
inj_wcfg = _waveform_cfg(cfg["Injection"]["Waveform"])
rec_wcfg = _waveform_cfg(cfg["Recovery"]["Waveform"])
if inj_wcfg.model != rec_wcfg.model:
    raise ValueError("Injection and recovery models must share a parameter "
                        "space (both '1PAT1R' or both '0PA_Kerr').")

param_names = inj_wcfg.param_names()
x_I0_index = param_names.index("x_I0") if "x_I0" in param_names else None

resp_cfg = _response_cfg(cfg["Response"], cfg["Data"]["orbit_file"])
use_gpu = bool(cfg["Sampler"]["use_gpu"])

# Derived timing
oem_orbits = OEMOrbits.from_included("esa-trailing")
t0_orbits = float(oem_orbits.t_start) + 10.0
DT = inj_wcfg.dt
T_response = (inj_wcfg.T
                + (2 * resp_cfg.offset
                    + 2 * resp_cfg.n_samples_delay * DT) / ASTRONOMICAL_YEAR)
t0_l0 = t0_l1 - resp_cfg.n_samples_delay * mojito_dt
t_init = t0_l0 - resp_cfg.offset

# Injection dataset (generate on the fly)
inj_params = _emri_vector(cfg["Injection"]["EMRI"], inj_wcfg.model)

logger.info(f"Building injection response ({inj_wcfg.model})")

inj_response = build_response(inj_wcfg, resp_cfg, t_init, t0_orbits,
                                T_response, use_gpu=use_gpu)
xyz_data = inj_response(*inj_params)

N_t = xyz_data.shape[1]
windowing = bool(cfg["Sampler"]["windowing"])
filter_freq = bool(cfg["Sampler"]["filter_freq"])

# window signal with a Tukey window to avoid spectral leakage
window = cp.asarray(tukey(N_t, alpha=0.01)) if windowing else cp.ones(N_t)

# reduce data to in-band frequencies for faster likelihood evaluation
freqs_inband, mask = inband_freqs(N_t, DT, filter_freq=filter_freq)
xyz_data_fft = cp.fft.rfft(xyz_data * window, axis=1)[:, mask]

# Keep a CPU copy of the time-domain injection for plots, then free GPU.
xyz_data_np = cp.asnumpy(xyz_data)
del inj_response, xyz_data
cp.get_default_memory_pool().free_all_blocks()

# Build covariance matrix and invert
logger.info("Building inverse covariance from Mojito noise")
inv_cov, psd_diag = build_inv_covariance(
    cfg["Data"]["noise_file"], central_freq,
    cp.asnumpy(freqs_inband), DT, N_t,
    channels=resp_cfg.tdi_chan,
)

# set up response for Recovery 
logger.info(f"Building recovery response ({rec_wcfg.model})")
rec_response = build_response(rec_wcfg, resp_cfg, t_init, t0_orbits,
                                T_response, use_gpu=use_gpu)

############################################################
# create prior and likelihood objects and do consistency checks before sampling
fixed_names = list(cfg["Sampler"]["fixed_params"])
priors, bounds, sampled_idx = build_priors(
    param_names, inj_params, fixed_names,
    n=float(cfg["Sampler"]["d"]),
    use_cupy=True,
)
fixed_idx = {param_names.index(n): inj_params[param_names.index(n)]
                for n in fixed_names if n in param_names and n != "x_I0"}

llike = LogLikelihood(
    data_fft=xyz_data_fft,
    inv_cov=inv_cov,
    recovery_response=rec_response,
    cfg=RecoveryConfig(
        param_names=param_names,
        fixed_params=fixed_idx,
        x_I0_index=x_I0_index,
    ),
    window=window,
    mask=mask,
)

# Do some consistency chekcs
logger.info("Running consistency checks")
xyz_rec_true_td = rec_response(*inj_params)
xyz_rec_true_fft = cp.fft.rfft(xyz_rec_true_td * window, axis=1)[:, mask]

# get SNR
snr = float(cp.sqrt(inner_prod_tdi(xyz_data_fft, xyz_data_fft, inv_cov)))

# get mismatch
mm = mismatch_tdi(xyz_data_fft, xyz_rec_true_fft, inv_cov)

# get loglikelihood at true value
ll_truth = float(llike([inj_params[i] for i in sampled_idx]))

logger.info(f"SNR = {snr:.2f}")
logger.info(f"Mismatch (injection vs recovery at truth) = {mm:.3e}")
logger.info(f"loglike at truth = {ll_truth:.3e}")
if snr < 20:
    logger.warning("Injection SNR below 20 -- may not be recoverable.")

# Draw starting positions for all walkers from prior. Tighten prior around truth value to ensure fast convergence during test run
ndim = len(sampled_idx)
nwalkers = int(cfg["Sampler"]["n_walkers"])
n_temps = int(cfg["Sampler"]["n_temps"])
start_val = np.array([inj_params[i] for i in sampled_idx])

def _draw(shape):
    pts = start_val + 1e-7 * float(cfg["Sampler"]["d"]) * np.random.randn(*shape, ndim)
    for k in range(ndim):
        lo, hi = bounds[k]
        pts[..., k] = np.clip(pts[..., k], lo, hi)
    return pts

start = _draw((n_temps, nwalkers)) if n_temps > 1 else _draw((nwalkers,))
flat = start.reshape(-1, ndim)
if not np.all(np.isfinite(priors.logpdf(flat))):
    raise ValueError("Starting positions outside prior support.")

# Create a shortt summary of set-up and checks: SNR, mismatch at true vals, starting loglikelihood values. 
home = os.getcwd()
data_dir = cfg["Sampler"]["sampling_data_path"]
# Add timestamp to avoid overwriting previous runs by accident.
timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
fixed_tag = "_".join(f"{n}_{cfg['Injection']['EMRI'][n]}" for n in fixed_names)
tag = f"{cfg['Sampler']['name']}_fixed{fixed_tag}_{timestamp}"
out_dir = os.path.join(home, "..", "..", data_dir)
os.makedirs(out_dir, exist_ok=True)
fp = os.path.join(out_dir, f"SamplingResults_{tag}.h5")
log_fp = os.path.join(out_dir, f"RunLog_{tag}.txt")

with open(log_fp, "w") as f:
    f.write(f"# PE run log -- {timestamp}\n")
    f.write(f"Config: {args.config}\n")
    f.write(f"Injection model: {inj_wcfg.model} "
            f"(evolve_chi1={inj_wcfg.evolve_chi1}, "
            f"include_1PA_amps={inj_wcfg.include_1PA_amps})\n")
    f.write(f"Recovery  model: {rec_wcfg.model} "
            f"(evolve_chi1={rec_wcfg.evolve_chi1}, "
            f"include_1PA_amps={rec_wcfg.include_1PA_amps})\n")
    f.write(f"TDI channels: {resp_cfg.tdi_chan}, gen: {resp_cfg.tdi_gen}\n")
    f.write(f"Orbit file: {resp_cfg.orbit_file}\n")
    f.write(f"Noise file: {cfg['Data']['noise_file']}\n")
    f.write(f"Mojito L1: {mojito_l1} (t0={t0_l1}, dt={mojito_dt})\n")
    f.write(f"Fixed params: {fixed_names}\n")
    f.write(f"Sampler: ntemps={n_temps}, nwalkers={nwalkers}, "
            f"iterations={cfg['Sampler']['num_samples']}, "
            f"d={cfg['Sampler']['d']}\n")
    f.write(f"SNR (injection): {snr:.4f}\n")
    f.write(f"Mismatch (inj vs recov at truth): {mm:.6e}\n")
    f.write(f"loglike at truth: {ll_truth:.6e}\n")
    f.write(f"Backend: {fp}\n")

###################
# Make some diagnostic plots
plots_dir = os.path.join(home, cfg["Sampler"].get("plots_path", "../../Plots"))
os.makedirs(plots_dir, exist_ok=True)
freqs_np = cp.asnumpy(freqs_inband)
data_fft_np = cp.asnumpy(xyz_data_fft)
rec_fft_np = cp.asnumpy(xyz_rec_true_fft)
xyz_rec_true_np = cp.asnumpy(xyz_rec_true_td)

# X-channel FD: injection vs recovery-at-truth vs noise PSD
fig, ax = plt.subplots(figsize=(9, 5))
ax.loglog(freqs_np, 2 * freqs_np * np.abs(data_fft_np[0]),
            label="Injection", alpha=0.8)
ax.loglog(freqs_np, 2 * freqs_np * np.abs(rec_fft_np[0]),
            label="Recovery at truth", alpha=0.8, ls="--")
ax.loglog(freqs_np, np.sqrt(freqs_np * psd_diag[0]),
            label="Noise ASD (X)", ls=":", color="k")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Characteristic strain")
ax.set_title(f"{cfg['Sampler']['name']} - X channel")
ax.grid(which="both", alpha=0.4)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(plots_dir, f"{tag}_X_fd.png"), dpi=150)
plt.close(fig)

# Time-domain X channel + residual
t_axis = t_init + np.arange(N_t) * DT
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].plot(t_axis, xyz_data_np[0], lw=0.6, label="Injection")
axes[0].plot(t_axis, xyz_rec_true_np[0], lw=0.6, ls="--",
                label="Recovery at truth")
axes[0].set_ylabel("TDI X")
axes[0].legend()
axes[1].plot(t_axis, xyz_data_np[0] - xyz_rec_true_np[0], lw=0.6, color="C3")
axes[1].set_ylabel("Residual")
axes[1].set_xlabel("Time [s]")
for ax in axes:
    ax.grid(alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(plots_dir, f"{tag}_X_td.png"), dpi=150)
plt.close(fig)
logger.info(f"Plots written to {plots_dir}")

###################
# Sampler and backend
backend = HDFBackend(fp)
if bool(cfg["Sampler"]["continue_run"]) and os.path.exists(fp):
    logger.info(f"Continuing from existing backend {fp}")
    start = backend.get_last_sample()

ensemble = EnsembleSampler(
    nwalkers,
    ndim,
    llike,
    priors,
    backend=backend,
    tempering_kwargs=dict(ntemps=n_temps),
    moves=StretchMove(a=2.0, use_gpu=True),
)

logger.info(f"Starting MCMC: {cfg['Sampler']['num_samples']} iterations "
            f"-> {fp}")
ensemble.run_mcmc(start, int(cfg["Sampler"]["num_samples"]), progress=True)
logger.info("Done.")

