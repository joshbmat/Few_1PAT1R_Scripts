'''
PE validation WITHOUT the LISA response.

Evaluates the Whittle likelihood directly as an inner product of the
detector-frame complex strain h+ - i hx against the analytic LISA PSD
(`lisatools.sensitivity.LISASens`). Supports both waveform models via
`src.waveform.EMRIWave`, so the same config schema as `PE_response.py`
works: `Injection.Waveform.model` and `Recovery.Waveform.model` can be
either '1PAT1R' or '0PA_Kerr' (must agree on parameter space).

Use this script for waveform-stability checks and fast exploratory runs
where TDI and realistic Mojito noise are not needed.
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
from lisatools.sensitivity import LISASens, get_sensitivity
from scipy.signal.windows import tukey

from src.io import param_load
from src.priors import build_priors
from src.utils import inband_freqs, inner_product
from src.waveform import EMRIWave, WaveformConfig, param_names_for

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


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


def _emri_vector(emri_block: dict, model: str) -> list[float]:
    z = float(emri_block.get("z", 0.0))
    params = []
    for n in param_names_for(model):
        val = float(emri_block[n])
        if n in ("M", "mu"):
            val *= (1.0 + z)
        params.append(val)
    return params


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True,
                    help="YAML configuration file.")
args = parser.parse_args()

if not os.path.exists(args.config):
    raise FileNotFoundError(args.config)

cfg = param_load(args.config)
logger.info(f"Loaded config from {args.config}")

use_gpu = bool(cfg["Sampler"]["use_gpu"])
if use_gpu and not cp.is_available():
    logger.warning("GPU requested but cupy is not available; falling back to CPU.")
    use_gpu = False
xp = cp if use_gpu else np
force_backend = "cuda12x" if use_gpu else None

# Injection / recovery configs. Allowed to differ for systematics studies, as
# long as they share the parameter space (both 1PAT1R or both 0PA_Kerr).
inj_wcfg = _waveform_cfg(cfg["Injection"]["Waveform"])
rec_wcfg = _waveform_cfg(cfg["Recovery"]["Waveform"])
if inj_wcfg.model != rec_wcfg.model:
    raise ValueError("Injection and recovery models must share a parameter "
                     "space (both '1PAT1R' or both '0PA_Kerr').")

param_names = inj_wcfg.param_names()
x_I0_index = param_names.index("x_I0") if "x_I0" in param_names else None
DT = inj_wcfg.dt

inj_params = _emri_vector(cfg["Injection"]["EMRI"], inj_wcfg.model)

# Waveform generators (no response)
logger.info(f"Building injection waveform ({inj_wcfg.model})")
inj_wave = EMRIWave(inj_wcfg, force_backend=force_backend)

logger.info(f"Building recovery waveform ({rec_wcfg.model})")
rec_wave = EMRIWave(rec_wcfg, force_backend=force_backend)

# Injection strain: complex h+ - i hx in detector frame
h_inj = xp.asarray(inj_wave(*inj_params))
N_t = len(h_inj)

windowing = bool(cfg["Sampler"]["windowing"])
filter_freq = bool(cfg["Sampler"]["filter_freq"])
window = xp.asarray(tukey(N_t, alpha=0.01)) if windowing else xp.ones(N_t)

freqs_inband, mask = inband_freqs(N_t, DT, filter_freq=filter_freq)
df = 1.0 / (N_t * DT)

h_inj_fft = xp.fft.rfft(h_inj * window)[mask]

# Analytic LISA PSD on the analysis grid
Sn = get_sensitivity(freqs_inband, sens_fn=LISASens, return_type="PSD")

# Priors
fixed_names = list(cfg["Sampler"].get("fixed_params", []) or [])
priors, bounds, sampled_idx = build_priors(
    param_names, inj_params, fixed_names,
    n=float(cfg["Sampler"]["d"]),
    use_cupy=use_gpu,
)
fixed_idx = {
    param_names.index(n): inj_params[param_names.index(n)]
    for n in fixed_names if n in param_names and n != "x_I0"
}


class LogLikelihoodDiagonal:
    """
    Callable returning -0.5 <d - h | d - h> with a diagonal analytic PSD.

    Uses the complex detector-frame strain h+ - i hx directly: for a real
    PSD the inner product collapses to <h+|h+> + <hx|hx>, since the cross
    term (h+|hx) is purely imaginary.
    """
    def __init__(self, data_fft, psd, df, dt, wave, param_names,
                 fixed_params, x_I0_index, window, mask, x_I0_value=1.0):
        self.data_fft = data_fft
        self.psd = psd
        self.df = df
        self.dt = dt
        self.wave = wave
        self.param_names = param_names
        self.fixed_params = fixed_params
        self.x_I0_index = x_I0_index
        self.x_I0_value = x_I0_value
        self.window = window
        self.mask = mask

    def _expand(self, sampled):
        full = []
        k = 0
        for i in range(len(self.param_names)):
            if i == self.x_I0_index:
                full.append(self.x_I0_value)
            elif i in self.fixed_params:
                full.append(self.fixed_params[i])
            else:
                full.append(float(sampled[k]))
                k += 1
        return full

    def __call__(self, sampled):
        full = self._expand(sampled)
        try:
            h_td = xp.asarray(self.wave(*full))
            h_fft = xp.fft.rfft(h_td * self.window)[self.mask]
            diff = h_fft - self.data_fft
            return -0.5 * float(inner_product(diff, diff, self.psd, self.df, self.dt))
        except Exception as exc:
            logger.warning(f"Likelihood failed: {exc}")
            for name, val in zip(self.param_names, full):
                logger.warning(f"  {name} = {val:.6e}")
            # large negative value -> barrier around failed points
            return np.float32(-1e5)


llike = LogLikelihoodDiagonal(
    data_fft=h_inj_fft,
    psd=Sn,
    df=df,
    dt=DT,
    wave=rec_wave,
    param_names=param_names,
    fixed_params=fixed_idx,
    x_I0_index=x_I0_index,
    window=window,
    mask=mask,
)

# Consistency checks
logger.info("Running consistency checks")
snr = float(np.sqrt(inner_product(h_inj_fft, h_inj_fft, Sn, df, DT)))
ll_truth = float(llike([inj_params[i] for i in sampled_idx]))

logger.info(f"SNR = {snr:.2f}")
logger.info(f"loglike at truth = {ll_truth:.3e}")
if snr < 20:
    logger.warning("Injection SNR below 20 -- may not be recoverable.")

# Starting positions (same tight-ball convention as PE_response.py)
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

# Run log + backend paths
home = os.getcwd()
data_dir = cfg["Sampler"]["sampling_data_path"]
timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
fixed_tag = "_".join(f"{n}_{cfg['Injection']['EMRI'][n]}" for n in fixed_names)
tag = f"{cfg['Sampler']['name']}_fixed{fixed_tag}_{timestamp}"
out_dir = os.path.join(home, "..", "..", data_dir)
os.makedirs(out_dir, exist_ok=True)
fp = os.path.join(out_dir, f"SamplingResults_{tag}.h5")
log_fp = os.path.join(out_dir, f"RunLog_{tag}.txt")

with open(log_fp, "w") as f:
    f.write(f"# PE run log (no response) -- {timestamp}\n")
    f.write(f"Config: {args.config}\n")
    f.write(f"Injection model: {inj_wcfg.model} "
            f"(evolve_chi1={inj_wcfg.evolve_chi1}, "
            f"include_1PA_amps={inj_wcfg.include_1PA_amps})\n")
    f.write(f"Recovery  model: {rec_wcfg.model} "
            f"(evolve_chi1={rec_wcfg.evolve_chi1}, "
            f"include_1PA_amps={rec_wcfg.include_1PA_amps})\n")
    f.write(f"Fixed params: {fixed_names}\n")
    f.write(f"Sampler: ntemps={n_temps}, nwalkers={nwalkers}, "
            f"iterations={cfg['Sampler']['num_samples']}, "
            f"d={cfg['Sampler']['d']}\n")
    f.write(f"SNR (injection): {snr:.4f}\n")
    f.write(f"loglike at truth: {ll_truth:.6e}\n")
    f.write(f"Backend: {fp}\n")

# Diagnostic plot: injection strain vs LISA ASD
plots_dir = os.path.join(home, cfg["Sampler"].get("plots_path", "../../Plots"))
os.makedirs(plots_dir, exist_ok=True)

freqs_np = cp.asnumpy(freqs_inband) if use_gpu else np.asarray(freqs_inband)
h_fft_np = cp.asnumpy(h_inj_fft) if use_gpu else np.asarray(h_inj_fft)
psd_np = cp.asnumpy(Sn) if use_gpu else np.asarray(Sn)

fig, ax = plt.subplots(figsize=(9, 5))
ax.loglog(freqs_np, 2 * freqs_np * np.abs(h_fft_np),
          label="Injection strain (h+ - i hx)", alpha=0.8)
ax.loglog(freqs_np, np.sqrt(freqs_np * psd_np),
          label="LISA noise ASD", ls=":", color="k")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Characteristic strain")
ax.set_title(f"{cfg['Sampler']['name']} - no response")
ax.grid(which="both", alpha=0.4)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(plots_dir, f"{tag}_fd.png"), dpi=150)
plt.close(fig)
logger.info(f"Plots written to {plots_dir}")

# Sampler
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
    moves=StretchMove(a=2.0, use_gpu=use_gpu),
)

logger.info(f"Starting MCMC: {cfg['Sampler']['num_samples']} iterations -> {fp}")
ensemble.run_mcmc(start, int(cfg["Sampler"]["num_samples"]), progress=True)
logger.info("Done.")
