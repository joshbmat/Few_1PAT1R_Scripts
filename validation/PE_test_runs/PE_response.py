'''
Code to run PE validation with fastlisaresponse using a self-consistent injection.
The injected dataset is generated from the same waveform/response model used for recovery.
'''

import os
import time
import argparse
import logging
from typing import Dict, Optional

import numpy as np
import cupy as cp
from scipy.signal.windows import tukey
import glob
import h5py

from lisaconstants import ASTRONOMICAL_YEAR
from lisaorbits import OEMOrbits

from fastlisaresponse import ResponseWrapper
from fastlisaresponse.tdiconfig import TDIConfig
from fastlisaresponse.utils.parallelbase import ParallelModuleBase
from few.waveform import GenerateEMRIWaveform
from lisatools.detector import Orbits
from lisatools.sensitivity import get_sensitivity, LISASens
from scipy.interpolate import CubicSpline

from eryn.ensemble import EnsembleSampler
from eryn.moves import StretchMove
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.backends import HDFBackend

from src.io import param_load
from mojito import MojitoL1File

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument("--inference_params", type=str, default=None,
                    help="YAML file with inference parameters (same format as sample.py).")
parser.add_argument("--test_case", type=int, default=1,
                    help="Used only when --inference_params is not provided.")
parser.add_argument("--orbit_file", type=str,
                    default="/data/leuven/367/vsc36785/LISA/Mojito_analysis/esa-trailing-orbits-mojito_validation_test_2.h5",
                    help="Path to orbit file used by fastlisaresponse.")
parser.add_argument("--cluster", type=str, default="vsc", choices=["vsc", "puhti"],
                    help="Cluster key used to discover a Mojito L1 file when --mojito_l1_file is not set.")
parser.add_argument("--mojito_l1_file", type=str, default=None,
                    help="Optional explicit Mojito L1 file path used to read t0 for timing.")
args = parser.parse_args()

if args.inference_params is None:
    here = os.path.dirname(os.path.abspath(__file__))
    args.inference_params = os.path.join(here, "config", f"config_test_{args.test_case}.yaml")

if not os.path.exists(args.inference_params):
    raise FileNotFoundError(f"Could not find inference params file: {args.inference_params}")

if args.mojito_l1_file is None:
    if args.cluster == "vsc":
        scratch = "/scratch/leuven/367/vsc36785/MojitoLight/SIM_data/brickmarket/mojito_light_v1_0_0/data/EMRI/L1"
    else:
        scratch = "/scratch/project_2004833/common_data/mojito/brickmarket/mojito_v1_0_0/data/EMRI/L1_0p4Hz"
    pattern = os.path.join(scratch, f"EMRI_731d_2.5s_L1_source0_*.h5")
    mojito_candidates = sorted(glob.glob(pattern))
    if not mojito_candidates:
        raise FileNotFoundError(f"No Mojito L1 files found for pattern: {pattern}")
    args.mojito_l1_file = mojito_candidates[-1]

if not os.path.exists(args.mojito_l1_file):
    raise FileNotFoundError(f"Could not find Mojito L1 file: {args.mojito_l1_file}")

with MojitoL1File(args.mojito_l1_file) as _l1:
    _time_sampling = _l1.tdis.time_sampling
    t0_l1 = float(_time_sampling.t0)
    mojito_dt = float(_time_sampling.dt)
    CENTRAL_FREQ = _l1.laser_frequency


logger.info(f"Loaded Mojito timing from {args.mojito_l1_file}: t0_l1={t0_l1}, dt={mojito_dt}")

params = param_load(args.inference_params)
logger.info(f"Loaded inference params from {args.inference_params}")

inspiral_kwargs = params['Waveform']['inspiral_kwargs']
summation_kwargs = params['Waveform']['summation_kwargs']
amplitude_kwargs = params['Waveform']['amplitude_kwargs']
waveform_kwargs = params['Waveform']['waveform_kwargs']

EMRI_params = [
    float(params['Waveform']['M']),
    float(params['Waveform']['mu']),
    float(params['Waveform']['a']),
    float(params['Waveform']['p0']),
    float(params['Waveform']['e0']),
    float(params['Waveform']['x_I0']),
    float(params['Waveform']['d_L']),
    float(params['Waveform']['theta_S']),
    float(params['Waveform']['phi_S']),
    float(params['Waveform']['theta_K']),
    float(params['Waveform']['phi_K']),
    float(params['Waveform']['Phi_phi0']),
    float(params['Waveform']['Phi_theta0']),
    float(params['Waveform']['Phi_r0']),
]

param_names = [
    'M', 'mu', 'a', 'p0', 'e0', 'chi2', 'x_I0', 'd_L',
    'theta_S', 'phi_S', 'theta_K', 'phi_K', 'Phi_phi0', 'Phi_theta0', 'Phi_r0'
]
param_indexing = {param_names[i]: i for i in range(len(param_names))}

logger.info(f"EMRI parameters = {EMRI_params}")

# Sampler settings mirrored from sample.py
use_gpu = params['Sampler']['use_gpu']
data_dir = params['Sampler']['sampling_data_path']
nwalkers = params['Sampler']['n_walkers']
n_temps = params['Sampler']['n_temps']
iterations = params['Sampler']['num_samples']
d = params['Sampler']['d']
continue_run = params['Sampler']['continue_run']
windowing = params['Sampler']['windowing']
filter_freq = params['Sampler']['filter_freq']
fixed_params = list(params['Sampler']['fixed_params'])[0]
# if len(list(params['Sampler']['fixed_params'])) == 1:
#     fixed_params = list(params['Sampler']['fixed_params'])
    
for item in fixed_params:
    logger.info(f'We have fixed the parameter {item} = {params['Waveform'][str(item)]}')

fixed_param_values = {i: EMRI_params[i] for name, i in param_indexing.items() if name in fixed_params}
sampling_params = [i for i in range(len(param_names)) if i not in fixed_param_values.keys()]

if not cp.is_available() and use_gpu:
    logger.warning("GPU requested but not available. Falling back to CPU arrays where possible.")

def check_memory() -> None:
    if cp.is_available():
        free, total = cp.cuda.Device(0).mem_info
        print(f'Free memory  : {free/1e9:.2f} Gb\\nUsed memory  : {(total-free)/1e9:.2f} Gb\\nTotal memory : {total/1e9:.2f} Gb\\n')


class EMRIWave_base(ParallelModuleBase):
    def __init__(self, force_backend=None, 
                use_gpu=True, 
                 inspiral_kwargs=inspiral_kwargs,
                #  sum_kwargs=sum_kwargs,
                 amplitude_kwargs=amplitude_kwargs,
                 mode_selection_threshold=1e-5,
                 t_init=33568152.5,
                 t0_orbits=33568152.5,
                 dt=5, 
                 n_samples=1000,
                 offset=550, # seconds
                 time=2.0
                ):
                 
        super().__init__(force_backend=force_backend)
        
        self.use_gpu = use_gpu
        self.mode_threshold = mode_selection_threshold
        
        # Initialize waveform generator

        self.waveform_gen = GenerateEMRIWaveform(
                "FastKerrEccentricEquatorialFlux",
                return_list=False,    # returns hp - i*hx as a complex cupy array
                inspiral_kwargs=inspiral_kwargs,
                # sum_kwargs=sum_kwargs,
                amplitude_kwargs=amplitude_kwargs,
                frame="detector"
            )
        self.t_init = t_init
        self.t0_orbits = t0_orbits
    
    @classmethod
    def supported_backends(cls):
        return ["fastlisaresponse_" + _tmp for _tmp in cls.GPU_RECOMMENDED()]

    def __call__(self, *params, T=2, dt=5):
        '''
        Call FEW waveform model and return the strain as h_+ - ih_x
        '''
        # define correct time grid for waveform generation.
        waveform_kwargs['T'] = T
        waveform_kwargs['dt'] = dt
        waveform_kwargs['mode_selection_threshold'] = self.mode_threshold
        strain = self.waveform_gen(*params, **waveform_kwargs)

        return strain


logger.info("Setting up waveform + response")
DT = float(params['Waveform']['waveform_kwargs']['dt'])
T_config = float(params['Waveform']['waveform_kwargs']['T'])

oemorbits = "esa-trailing"
t_dltt_orbits = 10.0
oem_orbits = OEMOrbits.from_included(oemorbits)
t0_orbits = float(oem_orbits.t_start) + t_dltt_orbits

offset = 550.0
n_samples_delay = 1000
T_response = T_config + (2 * offset + 2 * n_samples_delay * DT) / ASTRONOMICAL_YEAR
# Crucial absolute timing anchor from Mojito L1 metadata.
t0_l0 = t0_l1 - n_samples_delay * mojito_dt
t_init = t0_l0 - offset

if not os.path.exists(args.orbit_file):
    raise FileNotFoundError(f"Could not find orbit file: {args.orbit_file}")

force_backend = "cuda12x" if use_gpu else None

emri_waveform = EMRIWave_base(use_gpu=use_gpu, 
                         mode_selection_threshold=0.0,
                         t0_orbits=t0_orbits,
                         t_init=t_init,
                         dt=DT, 
                         n_samples=n_samples_delay,
                         offset=offset, # seconds
                        )
esa = Orbits(
    filename=args.orbit_file,
    use_gpu=use_gpu,
    force_backend=force_backend,
    linear_interp_setup=False,
    t0=t0_orbits,
)

index_beta = 7
index_lambda = 8
tdi_kwargs_esa = dict(orbits=esa, order=40, tdi=TDIConfig('2nd generation'), tdi_chan="XYZ")

emri_TDI_list = ResponseWrapper(
    emri_waveform,
    T_response,
    DT,
    index_lambda,
    index_beta,
    t0=t_init,
    t_buffer=10000.0,
    flip_hx=True,
    force_backend=force_backend,
    remove_sky_coords=False,
    is_ecliptic_latitude=False,
    remove_garbage=False,
    **tdi_kwargs_esa,
)


def emri_TDI(*wf_params):
    return cp.asarray(emri_TDI_list(*wf_params))


# Create synthetic injected dataset using the same model used for recovery.
logger.info("Generating self-consistent injected TDI dataset from config true parameters")
xyz_data = emri_TDI(*EMRI_params)
window = cp.asarray(tukey(xyz_data.shape[1], alpha=0.01)) if windowing else cp.ones(xyz_data.shape[1])

N_t = xyz_data.shape[1]
freqs = cp.fft.rfftfreq(N_t, d=DT)
if filter_freq:
    f_min = 1e-5
    f_max = 1.0 / (2.0 * DT)
    mask = (freqs >= f_min) & (freqs <= f_max)
else:
    mask = cp.ones_like(freqs, dtype=bool)

freqs_inband = freqs[mask]
xyz_data_fft = cp.fft.rfft(xyz_data * window, axis=1)[:, mask]

# Load and process noise model
logger.info('Loading and initializing covariance matrices')
if args.cluster == 'vsc':
    noise_file = '/scratch/leuven/367/vsc36785/MojitoLight/SIM_data/brickmarket/mojito_light_v1_0_0/data/NOISE/L1/NOISE_731d_2.5s_L1_source0_0_20251206T220508924302Z.h5'
elif args.cluster == 'puhti': 
    noise_file = f"/scratch/project_2004833/common_data/mojito/brickmarket/mojito_v1_0_0/data/INSTRUMENT/L1_0p4Hz/NOISE_731d_2.5s_L1_source0_0_20251206T220508924302Z.h5"

with h5py.File(noise_file, "r") as f:
    xyz_noise_estimate = np.mean(f['noise_estimates/XYZ'][:], axis=0) / CENTRAL_FREQ**2
    fmin_noise_psd = f['noise_estimates/log_frequency_sampling'].attrs['fmin']
    fmax_noise_psd = f['noise_estimates/log_frequency_sampling'].attrs['fmax']
    size_noise_psd = f['noise_estimates/log_frequency_sampling'].attrs['size']

    noise_freqs = np.logspace(np.log10(fmin_noise_psd), np.log10(fmax_noise_psd), size_noise_psd)

# Interpolate noise curves
freqs_inband_np = np.asarray(freqs_inband.get())

splined_noise_psd = np.array([
    CubicSpline(noise_freqs, xyz_noise_estimate[:, i, i])(freqs_inband_np) for i in range(3)
])

splined_noise_csd_real = np.array([
    CubicSpline(noise_freqs, xyz_noise_estimate[:, i, j].real)(freqs_inband_np) for i in range(3) for j in range(i, 3)
])

splined_noise_psd_imag = np.array([
    CubicSpline(noise_freqs, xyz_noise_estimate[:, i, j].imag)(freqs_inband_np) for i in range(3) for j in range(i, 3)   
])


# now re-assemble the covariance matrix
covariance_matrices = np.zeros((3, 3, len(freqs_inband)), dtype=complex)
for i in range(3):
    covariance_matrices[i, i, :] = splined_noise_psd[i]
    for j in range(i+1, 3):
        covariance_matrices[i, j, :] = splined_noise_csd_real[i*3 + j - (i+1)*i//2] + 1j * splined_noise_psd_imag[i*3 + j - (i+1)*i//2]
        covariance_matrices[j, i, :] = np.conj(covariance_matrices[i, j, :])

invC = cp.asarray(np.linalg.inv(covariance_matrices.transpose(2, 0, 1)))
del covariance_matrices
pre_fact = 2 * DT / N_t
invC *= pre_fact

n_freq = freqs_inband.shape[0]



def inner_prod_tdi(a_fft, b_fft, cov_inv_matrices):
    a_fft_T = a_fft.T
    b_fft_T = b_fft.T
    inner_per_freq = cp.einsum('fj,fjk,fk->f', cp.conj(a_fft_T), cov_inv_matrices, b_fft_T)
    return 2.0 * cp.real(cp.sum(inner_per_freq))


class loglikelihood:
    """
    TDI likelihood with optional fixed parameters, following sample.py style.
    """
    def __init__(self,
                 data_f,
                 cov_inv,
                 fixed_params: Dict[int, float] = {},
                 window_arr: np.ndarray = 1,
                 mask_arr: Optional[np.ndarray] = None) -> None:
        self.fixed_params = fixed_params
        self.window = cp.asarray(window_arr)
        self.mask = mask_arr
        self.data_fft = data_f
        self.cov_inv = cov_inv

    def __call__(self, params_vec):
        few_params = []
        k = 0
        for i in range(len(param_names)):
            if i == 5:
                few_params.append(1.0)
            elif i in self.fixed_params:
                few_params.append(self.fixed_params[i])
            else:
                few_params.append(float(params_vec[k]))
                k += 1
        try:
            waveform_prop = emri_TDI(*few_params)
            model_fft = cp.fft.rfft(waveform_prop * self.window, axis=1)[:, self.mask]
            diff_fft = model_fft - self.data_fft
            inn_prod = inner_prod_tdi(diff_fft, diff_fft, self.cov_inv)
            return cp.asnumpy(-0.5 * inn_prod)
        except Exception as exc:
            logger.info(f'Failed: {exc}')
            # log parameters for which the likelihood failed
            logger.info(f'Likelihood evaluation failed for the parameters:')
            for k, param in enumerate(few_params):
                logger.info(f'{param_names[i]} = {param:.6e}')
            # return very low loglikelihood as barrier for walkers
            return np.float32(-100000)


llike = loglikelihood(
    xyz_data_fft,
    invC,
    window_arr=window,
    mask_arr=mask,
    fixed_params=fixed_param_values,
)

validation_params = [EMRI_params[i] for i in range(len(param_names)) if i != 5 and i not in fixed_param_values]
llike_val = float(llike(validation_params))
logger.info(f"Validation loglike at injection parameters: {llike_val:.6e}")
logger.info(f"SNr of the injected signal: {np.sqrt(inner_prod_tdi(xyz_data_fft, xyz_data_fft, invC)):.2f}")

if np.sqrt(inner_prod_tdi(xyz_data_fft, xyz_data_fft, invC))< 20:
    logger.warning("Injected signal has low SNR, may not be recoverable. Consider adjusting parameters to increase SNR for a more meaningful test.")

#==========Plotting
import matplotlib.pyplot as plt
channel_labels = ['X', 'Y', 'Z']
for i, ch in enumerate(channel_labels):
    test_case = str(params['Sampler']['name'][-1])
    plt.figure(figsize=(10, 6))
    plt.loglog(freqs_inband_np, 2*freqs_inband_np*np.abs(cp.asnumpy(xyz_data_fft[0].get())),
                label='Injected TDI X (FFT)', alpha=0.7)

    plt.loglog(freqs_inband_np, np.sqrt(freqs_inband_np*splined_noise_psd[0]), 
               label='Interpolated Noise PSD X', linestyle='--')
    plt.grid(which='both', alpha=0.5)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Characteristic Strain')
    plt.title(f'TDI {ch} Channel - Injected Signal and Noise PSD')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{os.getcwd()}/../../Plots/tes_{test_case}_tdi_{ch}_injection_and_noise_psd.png')
    plt.close()
# breakpoint()
# ================= SET UP PRIORS ========================
logger.info("Setting up priors")
Delta_theta_intrinsic = [100, 1e-3, 1e-4, 1e-4, 1e-4, 1e-3]
Delta_theta_D = 0.5
n = d
epsilon_prior_width = 1e-12

priors_in = {}
prior_bounds = {}  # k -> (lower, upper) for clipping start positions
k = 0
for i, param_name in enumerate(param_names):
    if param_name in fixed_params:
        continue
    if param_name in ['M', 'mu']:
        lo = max(EMRI_params[i] - n * Delta_theta_intrinsic[i], 2)
        hi = EMRI_params[i] + n * Delta_theta_intrinsic[i]
        priors_in[k] = uniform_dist(lo, hi)
    elif param_name in ['a']:
        if EMRI_params[i] >= 0:
            lo = max(EMRI_params[i] - n * Delta_theta_intrinsic[i], -0.999)
            hi = min(EMRI_params[i] + n * Delta_theta_intrinsic[i], 0.999)
            if hi <= lo:
                hi = min(lo + max(Delta_theta_intrinsic[i], epsilon_prior_width), 0.999)
                lo = max(0.0, hi - max(Delta_theta_intrinsic[i], epsilon_prior_width))
        else:
            lo = max(EMRI_params[i] - n * Delta_theta_intrinsic[i], -0.999)
            hi = min(EMRI_params[i] + n * Delta_theta_intrinsic[i], 0.0)
        priors_in[k] = uniform_dist(lo, hi)
    elif param_name in ['e0']:
        e0 = params['Waveform']['e0']
        if e0 + n * Delta_theta_intrinsic[4] > 0.9:
            hi = 0.9
        else:
            hi = e0 + 2 * n * Delta_theta_intrinsic[4]
        lo = max(EMRI_params[i] - n * Delta_theta_intrinsic[i], 0.0)
        if hi <= lo:
            hi = min(lo + max(Delta_theta_intrinsic[i], epsilon_prior_width), 0.9)
            lo = max(0.0, hi - max(Delta_theta_intrinsic[i], epsilon_prior_width))
        priors_in[k] = uniform_dist(lo, hi)
        logger.info(f'Capped eccentricity prior between {lo} and {hi}')
    elif param_name in ['p0']:
        lo = EMRI_params[i] - n * Delta_theta_intrinsic[i]
        hi = EMRI_params[i] + n * Delta_theta_intrinsic[i]
        priors_in[k] = uniform_dist(lo, hi)
    elif param_name in ['x_I0']:
        logger.warning('Inclination is fixed to 1.')
        k -= 1
    elif param_name in ['d_L']:
        lo = max(EMRI_params[i] - n * Delta_theta_D, 0.001)
        hi = EMRI_params[i] + n * Delta_theta_D
        priors_in[k] = uniform_dist(lo, hi)
    elif param_name in ['theta_S', 'theta_K']:
        lo, hi = 0.0, np.pi
        priors_in[k] = uniform_dist(lo, hi)
    else:
        lo, hi = 0.0, 2 * np.pi
        priors_in[k] = uniform_dist(lo, hi)
    prior_bounds[k] = (lo, hi)
    logger.info(f'Prior for {param_name} added')
    k += 1

priors = ProbDistContainer(priors_in, use_cupy=True)
logger.info(f'Priors are set: {priors_in}')

# Instantiate starting points close to the injected parameters
ndim = len(sampling_params)
start_val = np.array([EMRI_params[i] for i in sampling_params])

if n_temps > 1:
    start = np.array([
        [start_val + d * 1e-7 * np.random.randn(ndim) for _ in range(nwalkers)]
        for _ in range(n_temps)
    ])
    # Clip to stay within prior bounds for all temps and walkers
    for t in range(n_temps):
        for k in range(ndim):
            lo, hi = prior_bounds[k]
            start[t, :, k] = np.clip(start[t, :, k], lo, hi)
else:
    start = np.array([start_val + d * 1e-7 * np.random.randn(ndim) for _ in range(nwalkers)])
    # Clip to stay within prior bounds
    for k in range(ndim):
        lo, hi = prior_bounds[k]
        start[:, k] = np.clip(start[:, k], lo, hi)

if np.size(start.shape) == 1:
    start = start.reshape(start.shape[-1], 1)
    ndim = 1
else:
    ndim = start.shape[-1]

start_for_prior_check = np.asarray(start)
if start_for_prior_check.ndim == 3:
    start_for_prior_check = start_for_prior_check.reshape(-1, start_for_prior_check.shape[-1])
start_logpdf = np.asarray(priors.logpdf(start_for_prior_check))
if not np.all(np.isfinite(start_logpdf)):
    raise ValueError("Initial walker positions contain points outside prior support.")

# =================== SET UP PROPOSAL ==================
tempering_kwargs = dict(ntemps=n_temps)
moves_stretch = StretchMove(a=2.0, use_gpu=True)

if n_temps > 1:
    if np.isinf(sum(priors.logpdf(np.asarray(start[0])))):
        raise ValueError("Initial point is outside the prior range.")
    # log all starting value loglikelihood values for debugging
    for i in range(start.shape[0]):
        for j in range(start.shape[1]):
            logger.info(f"Value of starting log-likelihood point (temp {i}, walker {j}): {llike(start[i][j])}")
    
else:
    for i in range(start.shape[0]):
        # log all starting value loglikelihood values for debugging
        logger.info(f"Value of starting log-likelihood point: {llike(start[i])}")

# =================== BACKEND ==================
home_folder = os.getcwd()
fixedparam_str = "_".join([f"{param_name}_{params['Waveform'][param_name]}" for param_name in fixed_params])
timestamp_str = time.strftime("%Y-%m-%d_%H-%M-%S")

fp = f"{home_folder}/../../{data_dir}/SamplingResults_{str(params['Sampler']['name'])}_fixed{fixedparam_str}_{timestamp_str}.h5"
os.makedirs(os.path.dirname(fp), exist_ok=True)
backend = HDFBackend(fp)
logger.info(f"Using backend file: {fp}")

if continue_run:
    if os.path.exists(fp):
        logger.info("Found existing backend file, continuing from last sample")
        backend = HDFBackend(fp)
        start = backend.get_last_sample()
    else:
        logger.info("No existing backend file found, starting fresh sampling run")

ensemble = EnsembleSampler(
    nwalkers,
    ndim,
    llike,
    priors,
    backend=backend,
    tempering_kwargs=tempering_kwargs,
    moves=moves_stretch,
)

logger.info("Starting MCMC sampling...")
out = ensemble.run_mcmc(start, iterations, progress=True)
