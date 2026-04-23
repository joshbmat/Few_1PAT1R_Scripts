'''
Main script to to PE run
'''

# Imports
import os 
import time
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from typing import Dict, Optional

import cupy as cp
import numpy as np
from few.waveform import GenerateEMRIWaveform
from lisatools.sensitivity import get_sensitivity, LISASens, CornishLISASens

# check for GPU availability
if not cp.is_available():
    xp = np
    logging.warning("GPU not available. Running on CPU, this will be very slow! ")
else:
    xp = cp

import argparse 

from scipy.signal.windows import tukey

# Import features from eryn
from eryn.ensemble import EnsembleSampler
from eryn.moves import StretchMove
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.backends import HDFBackend

# Load custom modules
from src.utils import zero_pad, inner_product 
from src.io import param_load

logger.info('Reading yaml file...')

parser = argparse.ArgumentParser()
parser.add_argument("--inference_params", type = str, help = "File with inference parameters", default=None)

args = parser.parse_args()
if args.inference_params is None:
    raise ValueError("Please provide a yaml file with parameters using --inference_params option.")
    
params = param_load(args.inference_params)
print(params)

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
    float(params['Waveform']['Phi_r0'])
]
logger.info('EMRI parameters read from yaml file.')
logger.info(f'EMRI parameters = {EMRI_params}')
logger.info("Reading sampler settings from yaml file...")
use_gpu = params['Sampler']['use_gpu']    

# define paths
data_dir = params['Sampler']['sampling_data_path']

# Read MCMC parameters from params
nwalkers = params['Sampler']['n_walkers']
n_temps = params['Sampler']['n_temps']
iterations = params['Sampler']['num_samples']
bruning = params['Sampler']['burn_in']
d = params['Sampler']['d']
continue_run = params['Sampler']['continue_run']

# Read waveform parameters
T = params['Waveform']['waveform_kwargs']['T']
DT = params['Waveform']['waveform_kwargs']['dt']
FS = 1/DT

# Read other settings
windowing = params['Sampler']['windowing']
filter_freq = params['Sampler']['filter_freq']
fixed_params = params['Sampler']['fixed_params']

logger.info('Sampler settings loaded')
logger.info('Checking GPU memory...')
def check_memory():
    free, total = cp.cuda.Device(0).mem_info
    print(f'Free memory  : {free/1e9:.2f} Gb\nUsed memory  : {(total-free)/1e9:.2f} Gb\nTotal memory : {total/1e9:.2f} Gb\n')
if use_gpu:
    check_memory()

param_names = ['M', 'mu', 'a', 'p0', 'e0', 'x_I0', 'd_L', 'theta_S', 'phi_S', 'theta_K', 'phi_K', 'Phi_phi0', 'Phi_theta0', 'Phi_r0']

param_indexing = {param_names[i]: i for i in range(len(param_names))}

fixed_param_values = {i: EMRI_params[i] for name, i in param_indexing.items() if name in fixed_params}
sampling_params = [i for i in range(len(param_names)) if i not in fixed_param_values.keys()]
##======================Waveform=======================
logger.info('Setting up waveform model...')


EMRI_waveform = GenerateEMRIWaveform("FastKerrEccentricEquatorialFlux", 
                                     inspiral_kwargs=inspiral_kwargs,
                                     sum_kwargs=summation_kwargs,
                                     amplitude_kwargs=amplitude_kwargs,
                                     return_list=True)

##======================Likelihood=====================
logger.info('Setting up likelihood function')

class loglikelihood:
    """
    Class to compute loglikelihood. We use a class here to be able to "fix" some parameters that we do not want to sample over. 
    The __call__ method allows us to call the class instance as a function, which is what the sampler expects. 
    """
    def __init__(self, 
                 data_f, 
                 PSD,
                 fixed_params: Dict[int, float] = {}, 
                 window: np.ndarray = 1,
                 mask: Optional[np.ndarray] = None
                 )-> None:
        self.fixed_params = fixed_params
        self.window = xp.asarray(window)
        if mask is not None:   
            self.mask = mask

        self.data_fft = data_f
        self.PSD = PSD

    def __call__(self, params):
        """
        Inputs: Parameters to sample over
        Outputs: log-whittle likelihood
        """
        few_params = []
        k = 0

        # Build full 14-parameter vector expected by FEW.
        # x_I0 (index 5) is fixed to 1.0 and not taken from sampled params.
        for i in range(len(param_names)):
            if i == 5:
                few_params.append(1.0)
            elif i in self.fixed_params:
                few_params.append(self.fixed_params[i])
            else:
                few_params.append(params[k])
                k += 1
        try:
            waveform_prop = EMRI_waveform(*few_params, **waveform_kwargs)

            # Taper and then zero pad. 
            EMRI_padded_plus = zero_pad(waveform_prop[0] * self.window)
            EMRI_padded_cross = zero_pad(waveform_prop[1] * self.window)

            # Compute in frequency domain
            EMRI_fft_plus = xp.fft.rfft(EMRI_padded_plus)[self.mask]
            EMRI_fft_cross = xp.fft.rfft(EMRI_padded_cross)[self.mask]

            # Compute (d - h| d- h)
            diff_f_plus = EMRI_fft_plus- self.data_fft[0]
            diff_f_cross = EMRI_fft_cross - self.data_fft[1]

            inn_prod_plus = inner_product(diff_f_plus,diff_f_plus, self.PSD, df, DT)
            inn_prod_cross = inner_product(diff_f_cross,diff_f_cross, self.PSD, df, DT)
            llike_val_np = xp.asnumpy(-0.5 * (inn_prod_plus + inn_prod_cross)) 
        except Exception as e:
            logger.info(f'Failed: {e}')
            # return very low loglikelihood to steer away from this area
            return np.float32(-10000)
        return (llike_val_np)

####=======================True  waveform==========================
logger.info("Generating True waveform...")
true_waveform = EMRI_waveform(*EMRI_params,
                         **waveform_kwargs)

####=======================True  waveform==========================
logger.info("Computing true SNR")

N_t_wav = len(true_waveform[0])
window = xp.asarray(tukey(N_t_wav, alpha=0.1)) if windowing else xp.ones(N_t_wav)

# Taper and then zero pad. 
EMRI_padded_plus = xp.asarray(zero_pad(true_waveform[0] * window))
EMRI_padded_cross = xp.asarray(zero_pad(true_waveform[1] * window))
N_t = len(EMRI_padded_plus)

# get frequencies and make mask
freq = xp.fft.rfftfreq(N_t, DT)

freq[0] = freq[1]   # To "retain" the zeroth frequency

if filter_freq:
    # filter frequencies between upper and lower bound
    upper_bound = 1/(2*DT) # nyquist limit
    lower_bound = 1e-5 # end of the LISA band

    # create boolean mask 
    mask = (freq >= lower_bound) & (freq <= upper_bound)
            
    # apply mask
    freqs = xp.asarray(freq[mask])
df = xp.diff(xp.asarray(freqs))

# get LISA SciRdv1 noise curve
Sn = get_sensitivity(freqs, sens_fn=LISASens, return_type="PSD")

# Compute in frequency domain
EMRI_fft_plus = xp.fft.rfft(EMRI_padded_plus)[mask]
EMRI_fft_cross = xp.fft.rfft(EMRI_padded_cross)[mask]

SNR2_true_plus = inner_product(EMRI_fft_plus, EMRI_fft_plus, Sn, df, DT)
SNR2_true_cross = inner_product(EMRI_fft_cross, EMRI_fft_cross, Sn, df, DT)
SNR2_true = SNR2_true_plus + SNR2_true_cross

SNR_true = np.sqrt(xp.asnumpy(SNR2_true))
print(f'    Optimal SNR of the signal = {float(SNR_true)}')

##===========================Likelihood tests============================
logger.info("Setting up likelihood object")

# define fixed parameters (if any) and window

fixed_params = {} # e.g. {0: M, 1: mu} to
##breakpoint()
llike = loglikelihood([EMRI_fft_plus, EMRI_fft_cross], 
                      Sn, 
                      window=window, 
                      mask=mask,
                      fixed_params=fixed_param_values)

validation_params = [
    float(params['Waveform']['M']), 
    float(params['Waveform']['mu']), 
    float(params['Waveform']['a']), 
    float(params['Waveform']['p0']),
    float(params['Waveform']['e0']), 
    # float(params['Waveform']['x_I0']), 
    float(params['Waveform']['d_L']), 
    float(params['Waveform']['theta_S']), 
    float(params['Waveform']['phi_S']), 
    float(params['Waveform']['theta_K']), 
    float(params['Waveform']['phi_K']), 
    float(params['Waveform']['Phi_phi0']),
    float(params['Waveform']['Phi_theta0']), 
    float(params['Waveform']['Phi_r0'])
]

if llike(validation_params) == 0.0:
    logger.info('Congrats, this is a good validation!')
else: 
    logger.info('Unfortunately, you have done something wrong :-(\n Go and debug your code. ')

##===========================MCMC Settings============================
logger.info("Setting up MCMC settings: starting points and temperatures")
tempering_kwargs=dict(ntemps=n_temps)  # Sampler requires the number of temperatures as a dictionary

# ================= SET UP PRIORS ========================
logger.info("Setting up priors")
#breakpoint()
Delta_theta_intrinsic = [100, 1e-3, 1e-4, 1e-4, 1e-4, 1e-3]  # M, mu, a, p0, e0 x0
Delta_theta_D = 0.5
n = d
epsilon_prior_width = 1e-12
# iterate over parameters to assign prior to each one. 
priors_in = {}
k = 0
for i, param_name in enumerate(param_names):
    if param_name in fixed_params:
        # we dont sample this parameters so no prior
        continue
    else:
        # add a tight prior around the true value, with respect to the waveform parameter space bounds
        if param_name in ['M', 'mu']:
            priors_in[k] = uniform_dist(max(EMRI_params[i] - n*Delta_theta_intrinsic[i], 2), EMRI_params[i] + n*Delta_theta_intrinsic[i])
        elif param_name in ['a']:
            if EMRI_params[i] >= 0:
                a_min = max(EMRI_params[i] - n*Delta_theta_intrinsic[i], 0.0)
                a_max = min(EMRI_params[i] + n*Delta_theta_intrinsic[i], 0.999)
                if a_max <= a_min:
                    a_max = min(a_min + max(Delta_theta_intrinsic[i], epsilon_prior_width), 0.999)
                    a_min = max(0.0, a_max - max(Delta_theta_intrinsic[i], epsilon_prior_width))
                priors_in[k] = uniform_dist(a_min, a_max)
            else:
                priors_in[k] = uniform_dist(max(EMRI_params[i] - n*Delta_theta_intrinsic[i], -0.999),
                                     min(EMRI_params[i] + n*Delta_theta_intrinsic[i], 0.0))
                
        elif param_name in ['e0']:
            e0 = params['Waveform']['e0']
            if e0 + n*Delta_theta_intrinsic[4] > 0.9:
                e0_max = 0.9
            else: 
                e0_max = e0 + 2*n*Delta_theta_intrinsic[4]
            e0_min = max(EMRI_params[i] - n*Delta_theta_intrinsic[i], 0.0)
            if e0_max <= e0_min:
                e0_max = min(e0_min + max(Delta_theta_intrinsic[i], epsilon_prior_width), 0.9)
                e0_min = max(0.0, e0_max - max(Delta_theta_intrinsic[i], epsilon_prior_width))
            priors_in[k] = uniform_dist(e0_min, e0_max)
            logger.info(f'Capped eccentricity prior between {e0_min} and {e0_max}')
        elif param_name in ['p0']:
            priors_in[k] = uniform_dist(EMRI_params[i] - n*Delta_theta_intrinsic[i], EMRI_params[i] + n*Delta_theta_intrinsic[i])
        elif param_name in ['x_I0']:
            logger.warning('Inclination is fixed to 1.')
            k -= 1
        elif param_name in ['d_L']:
            priors_in[k] = uniform_dist(max(EMRI_params[i] - n*Delta_theta_D, 0.1), EMRI_params[i] + n* Delta_theta_D)
        elif param_name in ['theta_S', 'theta_K']:
            priors_in[k] = uniform_dist(0, np.pi)
        else:
            priors_in[k] = uniform_dist(0, 2*np.pi)
        logger.info(f'Prior for {param_name} added')
        k += 1
logger.info(f'Priors are set: {priors_in}')
priors = ProbDistContainer(priors_in, use_cupy = True)   # Set up priors so they can be used with the sampler.

# Instantiate starting points only after priors are defined to ensure all walkers are in support.
# We add constraint that walkers must be within extremely small distance from the true value, to accelaret convergence
# Generate starting points constrained to be very close to true values
epsilon = 1e-6  # Small deviation from true values
constrained_start = []
max_attempts = 1000

for walker in range(nwalkers):
    for attempt in range(max_attempts):
        candidate = priors.rvs(size=(1,))[0]
        # Check if candidate is close enough to true values
        if np.all(np.abs(candidate - EMRI_params[sampling_params]) < epsilon):
            constrained_start.append(candidate)
            break
    else:
        # If max attempts exceeded, use closest valid point
        logger.warning(f"Walker {walker} exceeded max attempts, using closest valid point")
        constrained_start.append(candidate)

start = np.asarray(constrained_start)
start = priors.rvs(size=(nwalkers,))

if n_temps > 1:
    # For parallel tempering, initialize each temperature with valid points from the same prior support.
    start = np.asarray([priors.rvs(size=(nwalkers,)) for _ in range(n_temps)])

if np.size(start.shape) == 1:
    start = start.reshape(start.shape[-1], 1)
    ndim = 1
else:
    ndim = start.shape[-1]

start_logpdf = np.asarray(priors.logpdf(np.asarray(start)))
if not np.all(np.isfinite(start_logpdf)):
    raise ValueError("Initial walker positions contain points outside prior support.")

# =================== SET UP PROPOSAL ==================
#breakpoint()
moves_stretch = StretchMove(a=2.0, use_gpu=True)

# Quick checks
if n_temps > 1:
    print("Value of starting log-likelihood points", llike(start[0][0])) 
    if np.isinf(sum(priors.logpdf(np.asarray(start[0])))):
        logger.info("You are outside the prior range, you fucked up")
        quit()
else:
    for k in range(start.shape[0]):
        print("Value of starting log-likelihood points", llike(start[k])) 
        
#breakpoint()
home_folder = os.getcwd()
fixedparam_str = "_".join([f"{param_name}_{params['Waveform'][param_name]}" for param_name in fixed_params])

timestamp_str = time.strftime("%Y-%m-%d_%H-%M-%S")

fp = f"{home_folder}/../../{data_dir}/SamplingResults_{str(params['Sampler']['name'])}_fixed{fixedparam_str}_{timestamp_str}.h5"
backend = HDFBackend(fp)
logger.info(f"Using backend file: {fp}")
logger.info(f'Set up sampler and run!')


if continue_run:
    # check for existing backend file
    if os.path.exists(fp):
        logger.info("Found existing backend file, continuing from last sample")
        backend = HDFBackend(fp) # Set up backend

        start = backend.get_last_sample() # Start from last sample
    else:
        logger.info("No existing backend file found, starting fresh sampling run")
    ensemble = EnsembleSampler(
                            nwalkers,          
                            ndim,
                            llike,
                            priors,
                            backend = backend,                 # Store samples to a .h5 file
                            tempering_kwargs=tempering_kwargs,  # Allow tempering!
                            moves = moves_stretch
                            )
else:
    logger.info("Resetting backend and starting fresh sampling run")
    ensemble = EnsembleSampler(
                            nwalkers,          
                            ndim,
                            llike,
                            priors,
                            backend = backend,                 # Store samples to a .h5 file
                            tempering_kwargs=tempering_kwargs,  # Allow tempering!
                            moves = moves_stretch
                            )

out = ensemble.run_mcmc(start, iterations, progress=True)  # Run the sampler
