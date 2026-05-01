import numpy as np
import matplotlib.pyplot as plt
from few.waveform import Waveform1PAT1R
from few.trajectory.ode.IPAT1R import Trajectory1PAT1R

import argparse
import h5py
from concurrent.futures import ProcessPoolExecutor, as_completed

parser = argparse.ArgumentParser()
parser.add_argument("--tyr",        type=float, default=1.0)
parser.add_argument("--dt",         type=float, default=5.0)
parser.add_argument("--ntest",      type=int,   default=2000)
parser.add_argument("--nworkers",   type=int,   default=None)
parser.add_argument("--checkpoint", type=int,   default=100)
args = parser.parse_args()

# Set bounds
ode = Trajectory1PAT1R()
min_logm = -1.0
max_logm = 10.0
min_chi1 = ode._min_chi1
max_chi1 = ode._max_chi1
min_chi2 = ode._min_chi2
max_chi2 = ode._max_chi2

# Sample parameters
N_TESTS = args.ntest
dt = args.dt
T  = args.tyr
CHECKPOINT_EVERY = args.checkpoint

logm1 = np.random.uniform(min_logm, max_logm, N_TESTS)
logm2 = np.random.uniform(min_logm, max_logm, N_TESTS)
m1 = np.array([10.**np.max([logm1[i], logm2[i]]) for i in range(N_TESTS)])
m2 = np.array([10.**np.min([logm1[i], logm2[i]]) for i in range(N_TESTS)])
nu    = m1 * m2 / (m1 + m2)**2.
chi1  = np.random.uniform(min_chi1, max_chi1, N_TESTS)
chi2  = np.random.uniform(min_chi2, max_chi2, N_TESTS)
p     = np.array([np.random.uniform(*ode.bounds_p(0.0, 1.0, chi1[i], p_buffer=[0.01, 0.0])) for i in range(N_TESTS)])
theta = np.random.uniform(0, np.pi,   N_TESTS)
phi   = np.random.uniform(0, 2*np.pi, N_TESTS)

param_names  = ["m1", "m2", "nu", "chi1", "chi2", "p", "theta", "phi"]
param_arrays = [ m1,   m2,   nu,   chi1,   chi2,   p,   theta,   phi ]


# Instantiate waveform generator
wf = Waveform1PAT1R()

def test_waveform(i, m1_i, m2_i, chi1_i, p_i, theta_i, phi_i, chi2_i, dt, T):
    try:
        _ = wf(m1_i, m2_i, chi1_i, p_i, theta_i, phi_i, chi2_i, dt=dt, T=T)
        return i, True, None
    except ValueError as e:
        return i, False, str(e)


def save_params_h5(filepath, param_names, param_arrays, indices_mask):
    """Save selected rows (by boolean mask) of each parameter array to an HDF5 file."""
    n_samples = indices_mask.sum()
    if n_samples == 0:
        print(f"  No samples to save for {filepath}, skipping.")
        return
    with h5py.File(filepath, "w") as f:
        f.attrs["n_samples"] = n_samples
        f.attrs["T_yr"]      = T
        f.attrs["dt_s"]      = dt
        for name, arr in zip(param_names, param_arrays):
            f.create_dataset(name, data=arr[indices_mask])
    print(f"  Saved {n_samples} samples to {filepath}")


def compute_masks(completed_mask, failed_mask):
    """Derive success mask from completed and failed masks."""
    return completed_mask & ~failed_mask


def save_checkpoint(tag, completed_mask, failed_mask, param_names, param_arrays):
    """Overwrite a single pair of checkpoint HDF5 files each time."""
    success_mask = compute_masks(completed_mask, failed_mask)
    save_params_h5(f"checkpoint_success_{tag}.h5", param_names, param_arrays, success_mask)
    save_params_h5(f"checkpoint_failed_{tag}.h5",  param_names, param_arrays, failed_mask)


def plot_interim(tag, completed_mask, failed_mask, N_TESTS):
    """Overwrite a single interim plot each checkpoint."""
    success_mask = compute_masks(completed_mask, failed_mask)

    nu_f,  chi1_f,  chi2_f,  p_f  = nu[failed_mask],  chi1[failed_mask],  chi2[failed_mask],  p[failed_mask]
    nu_s,  chi1_s,  chi2_s,  p_s  = nu[success_mask], chi1[success_mask], chi2[success_mask], p[success_mask]

    n_done = completed_mask.sum()
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    try:
        fig.suptitle(f"Interim results: {n_done}/{N_TESTS} samples completed "
                     f"({failed_mask.sum()} failed, {success_mask.sum()} succeeded)")

        def scatter_pair(ax, x_f, y_f, x_s, y_s, xlabel, ylabel):
            ax.scatter(x_f, y_f, color='red',   label='Failed',    s=10)
            ax.scatter(x_s, y_s, color='green', label='Succeeded', s=10)
            ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.legend()

        scatter_pair(axs[0,0], np.log10(nu_f+1e-300), chi1_f, np.log10(nu_s+1e-300), chi1_s, 'log10(nu)', 'chi1')
        scatter_pair(axs[0,1], np.log10(nu_f+1e-300), chi2_f, np.log10(nu_s+1e-300), chi2_s, 'log10(nu)', 'chi2')
        scatter_pair(axs[0,2], np.log10(nu_f+1e-300), p_f,    np.log10(nu_s+1e-300), p_s,    'log10(nu)', 'p0')
        scatter_pair(axs[1,0], chi1_f, chi2_f, chi1_s, chi2_s, 'chi1', 'chi2')
        scatter_pair(axs[1,1], chi1_f, p_f,    chi1_s, p_s,    'chi1', 'p0')
        scatter_pair(axs[1,2], chi2_f, p_f,    chi2_s, p_s,    'chi2', 'p0')

        plt.tight_layout()
        plt.savefig(f"pass_fail_plot_{tag}_interim.png")
    finally:
        plt.close(fig)


if __name__ == "__main__":
    tag = f"T{T:.0f}_dt{dt:.0f}_N{N_TESTS}"

    failed_mask    = np.zeros(N_TESTS, dtype=bool)
    completed_mask = np.zeros(N_TESTS, dtype=bool)
    
    args_iter = [
        (i, m1[i], m2[i], chi1[i], p[i], theta[i], phi[i], chi2[i], dt, T)
        for i in range(N_TESTS)
    ]

    with ProcessPoolExecutor(max_workers=args.nworkers) as executor:
        futures = {executor.submit(test_waveform, *a): a[0] for a in args_iter}
        for future in as_completed(futures):
            i, success, err = future.result()
            completed_mask[i] = True
            if not success:
                print(f"  -> ValueError at i={i}: {err}")
                failed_mask[i] = True

            if completed_mask.sum() % CHECKPOINT_EVERY == 0:
                n_done = completed_mask.sum()
                print(f"\n--- Checkpoint at {n_done}/{N_TESTS} ---")
                save_checkpoint(tag, completed_mask, failed_mask, param_names, param_arrays)
                plot_interim(tag, completed_mask, failed_mask, N_TESTS)

    # Final save
    print(f"\nFailed indices: {np.where(failed_mask)[0].tolist()}")
    save_params_h5(f"params_success_{tag}.h5", param_names, param_arrays, ~failed_mask)
    save_params_h5(f"params_failed_{tag}.h5",  param_names, param_arrays,  failed_mask)

    # Final plot (reuses interim function, overwrites interim png)
    plot_interim(tag, completed_mask, failed_mask, N_TESTS)