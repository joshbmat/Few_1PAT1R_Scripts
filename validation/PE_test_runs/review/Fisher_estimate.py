#!/usr/bin/env python
# coding: utf-8

# ## Fisher estimates for review test cases
# 
# Goal: estimate Fisher information contained in the review test cases, to compare against PE run estimates.
# 
# Problem: installing the `stableemrifisher` package inside the PE search environment probably breaks this environment, which is not ideal as it was a pain to set up. First test notebook here with the pre-compiled binaries of the packages instead of the source-builds of specific commit messages.

# ### Environment
# 
# Run this with the **`few-1PAT1R`** kernel. Relevant versions there:
# 
# | package | version |
# |---|---|
# | `fastemriwaveforms` | dev install from `FEW/1PAT1R/FEW-dev` |
# | `fastlisaresponse` | 1.1.17 (pre-compiled wheel) |
# | `lisaanalysistools` | 1.2.8 (pre-compiled wheel) |
# | `stableemrifisher` | editable from `Projects/StableEMRIFisher` |
# 
# `fastlisaresponse` 1.1.17 is **not** the API used by `validation/PE_test_runs/src/waveform_updated.py`.
# Differences that matter here:
# 
# * `ResponseWrapper` takes `force_backend=("cpu"|"cuda11x"|"cuda12x")`, **not** `use_gpu=`.
#   Passing `use_gpu=` lands in `**kwargs` and is forwarded to `pyResponseTDI`, which rejects it.
# * `orbits=` must be an *instance* of `lisatools.detector.Orbits` (the signature's default is the
#   *class* `EqualArmlengthOrbits`, which would fail its own `isinstance` assert).
# * There is no `t_buffer` argument; `t0` is simply the garbage-removal buffer in seconds.
# * `__call__` returns a **list** of TDI channels.
# 
# Three fixes were needed to get this combination running; all of them are already applied:
# 
# 1. **Duplicate `libstdc++` abort.** The `fastlisaresponse` and `lisatools` wheels each vendor their
#    own copy of `libstdc++.6.dylib` (and `libgcc_s.1.1.dylib`) under `<pkg>/.dylibs/`. Loading both
#    C++ backends in one process makes dyld map two copies of the GNU C++ runtime and the process
#    dies with `Fatal Python error: Aborted` — in *either* import order. Fixed by pointing
#    `fastlisaresponse/.dylibs/*` at the `lisatools` copies:
# 
#    ```bash
#    SP=$(python -c "import site; print(site.getsitepackages()[0])")
#    cd $SP/fastlisaresponse/.dylibs
#    ln -sf ../../lisatools/.dylibs/libstdc++.6.dylib  libstdc++.6.dylib
#    ln -sf ../../lisatools/.dylibs/libgcc_s.1.1.dylib libgcc_s.1.1.dylib
#    ```
# 
#    **A `pip install --force-reinstall fastlisaresponse` will undo this and the abort comes back.**
# 
# 2. `stableemrifisher.noise` imported `lisatools` at module import time, so `import stableemrifisher`
#    died in any environment without it. The import is now deferred into `write_psd_file`.
# 
# 3. `stableemrifisher.fisher` built its default PSD path as `os.getcwd() + PSD_filename` (no
#    separator), dropping the file in the *parent* directory under a mangled name. Now `os.path.join`.
# 
# The cell below asserts fix 1 is in place before anything else is imported.

# In[1]:


import os
import site


# In[ ]:

# In[2]:


import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from lisaconstants import ASTRONOMICAL_YEAR
from few.utils.constants import YRSID_SI
from few.waveform import Circ1PAT1R, GenerateEMRIWaveform
from fastlisaresponse import ResponseWrapper
from lisatools.detector import EqualArmlengthOrbits, ESAOrbits, Orbits

from stableemrifisher.fisher import StableEMRIFisher

import fastlisaresponse
import lisatools

# ---- backend -------------------------------------------------------------
# Set USE_GPU by hand to override the autodetect.
try:
    import cupy as cp
    USE_GPU = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    cp = None
    USE_GPU = False

if USE_GPU:
    _cuda_major = cp.cuda.runtime.runtimeGetVersion() // 1000
    FORCE_BACKEND = "cuda11x" if _cuda_major < 12 else "cuda12x"
else:
    FORCE_BACKEND = "cpu"

xp = cp if USE_GPU else np

print("fastlisaresponse", fastlisaresponse.__version__)
print("lisatools       ", lisatools.__version__)
print("backend         ", FORCE_BACKEND)

CONFIG_DIR = Path("../config/review").resolve()
OUT_DIR = Path("fisher_out").resolve()
OUT_DIR.mkdir(exist_ok=True)
print("configs:", CONFIG_DIR)


# ### Review test-case parameters
# 
# Copied from `validation/PE_test_runs/config/review/Review_test_{1,2,3}_inj_1PA_rec_1PA.yaml`.
# 
# Two things are *not* verbatim copies of the YAML and are worth being explicit about:
# 
# * **Masses are redshifted.** `PE_response_updated.py::_emri_vector` multiplies `M` and `mu` by
#   `(1 + z)` before handing them to FEW, so the YAML holds source-frame masses and the waveform sees
#   detector-frame ones. The dicts below store the **detector-frame** values, i.e. what the sampler
#   actually conditions on. `d_L` is passed through unchanged (Gpc).
# * **Names are translated** to the `stableemrifisher` convention
#   (`M -> m1`, `mu -> m2`, `d_L -> dist`, `theta_S -> qS`, `phi_S -> phiS`, `theta_K -> qK`,
#   `phi_K -> phiK`), and the dict is ordered exactly as `GenerateEMRIWaveform` expects positionally.
#   `chi2` is *not* in the dict: FEW takes it as a trailing positional extra, which is what
#   `StableEMRIFisher`'s `add_param_args` produces.
# 
# `fisher_params` is the set of parameters the corresponding PE run actually samples: the full 1PA
# vector minus the YAML's `fixed_params`, minus `x_I0` (always held at 1.0).

# In[10]:


# Injection parameters for the three review test cases, in stableemrifisher naming and
# in FEW's positional order.  Masses are detector-frame, i.e. YAML value * (1 + z).

REVIEW_CASES = {
    1: dict(
        config="Review_test_1_inj_1PA_rec_1PA.yaml",
        z=0.004999974556005059,
        params={
            "m1": 5000000.0,               # 4975124.504066709 * (1 + z)
            "m2": 100.0,                   #      99.50249008133419 * (1 + z)
            "a": 8.2e-05,
            "p0": 8.8123,
            "e0": 0.0,
            "xI0": 1.0,
            "dist": 0.82213015032605164,
            "qS": 2.4314546363447995,
            "phiS": 2.757554564287996,
            "qK": 2.6973649175810768,
            "phiK": 4.381692553882582,
            "Phi_phi0": 0.5917337285168199,
            "Phi_theta0": 6.13001602516006,
            "Phi_r0": 4.782381792256834,
        },
        chi2=0.5,
        fixed=["x_I0", "e0", "Phi_theta0", "Phi_r0"],
    ),
    2: dict(
        config="Review_test_2_inj_1PA_rec_1PA.yaml",
        z=0.1,
        params={
            "m1": 549945.0549945006,       # 499950.0499950005 * (1 + z)
            "m2": 441.0890010998901,       #    400.9900009999 * (1 + z)
            "a": 0.0,
            "p0": 29.388286969253294,
            "e0": 0.0,
            "xI0": 1.0,
            "dist": 3.9425974141549913,
            "qS": 2.4314546363447995,
            "phiS": 2.757554564287996,
            "qK": 2.6973649175810768,
            "phiK": 4.381692553882582,
            "Phi_phi0": 0.5917337285168199,
            "Phi_theta0": 6.13001602516006,
            "Phi_r0": 4.782381792256834,
        },
        chi2=0.9,
        fixed=["x_I0", "e0", "a", "Phi_theta0", "Phi_r0"],
    ),
    3: dict(
        config="Review_test_3_inj_1PA_rec_1PA.yaml",
        z=0.01,
        params={
            "m1": 5044955.666098902,       # 4995005.609998913 * (1 + z)
            "m2": 504.49556660989015,      #  499.5005609998912 * (1 + z)
            "a": 0.0,
            "p0": 11.288286969253296,
            "e0": 0.0,
            "xI0": 1.0,
            "dist": 0.4429032864588028,
            "qS": 1.89,
            "phiS": 0.67,
            "qK": 1.7753,
            "phiK": 3.82,
            "Phi_phi0": 0.5917337285168199,
            "Phi_theta0": 6.13001602516006,
            "Phi_r0": 4.782381792256834,
        },
        chi2=0.998,
        fixed=["x_I0", "e0", "Phi_theta0", "Phi_r0"],
    ),
}

# Waveform / response settings, identical across the three configs.
T_OBS = 2.0          # years
DT = 5.0             # s
EVOLVE_CHI1 = True
INCLUDE_1PA_AMPS = True
INSPIRAL_KWARGS = {"DENSE_STEPPING": 0, "max_init_len": 1000}
SUMMATION_KWARGS = {"pad_output": True}
AMPLITUDE_KWARGS = {}
RESPONSE_ORDER = 40  # "order" in the Response block

# YAML name -> stableemrifisher name, in FEW's positional order for the 1PA model.
SEF_NAME = {
    "M": "m1", "mu": "m2", "a": "a", "p0": "p0", "e0": "e0", "chi2": "chi2",
    "x_I0": "xI0", "d_L": "dist", "theta_S": "qS", "phi_S": "phiS",
    "theta_K": "qK", "phi_K": "phiK",
    "Phi_phi0": "Phi_phi0", "Phi_theta0": "Phi_theta0", "Phi_r0": "Phi_r0",
}
PARAM_NAMES_1PA = list(SEF_NAME)


def fisher_params(case):
    """Parameters the matching PE run samples: full vector minus fixed minus x_I0."""
    fixed = set(case["fixed"]) | {"x_I0"}
    return [SEF_NAME[n] for n in PARAM_NAMES_1PA if n not in fixed]


for i, c in REVIEW_CASES.items():
    print(f"case {i}: {len(fisher_params(c)):2d} params  {fisher_params(c)}")


# #### Cross-check against the YAML files
# 
# Guards against the hard-coded dicts above drifting away from the configs.

# In[11]:


def check_against_yaml(case):
    cfg = yaml.safe_load((CONFIG_DIR / case["config"]).read_text())
    emri = cfg["Injection"]["EMRI"]
    wave = cfg["Injection"]["Waveform"]
    z = float(emri["z"])

    assert z == case["z"], f"z: {z} != {case['z']}"
    assert float(emri["chi2"]) == case["chi2"]
    assert sorted(cfg["Sampler"]["fixed_params"]) == sorted(case["fixed"])

    for yaml_name, sef_name in SEF_NAME.items():
        if yaml_name == "chi2":
            continue
        expected = float(emri[yaml_name])
        if yaml_name in ("M", "mu"):
            expected *= 1.0 + z
        got = case["params"][sef_name]
        assert np.isclose(got, expected, rtol=0, atol=0), \
            f"{yaml_name}/{sef_name}: {got!r} != {expected!r}"

    assert float(wave["T"]) == T_OBS and float(wave["dt"]) == DT
    assert bool(wave["evolve_chi1"]) is EVOLVE_CHI1
    assert bool(wave["include_1PA_amps"]) is INCLUDE_1PA_AMPS
    assert int(cfg["Response"]["order"]) == RESPONSE_ORDER
    assert cfg["Response"]["tdi_gen"] == "2nd generation"
    return True


for i, c in REVIEW_CASES.items():
    check_against_yaml(c)
    print(f"case {i}: matches {c['config']}")


# ### Response configuration
# 
# This mirrors `validation/PE_test_runs/src/waveform_updated.py::build_response` and the timing block
# of `PE_response_updated.py`, so the Fisher sees the same response the PE runs do:
# 
# * orbits from the `Data.orbit_file` of the review configs (`Orbits(filename=...)`),
# * `T_response = T + (2 * offset + 2 * n_samples_delay * dt) / ASTRONOMICAL_YEAR`,
# * `t0 = t_init = t0_L1 - n_samples_delay * dt_mojito - offset`, with `t0_L1` read from the
#   Mojito L1 file,
# * `order = 40`, `tdi = "2nd generation"`, `flip_hx = True`, `is_ecliptic_latitude = False`,
#   `remove_sky_coords = False`, `remove_garbage = False`,
# * the Tukey window and in-band frequency mask the sampler applies (`windowing`, `filter_freq`).
# 
# The orbit file and the Mojito L1 file only exist on the cluster. Off the GPU node the notebook
# falls back to the `lisatools` bundled `ESAOrbits` and to a plain garbage buffer for `t0`, and says
# so loudly. Everything else is identical either way.
# 
# **One deliberate departure from the PE configuration**, forced by what a Fisher matrix can
# represent: **the channels**. The PE likelihood uses `XYZ` with the full 3x3 Mojito noise covariance
# (`src/noise.py::build_inv_covariance`). `StableEMRIFisher`'s inner product is per channel with a
# diagonal PSD, and X, Y and Z are strongly correlated, so `XYZ` here would badly misestimate the
# information. `AE` (or `AET`) is noise-orthogonal to a good approximation, which is what the diagonal
# form assumes. `TDI_CHAN` is set to `"AE"`; pass `tdi_chan="XYZ"` only together with your own
# `noise_model`.
# 
# The noise *itself* is still the measured Mojito estimate, rotated into AET -- see the next section.
# 
# `ResponseWrapper` is also handed a padded waveform generator. `fastlisaresponse` and FEW carry
# values of the sidereal year that differ in the last two digits
# (`31558149.763545603` vs `...595`), so `int(T * YRSID / dt)` can disagree by a sample and the
# response then gets a waveform one point short. `src/waveform_updated.py` solves this with
# `EMRIWave.min_output_length`; `PaddedEMRIWaveform` below is the same fix in a form
# `StableEMRIFisher` can instantiate itself.

# In[12]:


# ---- Data block, identical across the three review configs ----------------------------------
DATA = {
    "orbit_file": "/data/leuven/367/vsc36785/LISA/Mojito_analysis/"
                  "esa-trailing-orbits-mojito_validation_test_2.h5",
    "mojito_l1_file": "/scratch/leuven/367/vsc36785/MojitoLight/SIM_data/brickmarket/"
                      "mojito_light_v1_0_0/data/EMRI/L1/"
                      "EMRI_731d_2.5s_L1_source0_0_20251203T225446987631Z.h5",
    "noise_file": "/scratch/leuven/367/vsc36785/MojitoLight/SIM_data/brickmarket/"
                  "mojito_light_v1_0_0/data/NOISE/L1/"
                  "NOISE_731d_2.5s_L1_source0_0_20251206T220508924302Z.h5",
}

# ---- Response block, identical across the three review configs ------------------------------
RESPONSE = {
    "tdi_gen": "2nd generation",
    "tdi_chan": "XYZ",          # what the PE runs use; see TDI_CHAN below for what we use here
    "order": 40,
    "offset": 550.0,
    "n_samples_delay": 1000,
    "t_buffer": 10000.0,
    "flip_hx": True,
    "is_ecliptic_latitude": False,
    # ResponseConfig defaults in src/waveform_updated.py; not set in the YAML
    "remove_sky_coords": False,
    "remove_garbage": False,
}

# ---- Sampler block ---------------------------------------------------------------------------
WINDOWING = True            # Sampler.windowing
FILTER_FREQ = True          # Sampler.filter_freq
TUKEY_ALPHA = 0.01          # PE_response_updated.py
F_MIN = 1e-5                # src/utils.py::inband_freqs default

# ---- Fisher-specific choices -----------------------------------------------------------------
TDI_CHAN = "AE"             # not XYZ: see the note above
T0_FALLBACK = 10_000.0      # used for t0 when the Mojito L1 file is unreachable


def mojito_timing(l1_file=None):
    """
    (t0_L1, dt_mojito, central_freq) from the Mojito L1 file, as PE_response_updated.py reads them.

    Returns None when `mojito` is not installed or the file is not reachable, which is the normal
    situation anywhere other than the cluster.
    """
    l1_file = l1_file or DATA["mojito_l1_file"]
    try:
        from mojito import MojitoL1File
    except ImportError:
        return None
    if not os.path.exists(l1_file):
        return None
    with MojitoL1File(l1_file) as l1:
        ts = l1.tdis.time_sampling
        return float(ts.t0), float(ts.dt), float(l1.laser_frequency)


def response_timing(T=T_OBS, dt=DT, resp=RESPONSE, verbose=True):
    """
    (T_response, t_init) exactly as PE_response_updated.py derives them.

    T_response pads the requested observation time by the response buffer on both ends;
    t_init walks the L1 epoch back through the delay buffer and the offset.
    """
    T_response = T + (2 * resp["offset"] + 2 * resp["n_samples_delay"] * dt) / ASTRONOMICAL_YEAR

    timing = mojito_timing()
    if timing is None:
        if verbose:
            warnings.warn(
                f"Mojito L1 file unreachable -> using t0 = {T0_FALLBACK} s instead of the "
                "L1-derived t_init. The response epoch, and therefore the antenna pattern, will "
                "not match the PE runs. Expected off the GPU node.",
                stacklevel=2,
            )
        return T_response, T0_FALLBACK

    t0_l1, mojito_dt, _ = timing
    t0_l0 = t0_l1 - resp["n_samples_delay"] * mojito_dt
    t_init = t0_l0 - resp["offset"]
    if verbose:
        print(f"  Mojito L1: t0 = {t0_l1}, dt = {mojito_dt} -> t_init = {t_init}")
    return T_response, t_init


def build_orbits(orbit_file=None, verbose=True):
    """Orbits from the config's orbit file, falling back to the bundled ESA trailing orbits."""
    orbit_file = orbit_file or DATA["orbit_file"]
    if os.path.exists(orbit_file):
        if verbose:
            print(f"  orbits: {orbit_file}")
        return Orbits(filename=orbit_file, force_backend=FORCE_BACKEND,
                      linear_interp_setup=False)
    if verbose:
        warnings.warn(
            f"Orbit file not found ({orbit_file}) -> falling back to the lisatools bundled "
            "ESAOrbits. Expected off the GPU node.",
            stacklevel=2,
        )
    return ESAOrbits(force_backend=FORCE_BACKEND)


# #### Noise: measured Mojito PSDs
# 
# The PE likelihood uses the measured Mojito noise, so the Fisher should too. The noise file stores a
# 3x3 XYZ covariance (`noise_estimates/XYZ`, averaged over time bins and divided by the laser
# frequency squared, exactly as `src/noise.py::load_mojito_xyz_covariance` reads it). Rotating that
# covariance with the orthonormal XYZ -> AET transform and taking the diagonal gives the A, E and T
# PSDs, which is what `StableEMRIFisher`'s per-channel inner product wants. If the file exposes an
# `AE`/`AET` estimate directly, `mojito_aet_psd` picks that up instead and skips the rotation;
# `inspect_noise_file` prints the tree so you can check which route applies.
# 
# The rotation is the point of doing this in AET rather than XYZ: the XYZ covariance has large
# off-diagonal terms which a diagonal weighting would simply throw away, whereas in AET what is
# discarded is only the residual off-diagonal left by unequal arms.
# 
# **Dead frequencies.** The measured estimate has bins where it collapses towards zero (and some
# non-finite entries) -- the TDI transfer-function nulls sit inside the analysis band, and the
# estimator drops out here and there. A Fisher matrix weights by `1/S`, so those bins would each
# contribute enormous spurious information. `smooth_psd` handles this in log-log space:
# 
# 1. a running median over the log-spaced frequency grid gives a baseline. A median is unbiased on
#    monotonic data, so it tracks the real spectral shape instead of flattening it, and it ignores
#    isolated dropouts;
# 2. bins more than `drop_tol` decades below, or `spike_tol` above, that baseline are flagged. The
#    baseline is then recomputed with the flagged bins excluded, and the whole thing iterates to a
#    fixed point. Before each re-interpolation the flagged mask is dilated by `reach` bins, so the
#    interpolation anchors sit *outside* a dropout rather than on its not-yet-flagged interior --
#    without that the mask never grows and contiguous dead blocks survive untouched;
# 3. a local median cannot see a dead block wider than its own window, so if the result still has a
#    notch the fit is retried once at `3 * width`, and kept only if it does better;
# 4. whatever is left is floored at `floor_decades` below a wide running median, which bounds `1/S`;
# 5. `mode="full"` returns the smooth baseline everywhere, `mode="repair"` keeps the measured value
#    wherever it was not flagged;
# 6. a running median estimates the *median* of a chi-squared-like PSD estimate, which sits below its
#    mean, so the baseline is rescaled by the mean ratio over the trusted bins (`debias`).
# 
# A dead region wide enough to defeat all of that cannot be repaired honestly, only papered over, so
# it is reported instead: any remaining bin-to-bin jump larger than `step_tol` decades raises a
# warning naming the frequencies. Interpolating across a hole that wide is a decision for you, not
# for this function.
# 
# The offline test two cells down checks this against a synthetic PSD of known truth, with 3%
# scattered dead bins, a contiguous dead block, NaNs and an upward spike. Dead blocks up to about
# three times `width` are repaired to a median accuracy of ~4% with the worst `1/S` overestimate held
# near 1.25, against 1.6e6 unrepaired; a clean PSD passes through with a maximum error of 0.1% and
# nothing flagged.

# In[13]:


import h5py
from scipy.ndimage import binary_dilation, median_filter

# XYZ -> AET orthonormal transform, as in src/noise.py.
_TO_AET = np.array([
    [-1.0 / np.sqrt(2),  0.0,              1.0 / np.sqrt(2)],
    [ 1.0 / np.sqrt(6), -2.0 / np.sqrt(6), 1.0 / np.sqrt(6)],
    [ 1.0 / np.sqrt(3),  1.0 / np.sqrt(3), 1.0 / np.sqrt(3)],
])

NOISE_SOURCE = "mojito"     # "mojito" = measured PSDs below; "analytic" = scirdv1 via SEF
SMOOTH_KWARGS = dict(width=21, drop_tol=0.5, spike_tol=1.0, reach=2,
                     mode="full", debias=True, floor_decades=1.0,
                     widen=True, step_tol=1.0, max_passes=60)


def inspect_noise_file(path=None, max_depth=3):
    """Print the HDF5 tree of a Mojito noise/L1 file, to check what estimates it carries."""
    path = path or DATA["noise_file"]
    with h5py.File(path, "r") as h5:
        def walk(name, obj, depth=0):
            pad = "  " * depth
            if isinstance(obj, h5py.Dataset):
                print(f"{pad}{name}  {obj.shape} {obj.dtype}")
            else:
                print(f"{pad}{name}/")
                for k, a in obj.attrs.items():
                    print(f"{pad}  .{k} = {a}")
                if depth < max_depth:
                    for k in obj:
                        walk(k, obj[k], depth + 1)
        walk(os.path.basename(path), h5)


def _log_freq_grid(attrs):
    return np.logspace(np.log10(attrs["fmin"]), np.log10(attrs["fmax"]), int(attrs["size"]))


def load_mojito_aet_raw(noise_file=None, central_freq=None, verbose=True):
    """
    (freqs, psd_aet) with psd_aet of shape (3, n_f), un-smoothed.

    Prefers an AE/AET estimate stored in the file; otherwise reads the XYZ covariance and rotates
    it, which is the same read `src/noise.py` performs.
    """
    noise_file = noise_file or DATA["noise_file"]
    if central_freq is None:
        timing = mojito_timing()
        if timing is None:
            raise FileNotFoundError(
                "Need the laser frequency from the Mojito L1 file to normalise the noise "
                "estimate, and that file is unreachable. Pass central_freq= explicitly."
            )
        central_freq = timing[2]

    with h5py.File(noise_file, "r") as h5:
        grp = h5["noise_estimates"]
        freqs = _log_freq_grid(grp["log_frequency_sampling"].attrs)
        direct = next((k for k in ("AET", "AE") if k in grp), None)

        if direct is not None:
            raw = np.asarray(grp[direct][:]) / central_freq**2
            # (n_t, n_f, k[, k]) -> average over time bins, as done for XYZ
            if raw.ndim in (3, 4):
                raw = np.mean(raw, axis=0)
            if raw.ndim == 3 and raw.shape[1] == raw.shape[2]:       # (n_f, k, k)
                psd = np.stack([raw[:, i, i].real for i in range(raw.shape[1])])
            elif raw.ndim == 2:                                       # (n_f, k)
                psd = raw.real.T
            else:
                warnings.warn(
                    f"noise_estimates/{direct} has unexpected shape {raw.shape}; "
                    "falling back to rotating the XYZ covariance.", stacklevel=2)
                direct = None
            if direct is not None:
                if verbose:
                    print(f"  noise: noise_estimates/{direct} ({psd.shape[0]} channels)")
                if psd.shape[0] == 2:      # A, E only -> pad T with NaN so indexing stays uniform
                    psd = np.vstack([psd, np.full(psd.shape[1], np.nan)])
                return freqs, psd

        cov_xyz = np.mean(np.asarray(grp["XYZ"][:]), axis=0) / central_freq**2

    # C_AET = U C_XYZ U^T ; the diagonal is real because C_XYZ is Hermitian
    cov_aet = np.einsum("ij,fjk,lk->fil", _TO_AET, cov_xyz, _TO_AET)
    psd = np.stack([cov_aet[:, i, i].real for i in range(3)])
    if verbose:
        print(f"  noise: rotated noise_estimates/XYZ -> AET ({len(freqs)} bins, "
              f"{freqs[0]:.2e} - {freqs[-1]:.2e} Hz)")

    # S_T is a near-cancellation of O(S_A) terms, so it is only recoverable from the XYZ
    # covariance where it stays above the double-precision noise floor of that subtraction.
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = psd[2] / np.maximum(np.abs(psd[0]), np.finfo(float).tiny)
    unresolved = np.isfinite(ratio) & (ratio < 1e-13)
    if unresolved.mean() > 0.01:
        f_bad = freqs[unresolved]
        warnings.warn(
            f"S_T / S_A falls below 1e-13 over {unresolved.mean() * 100:.0f}% of the grid "
            f"(f in [{f_bad.min():.2e}, {f_bad.max():.2e}] Hz), which is at or below what float64 "
            "can resolve when subtracting O(S_A) terms. The T channel is numerical noise there; "
            "use channels='AE'.",
            stacklevel=2,
        )
    return freqs, psd


# In[14]:


def _repair_pass(log_filled, good, width, drop_tol, spike_tol, max_passes, reach):
    """Iterate median baseline <-> outlier mask to a fixed point. Returns (baseline, bad)."""
    idx = np.arange(log_filled.size, dtype=float)
    struct = np.ones(2 * reach + 1, dtype=bool)

    bad = ~good
    baseline = median_filter(log_filled, size=width, mode="nearest")
    for _ in range(max_passes):
        # Dilating the mask before interpolating is what lets a dropout be eaten from its edges
        # inwards: the anchors then sit outside it, instead of on its own un-flagged interior.
        keep = ~binary_dilation(bad, struct) if bad.any() else np.ones_like(bad)
        if keep.sum() < width:
            keep = ~bad
        if keep.sum() >= 2:
            baseline = median_filter(
                np.interp(idx, idx[keep], log_filled[keep]), size=width, mode="nearest"
            )
        resid = log_filled - baseline
        new_bad = (~good) | (resid < -drop_tol) | (resid > spike_tol)
        if np.array_equal(new_bad, bad):
            break
        bad = new_bad
    return baseline, bad


def _notch_count(baseline, width, floor_decades):
    trend = median_filter(baseline, size=min(8 * width, baseline.size), mode="nearest")
    return int(np.sum(baseline - trend < -floor_decades))


def smooth_psd(freqs, psd, width=21, drop_tol=0.5, spike_tol=1.0, max_passes=60,
               mode="full", debias=True, reach=2, floor_decades=1.0,
               widen=True, step_tol=1.0, warn=True):
    """
    Repair the dead-frequency dropouts in a measured PSD, in log-log space.

    `freqs` must be ascending and is assumed log-spaced, which makes a fixed-width running median a
    fixed fraction of a decade.  See the markdown above for what each stage does.

    Returns a dict with keys `psd`, `bad`, `floored`, `scale`, `width`, `max_step`, `ok`.
    `ok` is False when a dropout was too wide to repair, in which case a warning names the
    frequencies and the result should not be trusted there.
    """
    freqs = np.asarray(freqs, dtype=float)
    psd = np.asarray(psd, dtype=float)
    if freqs.shape != psd.shape:
        raise ValueError(f"freqs {freqs.shape} and psd {psd.shape} must have the same shape")

    good = np.isfinite(psd) & (psd > 0.0)
    if good.sum() < max(width, 8):
        raise ValueError(f"only {good.sum()} usable PSD bins; cannot smooth")

    idx = np.arange(psd.size, dtype=float)
    log_psd = np.log10(np.where(good, psd, 1.0))
    log_filled = np.interp(idx, idx[good], log_psd[good])

    w = width
    baseline, bad = _repair_pass(log_filled, good, w, drop_tol, spike_tol, max_passes, reach)

    if widen and _notch_count(baseline, w, floor_decades):
        # a median filter is blind to a dropout wider than its own window; try once more, wider
        w2 = min((3 * w) | 1, psd.size // 8)
        if w2 > w:
            b2, bad2 = _repair_pass(log_filled, good, w2, drop_tol, spike_tol, max_passes, reach)
            if _notch_count(b2, w2, floor_decades) < _notch_count(baseline, w, floor_decades):
                baseline, bad, w = b2, bad2, w2

    scale = 1.0
    if debias and (~bad).any():
        # a running median tracks the median of a chi2-like estimate, which sits below its mean
        scale = float(np.mean(psd[~bad] / 10.0 ** baseline[~bad]))
        baseline = baseline + np.log10(scale)

    log_out = baseline if mode == "full" else np.where(bad, baseline, log_filled)

    trend = median_filter(log_out, size=min(8 * w, log_out.size), mode="nearest")
    floored = log_out < trend - floor_decades
    log_out = np.where(floored, trend - floor_decades, log_out)

    # A residual dead block leaves a cliff; a real PSD plus estimator scatter does not.
    steps = np.abs(np.diff(log_out))
    max_step = float(steps.max()) if steps.size else 0.0
    ok = max_step <= step_tol
    if not ok and warn:
        f_bad = freqs[:-1][steps > step_tol]
        warnings.warn(
            f"PSD repair did not converge: bin-to-bin jumps up to {max_step:.1f} decades remain "
            f"near f in [{f_bad.min():.3e}, {f_bad.max():.3e}] Hz. The dead region is wider than "
            f"the {w}-bin filter. Increase `width`, or exclude that band with fmin/fmax, rather "
            "than trusting the PSD there.",
            stacklevel=2,
        )

    return {
        "psd": 10.0 ** log_out, "bad": bad, "floored": floored,
        "scale": scale, "width": w, "max_step": max_step, "ok": ok,
    }


def _loglog_interp(freqs, values):
    """Linear interpolation in log10(f)-log10(S), clamped to the endpoints outside the grid."""
    log_f, log_v = np.log10(freqs), np.log10(values)
    f_lo, f_hi = freqs[0], freqs[-1]

    def evaluate(f):
        f = np.asarray(f, dtype=float)
        return 10.0 ** np.interp(np.log10(np.clip(f, f_lo, f_hi)), log_f, log_v)

    return evaluate


_PSD_CACHE = {}


def mojito_psd_model(channels="AE", noise_file=None, central_freq=None,
                     f_min=None, verbose=True, **smooth_kwargs):
    """
    Noise model for StableEMRIFisher built from the measured Mojito PSDs.

    Returns (psd_callable, freqs, psd_raw, psd_smoothed).  The callable has the signature
    `psd(f, channel="A")`, which is what `generate_PSD` calls once per entry of a list-valued
    `noise_kwargs`.
    """
    key = (channels, noise_file or DATA["noise_file"], tuple(sorted(smooth_kwargs.items())))
    if key in _PSD_CACHE:
        return _PSD_CACHE[key]

    opts = {**SMOOTH_KWARGS, **smooth_kwargs}
    freqs, raw = load_mojito_aet_raw(noise_file, central_freq, verbose=verbose)

    f_min = F_MIN if f_min is None else f_min
    if f_min < freqs[0]:
        warnings.warn(
            f"the inner product starts at {f_min:g} Hz but the noise estimate only reaches down "
            f"to {freqs[0]:.3e} Hz; below that the PSD is held flat, which understates the noise "
            "and so overstates the information. Raise F_MIN or restrict fmin per call.",
            stacklevel=2,
        )

    index = {"A": 0, "E": 1, "T": 2}
    smoothed = np.full_like(raw, np.nan)
    interps = {}
    for c in channels:
        i = index[c]
        if not np.any(np.isfinite(raw[i])):
            raise ValueError(f"no usable estimate for channel {c}")
        res = smooth_psd(freqs, raw[i], **opts)
        smoothed[i] = res["psd"]
        interps[c] = _loglog_interp(freqs, res["psd"])
        if verbose:
            flag = "" if res["ok"] else "  [!! unrepaired notch]"
            print(f"  channel {c}: repaired {res['bad'].sum()}/{len(freqs)} bins, "
                  f"floored {res['floored'].sum()}, width {res['width']}, "
                  f"debias x{res['scale']:.4f}{flag}")

    def psd(f, channel="A"):
        return interps[channel](f)

    result = (psd, freqs, raw, smoothed)
    _PSD_CACHE[key] = result
    return result


def noise_for(tdi_chan, verbose=True):
    """
    (noise_model, noise_kwargs, channels) for build_sef, honouring NOISE_SOURCE.

    (None, None, None) leaves StableEMRIFisher on its own analytic scirdv1 TDI2 PSDs, which is also
    what happens when NOISE_SOURCE == "mojito" but the noise file is out of reach.
    """
    if NOISE_SOURCE != "mojito":
        return None, None, None
    try:
        psd, *_ = mojito_psd_model(channels=tdi_chan, verbose=verbose)
    except (FileNotFoundError, OSError, KeyError) as e:
        warnings.warn(
            f"Mojito noise unavailable ({type(e).__name__}: {e}) -> falling back to the analytic "
            "scirdv1 TDI2 PSDs. Expected off the GPU node.",
            stacklevel=2,
        )
        return None, None, None
    return psd, [{"channel": c} for c in tdi_chan], list(tdi_chan)


# ##### Offline check of the smoothing
# 
# Runs against a synthetic PSD with known truth, so it works without the cluster files. It injects
# the pathologies the repair is meant to survive: scattered dead bins, a contiguous dead block, NaNs,
# and an upward spike.

# In[15]:


def _test_smooth_psd(seed=0, blocks=(0, 5, 15, 31, 61), plot=True):
    """
    Check the repair against a synthetic PSD of known truth.

    Injects the pathologies it is meant to survive: scattered dead bins, a contiguous dead block of
    varying width, missing bins, and an upward spike.
    """
    from stableemrifisher.noise import sensitivity_LWA

    f = np.logspace(-5, np.log10(0.4), 1200)
    truth = sensitivity_LWA(f)
    results = {}

    for block in blocks:
        rng = np.random.default_rng(seed)
        meas = truth * rng.gamma(30.0, 1 / 30.0, f.size)          # chi2-like estimator scatter
        meas[rng.choice(f.size, 40, replace=False)] *= 1e-8       # scattered dead bins
        if block:
            meas[500:500 + block] *= 1e-8                         # contiguous dead block
        meas[900:903] = np.nan                                    # missing bins
        meas[1100] = truth[1100] * 1e4                            # upward spike

        res = smooth_psd(f, meas, **SMOOTH_KWARGS)
        sm = res["psd"]
        rel = np.abs(sm - truth) / truth
        with np.errstate(divide="ignore", invalid="ignore"):
            worst_raw = np.nanmax(truth / np.where(meas > 0, meas, np.nan))
        worst = float(np.max(truth / sm))
        results[block] = (res, sm, rel, worst)

        print(f"block={block:3d}  width->{res['width']:3d}  flagged {res['bad'].sum():3d}  "
              f"floored {res['floored'].sum():3d}  1/S overestimate {worst_raw:.1e} -> {worst:.2f}  "
              f"err median {np.median(rel) * 100:4.1f}%  p95 {np.percentile(rel, 95) * 100:5.1f}%")

        assert np.all(np.isfinite(sm) & (sm > 0)), f"block={block}: non-finite PSD"
        assert res["ok"], f"block={block}: repair reported an unconverged notch"
        assert worst < 2.0, f"block={block}: 1/S still overestimated by {worst:.1f}"
        assert np.median(rel) < 0.10, f"block={block}: median error {np.median(rel):.3f}"

    # a clean PSD must pass through essentially untouched
    clean = smooth_psd(f, truth.copy(), **SMOOTH_KWARGS)
    err = np.max(np.abs(clean["psd"] - truth) / truth)
    print(f"clean input: flagged {clean['bad'].sum()}, floored {clean['floored'].sum()}, "
          f"max error {err * 100:.2f}%")
    assert clean["bad"].sum() == 0 and clean["floored"].sum() == 0
    assert err < 0.02

    # and a dropout too wide to repair must be reported, not silently patched
    rng = np.random.default_rng(seed)
    hopeless = truth * rng.gamma(30.0, 1 / 30.0, f.size)
    hopeless[400:640] *= 1e-8
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = smooth_psd(f, hopeless, **SMOOTH_KWARGS)
    assert not res["ok"] and caught, "a 240-bin dead block should have been reported"
    print(f"240-bin dead block correctly reported: max step {res['max_step']:.1f} decades")

    if plot:
        block = max(blocks)
        res, sm, _, _ = results[block]
        rng = np.random.default_rng(seed)
        meas = truth * rng.gamma(30.0, 1 / 30.0, f.size)
        meas[rng.choice(f.size, 40, replace=False)] *= 1e-8
        if block:
            meas[500:500 + block] *= 1e-8
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.loglog(f, meas, lw=0.5, alpha=0.5, label=f"measured ({block}-bin dead block)")
        ax.loglog(f, truth, lw=1.2, label="truth")
        ax.loglog(f, sm, lw=1.2, ls="--", label="repaired")
        ax.set_xlabel("f [Hz]"); ax.set_ylabel("S(f) [1/Hz]")
        ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    return results


_test_smooth_psd();


# In[16]:


def plot_mojito_psd(channels=None, f_min=F_MIN, f_max=None):
    """Raw vs smoothed measured PSD, with the analytic scirdv1 curve for reference."""
    channels = channels or TDI_CHAN
    _, freqs, raw, smoothed = mojito_psd_model(channels=channels, verbose=False)
    f_max = f_max or 1.0 / (2 * DT)

    fig, axes = plt.subplots(len(channels), 1, figsize=(9, 3.2 * len(channels)),
                             sharex=True, squeeze=False)
    index = {"A": 0, "E": 1, "T": 2}
    for ax, c in zip(axes[:, 0], channels):
        i = index[c]
        ax.loglog(freqs, raw[i], lw=0.5, alpha=0.5, label=f"{c} measured")
        ax.loglog(freqs, smoothed[i], lw=1.3, label=f"{c} smoothed")
        ax.axvspan(f_min, f_max, color="k", alpha=0.05, label="analysis band")
        ax.set_ylabel(f"$S_{c}(f)$  [1/Hz]")
        ax.grid(alpha=0.3, which="both")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1, 0].set_xlabel("f [Hz]")
    fig.tight_layout()
    return fig


# In[17]:


plot_mojito_psd()


# In[18]:


class PaddedEMRIWaveform(GenerateEMRIWaveform):
    """
    GenerateEMRIWaveform that zero-pads its output up to `min_output_length`.

    fastlisaresponse sizes its buffers with its own value of the sidereal year, which differs from
    FEW's in the last two digits, so `int(T * YRSID / dt)` can come out one sample short of what
    pyResponseTDI expects.  Same fix as `EMRIWave.min_output_length` in
    `src/waveform_updated.py`, but as a GenerateEMRIWaveform subclass so that StableEMRIFisher can
    construct it itself.  `attach_padding` sets the length once the response exists.
    """

    min_output_length: int = 0

    def __call__(self, *args, **kwargs):
        h = super().__call__(*args, **kwargs)
        if self.min_output_length > len(h):
            _xp = cp.get_array_module(h) if cp is not None else np
            h = _xp.concatenate(
                [h, _xp.zeros(self.min_output_length - len(h), dtype=h.dtype)]
            )
        return h


def attach_padding(sef):
    """Point the waveform generator(s) inside `sef` at the response's expected length."""
    n_pts = sef.waveform_generator.response_model.num_pts
    sef.waveform_generator.waveform_gen.min_output_length = n_pts
    if hasattr(sef.derivative, "waveform_gen"):        # deriv_type="stable" wraps a second one
        gen = sef.derivative.waveform_gen
        if isinstance(gen, PaddedEMRIWaveform):
            gen.min_output_length = n_pts
    return n_pts


def waveform_generator(case, T=T_OBS, dt=DT):
    """Bare FEW generator for the 1PAT1R model with this case's toggles."""
    return GenerateEMRIWaveform(
        Circ1PAT1R,
        return_list=False,
        frame="detector",
        inspiral_kwargs={**INSPIRAL_KWARGS, "evolve_primary": EVOLVE_CHI1},
        amplitude_kwargs={**AMPLITUDE_KWARGS, "zero_PA_amps_only": not INCLUDE_1PA_AMPS},
        sum_kwargs=dict(SUMMATION_KWARGS),
    )


def plunge_trimmed_T(case, T=T_OBS, dt=DT, trim_hours=6.0, _gen_cache={}):
    """
    Observation time in years, shortened if the secondary plunges inside T.

    StableEMRIFisher has its own `plunge_check`, but it only rewrites the waveform kwargs --
    `ResponseWrapper.__call__` overwrites `kwargs["T"]` with its own `Tobs`, so with a response
    attached the trimming is silently discarded.  Doing it here instead means the ResponseWrapper
    is *built* with the trimmed duration and the trimming actually takes effect.
    """
    key = case["config"]
    gen = _gen_cache.get(key) or _gen_cache.setdefault(key, waveform_generator(case, T, dt))
    traj = gen.waveform_generator.inspiral_generator
    p = case["params"]
    t_traj = traj(
        p["m1"], p["m2"], p["a"], p["p0"], p["e0"], p["xI0"], case["chi2"],
        Phi_phi0=p["Phi_phi0"], Phi_theta0=p["Phi_theta0"], Phi_r0=p["Phi_r0"],
        T=T, dt=dt,
    )[0]
    if t_traj[-1] < T * YRSID_SI - 1.0:
        t_end = t_traj[-1] - trim_hours * 3600.0
        print(f"  plunges at {t_traj[-1] / YRSID_SI:.4f} yr -> "
              f"using T = {t_end / YRSID_SI:.4f} yr (last {trim_hours:g} h dropped)")
        return t_end / YRSID_SI
    print(f"  no plunge within {T:g} yr")
    return T


def response_kwargs(T, dt=DT, tdi_chan=None, orbits=None, verbose=True):
    """
    ResponseWrapper kwargs matching src/waveform_updated.py::build_response.

    Note the fastlisaresponse 1.1.17 spellings: `force_backend` rather than `use_gpu`, and `orbits`
    as an instance (the signature default is the *class*, which fails its own isinstance assert).
    """
    T_response, t_init = response_timing(T, dt, verbose=verbose)
    return dict(
        Tobs=T_response,
        dt=dt,
        index_lambda=8,        # phiS in the 1PA vector (chi2 is appended after Phi_r0)
        index_beta=7,          # qS
        t0=t_init,
        flip_hx=RESPONSE["flip_hx"],
        is_ecliptic_latitude=RESPONSE["is_ecliptic_latitude"],
        remove_sky_coords=RESPONSE["remove_sky_coords"],
        remove_garbage=RESPONSE["remove_garbage"],
        force_backend=FORCE_BACKEND,
        orbits=orbits if orbits is not None else build_orbits(verbose=verbose),
        order=RESPONSE["order"],
        tdi=RESPONSE["tdi_gen"],
        tdi_chan=tdi_chan or TDI_CHAN,
    )


def build_sef(case, T=None, dt=DT, tdi_chan=None, orbits=None,
              deriv_type="direct", der_order=4, Ndelta=8, filename=None,
              noise_model=None, noise_kwargs=None, channels=None, verbose=True):
    """StableEMRIFisher configured for the 1PAT1R waveform plus the PE-run LISA response."""
    if T is None:
        T = plunge_trimmed_T(case, dt=dt)

    tdi_chan = tdi_chan or TDI_CHAN
    if tdi_chan not in ("AE", "AET") and noise_model is None:
        raise ValueError(
            f"tdi_chan={tdi_chan!r} has no diagonal PSD description. StableEMRIFisher weights each "
            "channel independently, which is only valid for (approximately) noise-orthogonal "
            "channels. Use 'AE'/'AET', or pass an explicit noise_model."
        )

    if noise_model is None:
        noise_model, noise_kwargs, channels = noise_for(tdi_chan, verbose=verbose)

    rw_kwargs = response_kwargs(T, dt, tdi_chan, orbits, verbose=verbose)

    sef = StableEMRIFisher(
        waveform_class=Circ1PAT1R,
        waveform_class_kwargs=dict(
            inspiral_kwargs={**INSPIRAL_KWARGS, "evolve_primary": EVOLVE_CHI1},
            amplitude_kwargs={**AMPLITUDE_KWARGS, "zero_PA_amps_only": not INCLUDE_1PA_AMPS},
            sum_kwargs=dict(SUMMATION_KWARGS),
        ),
        waveform_generator=PaddedEMRIWaveform,
        waveform_generator_kwargs={"return_list": False, "frame": "detector"},
        ResponseWrapper=ResponseWrapper,
        ResponseWrapper_kwargs=rw_kwargs,
        noise_model=noise_model,
        noise_kwargs=noise_kwargs,
        channels=channels,
        use_gpu=USE_GPU,
        deriv_type=deriv_type,
        der_order=der_order,
        Ndelta=Ndelta,
        T=T,
        dt=dt,
        plunge_check=False,     # already handled by plunge_trimmed_T
        filename=filename,
    )
    n_pts = attach_padding(sef)
    if verbose:
        src = NOISE_SOURCE if noise_model is not None else "analytic (scirdv1 TDI2)"
        print(f"  T = {T:.4f} yr, T_response = {rw_kwargs['Tobs']:.4f} yr, "
              f"t0 = {rw_kwargs['t0']:g} s, {n_pts} samples, channels {tdi_chan}, noise {src}")
    return sef, T


# #### Window and frequency mask
# 
# `PE_response_updated.py` windows with a Tukey of `alpha = 0.01` that spans only the non-zero part
# of the response output, so that the zero padding past the plunge is left flat, then restricts the
# inner product to `f > 1e-5 Hz`. `StableEMRIFisher.__call__` takes `window` and `fmin`/`fmax`
# directly, so the same treatment carries over.
# 
# Building the window costs one response evaluation, which is also a useful check that the response
# is producing something sane before the derivatives start.

# In[19]:


from scipy.signal.windows import tukey


def build_window(sef, case, T, dt=DT, alpha=TUKEY_ALPHA, windowing=WINDOWING, verbose=True):
    """
    Tukey window over the non-zero extent of the response output, as in PE_response_updated.py.

    Returns (window, n_samples, peak_amplitude); `window` is None when windowing is off.
    """
    h = xp.asarray(sef.waveform_generator(
        *(list(case["params"].values()) + [case["chi2"]]), T=T, dt=dt
    ))
    n_t = h.shape[1]
    nonzero = xp.where(xp.any(h != 0.0, axis=0))[0]
    n_inj = int(nonzero[-1]) + 1 if nonzero.size > 0 else n_t
    peak = float(xp.abs(h).max())
    if verbose:
        print(f"  signal occupies {n_inj}/{n_t} samples ({100 * n_inj / n_t:.1f}%), "
              f"peak |TDI| = {peak:.3e}")

    if not windowing:
        return None, n_t, peak

    window = xp.zeros(n_t)
    window[:n_inj] = xp.asarray(tukey(n_inj, alpha=alpha))
    return window, n_t, peak


# #### Finite-difference step ranges
# 
# `Fisher_Stability` falls back to `geomspace(1e-4 * value, 1e-9 * value)` for the intrinsic
# parameters, which misbehaves for the two parameters that sit near a boundary here:
# 
# * `a` is `8.2e-5` (case 1) or exactly `0.0` (cases 2, 3), so a value-scaled grid is either far too
#   small or undefined. A fixed absolute grid is used instead.
# * `chi2` would get steps up to `0.1 * chi2`, pushing `chi2 = 0.998` (case 3) above 1. The grid is
#   capped at `1e-2` and `chi2` is registered in `sef.minmax` so that values near the edge switch to
#   one-sided differences automatically.

# In[20]:


def delta_ranges(Ndelta=8):
    return dict(
        a=np.geomspace(1e-4, 1e-9, Ndelta),        # absolute: a is ~0 in all three cases
        chi2=np.geomspace(1e-2, 1e-7, Ndelta),     # absolute: keeps chi2 = 0.998 below 1
    )


# Bounds used by Fisher_Stability to pick central/forward/backward differences.
# a is already there ([0.05, 0.95]); chi2 lives on [-1, 1] and needs the same treatment.
CHI2_MINMAX = [-0.95, 0.95]


# ### SNR check
# 
# Cheap sanity pass before committing to the derivatives: build the response once per case and read
# off the optimal SNR. Compare against the SNRs quoted for the PE runs.
# 
# Reference values measured locally, i.e. with the *fallback* orbits and `t0` (`AE`, TDI2, bundled
# `ESAOrbits`, `scirdv1` without confusion foreground, no window). On the GPU node, with the config
# orbit file and the L1-derived epoch, expect these to shift somewhat:
# 
# | case | plunges at | SNR |
# |---|---|---|
# | 1 | 1.519 yr | 975 |
# | 2 | 1.499 yr | 199 |
# | 3 | 1.435 yr | 160 |
# 
# All three plunge well inside the configured `T = 2 yr`, which is why `plunge_trimmed_T` matters here.

# In[21]:


def snr_only(case, T=None, dt=DT, tdi_chan=None):
    sef, T_used = build_sef(case, T=T, dt=dt, tdi_chan=tdi_chan)
    window, _, _ = build_window(sef, case, T_used, dt=dt)
    rho = sef.SNRcalc_SEF(
        *(list(case["params"].values()) + [case["chi2"]]),
        window=window,
        fmin=F_MIN if FILTER_FREQ else None,
        use_gpu=USE_GPU,
        dt=dt,
        T=T_used,
    )
    return rho, T_used


snrs = {}
for i, c in REVIEW_CASES.items():
    print(f"--- case {i} ---")
    rho, T_used = snr_only(c)
    snrs[i] = rho
    print(f"  SNR (AE, TDI2, T = {T_used:.4f} yr) = {rho:.1f}\n")


# ### Fisher matrices
# 
# Measured on this machine (CPU, no CuPy): a bare `T = 2 yr`, `dt = 5 s` `Circ1PAT1R` waveform takes
# about 1 s, but one **response** evaluation takes about **20 s**, and that is what dominates.
# 
# Per case, the number of response calls is roughly `n_params * Ndelta * der_order` for the stable-delta
# search plus `n_params * der_order` for the matrix itself:
# 
# | settings | calls (11 params) | wall time |
# |---|---|---|
# | `der_order=2, Ndelta=3` | ~90 | ~30 min |
# | `der_order=2, Ndelta=4` | ~110 | ~40 min |
# | `der_order=4, Ndelta=8` | ~400 | ~2 h |
# 
# `live_dangerously=True` skips the stability search entirely (~`n_params * der_order` calls, a couple
# of minutes) and falls back to a mass-ratio/SNR heuristic for the step sizes. Good for a first look,
# not for numbers you would quote.
# 
# On the GPU node the response is what speeds up, so `der_order=4, Ndelta=8` should be the default
# there rather than something to work up to.

# In[ ]:


def run_case(case_id, der_order=4, Ndelta=8, tdi_chan=None, T=None,
             deriv_type="direct", live_dangerously=False, save=True):
    case = REVIEW_CASES[case_id]
    names = fisher_params(case)
    tag = f"review_test_{case_id}"

    print(f"=== case {case_id} ===")
    sef, T_used = build_sef(
        case, T=T, tdi_chan=tdi_chan, deriv_type=deriv_type,
        der_order=der_order, Ndelta=Ndelta,
        filename=str(OUT_DIR / tag) if save else None,
    )
    sef.minmax["chi2"] = CHI2_MINMAX
    window, _, _ = build_window(sef, case, T_used)

    fisher = sef(
        dict(case["params"]),                 # copied: sef mutates it to append chi2
        add_param_args={"chi2": case["chi2"]},
        param_names=names,
        delta_range=delta_ranges(Ndelta),
        window=window,
        fmin=F_MIN if FILTER_FREQ else None,
        live_dangerously=live_dangerously,
        plunge_check=False,
    )

    cov = np.linalg.inv(fisher)
    sigma = np.sqrt(np.diag(cov))
    result = {
        "fisher": fisher, "cov": cov, "sigma": dict(zip(names, sigma)),
        "names": names, "snr": float(np.sqrt(sef.SNR2)), "T": T_used,
        "deltas": dict(sef.deltas) if sef.deltas else None,
    }
    if save:
        np.savez(OUT_DIR / f"{tag}_fisher.npz", fisher=fisher, cov=cov,
                 names=np.array(names), snr=result["snr"], T=T_used)
    return result


def report(result, case_id):
    case = REVIEW_CASES[case_id]
    truths = {**case["params"], "chi2": case["chi2"]}
    print(f"\ncase {case_id}:  SNR = {result['snr']:.1f},  T = {result['T']:.4f} yr")
    print(f"{'param':>10} {'truth':>18} {'sigma':>14} {'sigma/truth':>14}")
    for n in result["names"]:
        s, t = result["sigma"][n], truths[n]
        rel = f"{s / abs(t):.3e}" if t != 0 else "--"
        print(f"{n:>10} {t:18.8g} {s:14.6e} {rel:>14}")


# In[ ]:


# Quick pass over all three cases (~40 min each).  For production numbers use
#     results[case_id] = run_case(case_id, der_order=4, Ndelta=8)
# and for a first look in a couple of minutes
#     results[case_id] = run_case(case_id, der_order=2, live_dangerously=True)
results = {}
for case_id in (1, 2, 3):
    results[case_id] = run_case(case_id, der_order=2, Ndelta=4)
    report(results[case_id], case_id)


# ### Summary
# 
# Compare `sigma` here against the marginal posterior widths from the matching PE runs in
# `validation/PE_test_runs/sampling_data/`.

# In[ ]:


def summary_table(results):
    all_names = []
    for r in results.values():
        for n in r["names"]:
            if n not in all_names:
                all_names.append(n)
    header = f"{'param':>10} " + " ".join(f"{'case ' + str(i):>14}" for i in results)
    print(header)
    print("-" * len(header))
    for n in all_names:
        row = f"{n:>10} "
        for r in results.values():
            row += f"{r['sigma'][n]:14.4e} " if n in r["sigma"] else f"{'--':>14} "
        print(row)
    print("-" * len(header))
    print(f"{'SNR':>10} " + " ".join(f"{r['snr']:14.1f}" for r in results.values()))


summary_table(results)


# In[ ]:


def plot_sigmas(results):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    all_names = []
    for r in results.values():
        for n in r["names"]:
            if n not in all_names:
                all_names.append(n)
    x = np.arange(len(all_names))
    width = 0.8 / len(results)
    for k, (case_id, r) in enumerate(results.items()):
        truths = {**REVIEW_CASES[case_id]["params"], "chi2": REVIEW_CASES[case_id]["chi2"]}
        vals = [
            r["sigma"][n] / abs(truths[n]) if n in r["sigma"] and truths[n] != 0 else np.nan
            for n in all_names
        ]
        ax.bar(x + k * width, vals, width, label=f"case {case_id} (SNR {r['snr']:.0f})")
    ax.set_xticks(x + 0.4 - width / 2)
    ax.set_xticklabels(all_names, rotation=45, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel(r"$\sigma_\theta / |\theta|$")
    ax.set_title("Fisher fractional errors, review test cases (1PAT1R, AE, TDI2)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


plot_sigmas(results);

