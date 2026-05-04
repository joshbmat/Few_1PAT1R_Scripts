'''
Prior construction for the PE validation runs.

Returns an eryn ProbDistContainer and a bounds dict keyed by the sampled
parameter index. Non-sampled parameters (fixed_params, plus x_I0 which is
always fixed to 1.0) are skipped.
'''

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from eryn.prior import ProbDistContainer, uniform_dist


# Widths per intrinsic parameter, keyed by name. Everything not listed
# falls back to its physical parameter space (sky angles, phases, d_L).
_WIDTHS_INTRINSIC = {
    "M": 100.0,
    "mu": 1e-3,
    "a": 1e-4,
    "p0": 1e-4,
    "e0": 1e-4,
    "chi2": 1e-3,
}
_WIDTH_DL = 0.5
_EPS = 1e-12


def _clamp(lo: float, hi: float, lo_bound: float, hi_bound: float,
           min_width: float) -> Tuple[float, float]:
    lo = max(lo, lo_bound)
    hi = min(hi, hi_bound)
    if hi <= lo:
        hi = min(lo + max(min_width, _EPS), hi_bound)
        lo = max(lo_bound, hi - max(min_width, _EPS))
    return lo, hi


def build_priors(
    param_names: List[str],
    emri_params: List[float],
    fixed_params: Iterable[str],
    n: float,
    use_cupy: bool = True,
    width_overrides: Optional[Dict[str, float]] = None,
) -> Tuple[ProbDistContainer, Dict[int, Tuple[float, float]], List[int]]:
    """
    Build priors for the sampled subset of parameters.

    Parameters
    ----------
    param_names     : full list of parameter names (model-dependent)
    emri_params     : true parameter values aligned with param_names
    fixed_params    : names of parameters held fixed (besides x_I0)
    n               : width scale factor (d from config)
    width_overrides : optional {param_name: width} to override _WIDTHS_INTRINSIC
                      on a per-parameter basis.  Useful when the posterior is
                      expected to be biased away from the injection truth (e.g.
                      cross-model 1PA→0PA runs): increase the width so the prior
                      does not rail the chain.  Sky angles and phases always use
                      their full physical range regardless of this dict.

    Returns
    -------
    priors       : ProbDistContainer indexed by sampled index k
    bounds       : {k: (lo, hi)} for clipping starting positions
    sampled_idx  : indices into param_names that are actually sampled

    Notes
    -----
    **Posterior railing at prior boundary** is almost always caused by a prior
    that is centred on the injection truth but narrower than the systematic bias
    introduced by model mismatch.  The quickest fix is to increase ``n`` (the
    ``d`` config key) or supply ``width_overrides`` for the affected parameters.
    The default widths in ``_WIDTHS_INTRINSIC`` were chosen for 1PA self-
    consistent runs; they can be too narrow for 0PA recovery on a 1PA dataset.
    """
    fixed_set = set(fixed_params)
    widths = {**_WIDTHS_INTRINSIC, **(width_overrides or {})}
    priors_in: Dict[int, object] = {}
    bounds: Dict[int, Tuple[float, float]] = {}
    sampled_idx: List[int] = []
    k = 0
    for i, name in enumerate(param_names):
        if name in fixed_set or name == "x_I0":
            continue
        true = emri_params[i]
        if name in ("M", "mu"):
            w = widths[name]
            lo, hi = max(true - n * w, 2.0), true + n * w
        elif name == "a":
            w = widths[name]
            # max_valid_spin = 5*mass ratio for primary spin if using 1PAT1R model, 0.999 otherwise. need to do this to avoid weird effects
            if true >= 0:
                lo, hi = _clamp(true - n * w, true + n * w, -0.999, 0.999, w)
            else:
                lo, hi = _clamp(true - n * w, true + n * w, -0.999, 0.0, w)
        elif name == "chi2":
            w = widths[name]
            lo, hi = _clamp(true - n * w, true + n * w, -0.999, 0.999, w)
        elif name == "e0":
            w = widths[name]
            hi_cand = min(true + 2 * n * w, 0.9)
            lo, hi = _clamp(true - n * w, hi_cand, 0.0, 0.9, w)
        elif name == "p0":
            w = widths[name]
            lo, hi = true - n * w, true + n * w
        elif name == "d_L":
            lo, hi = max(true - n * _WIDTH_DL, 1e-3), true + n * _WIDTH_DL
        elif name in ("theta_S", "theta_K"):
            lo, hi = 0.0, np.pi
        else:  # phi_S, phi_K, Phi_phi0, Phi_theta0, Phi_r0
            lo, hi = 0.0, 2 * np.pi
        priors_in[k] = uniform_dist(lo, hi)
        bounds[k] = (lo, hi)
        sampled_idx.append(i)
        k += 1
    return ProbDistContainer(priors_in, use_cupy=use_cupy), bounds, sampled_idx
