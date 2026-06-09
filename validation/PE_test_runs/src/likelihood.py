'''
Log-likelihood for TDI data with a recovery response callable.
'''

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import cupy as cp
import numpy as np

from .utils import inner_prod_tdi

logger = logging.getLogger(__name__)


@dataclass
class RecoveryConfig:
    """Bundles everything the LogLikelihood needs beyond the data itself."""
    param_names: List[str]
    fixed_params: Dict[int, float] = field(default_factory=dict)
    x_I0_index: Optional[int] = None  # set to None if not in param_names
    x_I0_value: float = 1.0


class LogLikelihood:
    """
    Callable returning -0.5 <d-h|d-h> for sampled parameter vectors.

    The recovery response is the *same kind* of callable as the injection's
    (cupy TDI array); it may use a different WaveformConfig so that the
    recovery template differs from the injection.
    """

    def __init__(
        self,
        data_fft: cp.ndarray,
        inv_cov: cp.ndarray,
        recovery_response: Callable[..., cp.ndarray],
        cfg: RecoveryConfig,
        window: cp.ndarray,
        mask: cp.ndarray,
    ):
        self.data_fft = data_fft
        self.inv_cov = inv_cov
        self.response = recovery_response
        self.cfg = cfg
        self.window = window
        self.mask = mask

    def _expand(self, params_vec) -> List[float]:
        """Reconstruct the full FEW parameter list from a sampled vector."""
        full: List[float] = []
        k = 0
        for i in range(len(self.cfg.param_names)):
            if i == self.cfg.x_I0_index:
                full.append(self.cfg.x_I0_value)
            elif i in self.cfg.fixed_params:
                full.append(self.cfg.fixed_params[i])
            else:
                full.append(float(params_vec[k]))
                k += 1
        return full

    def __call__(self, params_vec) -> float:
        full = self._expand(params_vec)
        try:
            waveform_td = self.response(*full)
            model_fft = cp.fft.rfft(waveform_td * self.window, axis=1)[:, self.mask]
            diff = model_fft - self.data_fft
            ll = -0.5 * inner_prod_tdi(diff, diff, self.inv_cov)
            return float(cp.asnumpy(ll))
        except Exception as exc:
            logger.warning(f"Likelihood failed: {exc}")
            # log properties of point that failed the likelihood evaluation for debugging
            for name, val in zip(self.cfg.param_names, full):
                logger.warning(f"  {name} = {val:.6e}")
            # return large negative loglikelihood to build barrier around failed point for sampler
            return np.float32(-1e10)
