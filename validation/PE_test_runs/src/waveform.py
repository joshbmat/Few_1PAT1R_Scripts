'''
Waveform + response construction for PE validation studies.

Supports two models:
  - "1PAT1R" : few.waveform.Waveform1PAT1R, with toggles for primary spin
               evolution (evolve_primary) and 1PA amplitude corrections
               (zero_PA_amps_only). Parameter vector includes chi2 at index 5.
  - "0PA_Kerr" : few.waveform.FastKerrEccentricEquatorialFlux wrapped by
                 GenerateEMRIWaveform. No chi2 in the parameter vector.
'''

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import cupy as cp

from fastlisaresponse import ResponseWrapper
from fastlisaresponse.tdiconfig import TDIConfig
from fastlisaresponse.utils.parallelbase import ParallelModuleBase
from lisatools.detector import Orbits


PARAM_NAMES_1PA = [
    'M', 'mu', 'a', 'p0', 'e0', 'chi2', 'x_I0', 'd_L',
    'theta_S', 'phi_S', 'theta_K', 'phi_K',
    'Phi_phi0', 'Phi_theta0', 'Phi_r0',
]

PARAM_NAMES_0PA = [
    'M', 'mu', 'a', 'p0', 'e0', 'x_I0', 'd_L',
    'theta_S', 'phi_S', 'theta_K', 'phi_K',
    'Phi_phi0', 'Phi_theta0', 'Phi_r0',
]


def param_names_for(model: str) -> list[str]:
    return PARAM_NAMES_1PA if model == "1PAT1R" else PARAM_NAMES_0PA


def sky_indices_for(model: str) -> tuple[int, int]:
    """(index_beta, index_lambda) for fastlisaresponse.ResponseWrapper."""
    if model == "1PAT1R":
        return 8, 9   # theta_S, phi_S with chi2 inserted at index 5
    return 7, 8       # standard FEW ordering without chi2


@dataclass
class WaveformConfig:
    """Settings for a single waveform evaluation (injection or recovery)."""
    model: Literal["1PAT1R", "0PA_Kerr"] = "1PAT1R"
    dt: float = 5.0
    T: float = 2.0
    mode_selection_threshold: float = 0.0
    # 1PAT1R toggles
    evolve_chi1: bool = True
    include_1PA_amps: bool = True
    # General FEW kwargs
    inspiral_kwargs: Dict[str, Any] = field(default_factory=dict)
    amplitude_kwargs: Dict[str, Any] = field(default_factory=dict)
    summation_kwargs: Dict[str, Any] = field(default_factory=dict)
    # Restrict mode content to l <= lmax.  None means no restriction.
    # Set to 5 when recovering a 1PA injection with a 0PA model so both
    # templates use the same harmonic content (Waveform1PAT1R only has
    # amplitudes up to l=5).
    lmax: Optional[int] = None

    # define param names based on model (handles chi2 as extra param)
    def param_names(self) -> list[str]:
        return param_names_for(self.model)

    # define sky indices based on model
    def sky_indices(self) -> tuple[int, int]:
        return sky_indices_for(self.model)


@dataclass
class ResponseConfig:
    # settings for ResponseWrapper construction
    orbit_file: str
    tdi_gen: str = "2nd generation"
    tdi_chan: str = "XYZ"
    order: int = 40
    offset: float = 550.0
    n_samples_delay: int = 1000
    t_buffer: float = 10000.0
    flip_hx: bool = True
    is_ecliptic_latitude: bool = False
    remove_sky_coords: bool = False
    remove_garbage: bool = False


def _mode_selection_from_lmax(lmax: int) -> List[Tuple]:
    """
    Return all (l, m, n=0, k=0) modes with l <= lmax for equatorial circular
    orbits.  Positive m only; FEW derives negative-m counterparts internally.
    """
    return [(l, m, 0, 0) for l in range(2, lmax + 1) for m in range(1, l + 1)]


class EMRIWave(ParallelModuleBase):
    """
    Unified waveform callable that dispatches to the right FEW backend
    based on `WaveformConfig.model`. Returns h_+ - i h_x for ResponseWrapper.
    """

    def __init__(self,
                 cfg: WaveformConfig,
                 T_waveform: Optional[float] = None,
                 force_backend: Optional[str] = None):
        self.cfg = cfg
        # T_waveform must cover cfg.T plus all TDI/response buffer padding.
        self.T_waveform = T_waveform if T_waveform is not None else cfg.T
        # Set by build_response after ResponseWrapper is constructed so we can
        # zero-pad plunging waveforms to the exact length fastlisaresponse expects.
        self.min_output_length: int = 0
        # Should return h+ - ihx in detector frame, with 1PA effects toggled on/off
        if cfg.model == "1PAT1R":
            from few.waveform import GenerateEMRIWaveform
            self._gen = GenerateEMRIWaveform(
                'Waveform1PAT1R',
                return_list=False,
                inspiral_kwargs=cfg.inspiral_kwargs,
                amplitude_kwargs=cfg.amplitude_kwargs,
                frame='detector',
            )
            self._call = self._call_1pa
        elif cfg.model == "0PA_Kerr":
            from few.waveform import GenerateEMRIWaveform
            mode_selector_kwargs = {}
            if cfg.lmax is not None:
                mode_selector_kwargs["mode_selection"] = _mode_selection_from_lmax(cfg.lmax)
            self._gen = GenerateEMRIWaveform(
                "FastKerrEccentricEquatorialFlux",
                return_list=False,
                inspiral_kwargs=cfg.inspiral_kwargs,
                amplitude_kwargs=cfg.amplitude_kwargs,
                mode_selector_kwargs=mode_selector_kwargs or None,
                frame="detector",
            )
            self._call = self._call_0pa
        else:
            raise ValueError(f"Unknown waveform model: {cfg.model}")

    @classmethod
    def supported_backends(cls):
        return ["fastlisaresponse_" + b for b in cls.GPU_RECOMMENDED()]

    def _call_1pa(self, *params, **waveform_kwargs):
        # chi2 lives at index 5 in PARAM_NAMES_1PA; FEW expects it as a keyword.
        # Strip it from the positional vector and forward the 0PA-style params.
        chi2 = params[5]
        params_0pa = params[:5] + params[6:]
        return self._gen(
            *params_0pa,
            chi2=chi2,
            dt=self.cfg.dt, T=self.T_waveform,
            zero_PA_amps_only=(not self.cfg.include_1PA_amps),
            evolve_primary=self.cfg.evolve_chi1,
            pad_output=True,
        )

    def _call_0pa(self, *params, **waveform_kwargs):
        return self._gen(
            *params,
            T=self.T_waveform, dt=self.cfg.dt,
            mode_selection_threshold=self.cfg.mode_selection_threshold,
            pad_output=True,
        )

    def __call__(self, *params, **waveform_kwargs):
        h = self._call(*params, **waveform_kwargs)
        if self.min_output_length > 0 and len(h) < self.min_output_length:
            xp = cp.get_array_module(h)
            h = xp.concatenate([h, xp.zeros(self.min_output_length - len(h), dtype=h.dtype)])
        return h


def build_response(
    cfg: WaveformConfig,
    resp_cfg: ResponseConfig,
    t_init: float,
    t0_orbits: float,
    T_response: float,
    use_gpu: bool = True,
):
    """Sets up a responseWarpper callable wrapped around waveform generator and orbits.
    This returns a callable that returns 3xNt array of TDI data for given set of EMRI params."""
    force_backend = "cuda12x" if use_gpu else None

    wave = EMRIWave(cfg, T_waveform=T_response, force_backend=force_backend)
    orbits = Orbits(
        filename=resp_cfg.orbit_file,
        use_gpu=use_gpu,
        force_backend=force_backend,
        linear_interp_setup=False,
        t0=t0_orbits,
    )

    index_beta, index_lambda = cfg.sky_indices()
    tdi_kwargs = dict(
        orbits=orbits,
        order=resp_cfg.order,
        tdi=TDIConfig(resp_cfg.tdi_gen),
        tdi_chan=resp_cfg.tdi_chan,
    )

    response = ResponseWrapper(
        wave,
        T_response,
        cfg.dt,
        index_lambda,
        index_beta,
        t0=t_init,
        t_buffer=resp_cfg.t_buffer,
        flip_hx=resp_cfg.flip_hx,
        force_backend=force_backend,
        remove_sky_coords=resp_cfg.remove_sky_coords,
        is_ecliptic_latitude=resp_cfg.is_ecliptic_latitude,
        remove_garbage=resp_cfg.remove_garbage,
        **tdi_kwargs,
    )

    # Tell the wave generator to zero-pad up to the exact length that
    # fastlisaresponse expects (pyResponseTDI.num_pts). Without this,
    # plunging systems whose inspiral ends before T_response would produce
    # a waveform that is too short, triggering the assert inside
    # get_projections even when pad_output=True is set on the FEW generator.
    wave.min_output_length = response.response_model.num_pts

    def _call(*params):
        return cp.asarray(response(*params))

    return _call
