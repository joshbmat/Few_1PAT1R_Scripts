"""
Post-processing utilities for EMRI PE runs with Eryn.

Implemented functions:
----------
========= data processing functions =========
sampled_params_from_config  derive sampled param names and true values from config dict
load_samples                load Eryn HDF backend; return per-temperature sample arrays
load_samples_named          load Eryn HDF backend; return {param_name: (N_temps, N_iters, N_walkers)}
plot_log_like               log-likelihood trace plot for one temperature
cut_samples_autocorr        thin samples by integrated auto-correlation time
    This only makes sense for the cold chain
filter_lost_walkers         remove stuck/lost walkers from the cold chain
    Again, only makes sense for cold chain

========= Plotting functions =========
corner_plot                 multi-posterior corner plot driven by config
    Flexible function which plots several posteriors on the same figure, allowing comparison.
    entry['samples'] may be an ndarray (N, P) or a named dict {param: (N_temps, N_iters, N_walkers)}.
plot_sky_position           Mollweide sky map with 99 % HDR inset

========= Statistical functions =========
plot_chain_convergence      per-parameter trace plots
plot_covariance_evolution   running marginal-variance curves
plot_gelman_rubin           running Gelman-Rubin R-hat convergence diagnostics
plot_seaborn_diagnostics    seaborn pair-plot + violin plots
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import corner

from eryn.backends import HDFBackend as ErynHDFBackend

from src.waveform import param_names_for


# Parameter metadata

PARAM_LABELS: Dict[str, str] = {
    'M':          r'$M\,/\,M_{\odot}$',
    'mu':         r'$\mu\,/\,M_{\odot}$',
    'a':          r'$a$',
    'p0':         r'$p_0$',
    'e0':         r'$e_0$',
    'chi2':       r'$\chi_2$',
    'd_L':        r'$D_L\,/\,\mathrm{Gpc}$',
    'theta_S':    r'$\theta_S$',
    'phi_S':      r'$\phi_S$',
    'theta_K':    r'$\theta_K$',
    'phi_K':      r'$\phi_K$',
    'Phi_phi0':   r'$\Phi_{\phi_0}$',
    'Phi_theta0': r'$\Phi_{\theta_0}$',
    'Phi_r0':     r'$\Phi_{r_0}$',
}

SKY_PARAMS = frozenset({'theta_S', 'phi_S', 'theta_K', 'phi_K'})


# Config helpers 

def _emri_dict(config: dict) -> dict:
    """Return the EMRI parameter sub-dict, supporting both old and new layouts."""
    if 'Injection' in config and 'EMRI' in config.get('Injection', {}):
        return config['Injection']['EMRI']
    return config.get('Waveform', {})


def _inj_waveform(config: dict) -> dict:
    if 'Injection' in config and 'Waveform' in config.get('Injection', {}):
        return config['Injection']['Waveform']
    return config.get('Waveform', {})


def _rec_waveform(config: dict) -> dict:
    if 'Recovery' in config and 'Waveform' in config.get('Recovery', {}):
        return config['Recovery']['Waveform']
    return {}


def sampled_params_from_config(
    config: dict,
) -> Tuple[List[str], np.ndarray]:
    """
    Derive (sampled_param_names, true_values) from a config dict.

    Uses the *recovery* model to determine the parameter vector, then removes
    x_I0 and any params listed under Sampler.fixed_params.

    Returns
    -------
    sampled_names : list of str
    true_vals     : float array aligned with sampled_names (NaN if absent)
    """
    rec_wf  = _rec_waveform(config)
    model   = rec_wf.get('model', '0PA_Kerr')
    all_names = param_names_for(model)

    emri = _emri_dict(config)
    fixed = set(config.get('Sampler', {}).get('fixed_params', []))
    fixed.add('x_I0')

    sampled_names = [n for n in all_names if n not in fixed]
    z = float(emri.get('z', 0.0))
    true_vals = []
    for n in sampled_names:
        if n in emri and emri[n] is not None:
            val = float(emri[n])
            if n in ('M', 'mu'):
                val *= (1.0 + z)
            true_vals.append(val)
        else:
            true_vals.append(float('nan'))
    return sampled_names, np.array(true_vals)


# sample loading

def load_samples(
    file_path: str,
    discard: int = 0,
) -> Tuple[List[np.ndarray], ErynHDFBackend, np.ndarray, tuple]:
    """
    Load samples from an Eryn HDF backend.

    Parameters
    ----------
    file_path : path to the .h5 file
    discard   : number of burn-in iterations to discard

    Returns
    -------
    samples_per_temp : list[ndarray], one per temperature,
                       each shaped (N_walkers * N_iters, N_params)
    reader           : open ErynHDFBackend
    log_like         : ndarray (N_iters, N_temps, N_walkers)
    shape            : (N_iters, N_temps, N_walkers, N_params)
    """
    reader  = ErynHDFBackend(file_path, read_only=True)
    chain   = reader.get_chain(discard=discard)['model_0']
    N_iters, N_temps, N_walkers = chain.shape[:3]
    N_params = chain.shape[-1]

    samples_per_temp = [
        chain[:, t].reshape(-1, N_params) for t in range(N_temps)
    ]
    log_like = reader.get_log_like(discard=discard)
    return samples_per_temp, reader, log_like, (N_iters, N_temps, N_walkers, N_params)


def load_samples_named(
    file_path: str,
    param_names: List[str],
    discard: int = 0,
) -> Tuple[Dict[str, np.ndarray], 'ErynHDFBackend', np.ndarray, tuple]:
    """
    Load samples from an Eryn HDF backend into a parameter-keyed dictionary.

    Parameters
    ----------
    file_path   : path to the .h5 file
    param_names : ordered list of sampled parameter names (as returned by
                  ``sampled_params_from_config``); must match the number of
                  parameters stored in the backend
    discard     : number of burn-in iterations to discard

    Returns
    -------
    named_samples : {param_name: ndarray (N_temps, N_iters, N_walkers)}
    reader        : open ErynHDFBackend
    log_like      : ndarray (N_iters, N_temps, N_walkers)
    shape         : (N_iters, N_temps, N_walkers, N_params)
    """
    reader = ErynHDFBackend(file_path, read_only=True)
    chain  = reader.get_chain(discard=discard)['model_0']
    N_iters, N_temps, N_walkers = chain.shape[:3]
    N_params = chain.shape[-1]

    if len(param_names) != N_params:
        raise ValueError(
            f"len(param_names)={len(param_names)} does not match "
            f"N_params={N_params} stored in {file_path}"
        )

    # chain: (N_iters, N_temps, N_walkers, 1, N_params) — squeeze model dim
    traces = chain[:, :, :, 0, :]          # (N_iters, N_temps, N_walkers, N_params)
    traces = traces.transpose(1, 0, 2, 3)  # (N_temps, N_iters, N_walkers, N_params)

    named_samples = {
        name: traces[:, :, :, k]
        for k, name in enumerate(param_names)
    }
    log_like = reader.get_log_like(discard=discard)
    return named_samples, reader, log_like, (N_iters, N_temps, N_walkers, N_params)


# Log-likelihood plot

def plot_log_like(
    log_like: np.ndarray,
    temps: Optional[List[int]] = None,
    discard: int = 0,
    title: Optional[str] = None,
):
    """
    Plot log-likelihood traces — one stacked sub-panel per temperature so the
    cold chain (T=0) is never buried by the hot chains.

    Parameters
    ----------
    log_like : array (N_iters, N_temps, N_walkers)
    temps    : which temperature indices to plot; default is all
    discard  : offset added to iteration axis labels
    """
    N_iters, N_temps, _ = log_like.shape
    if temps is None:
        temps = list(range(N_temps))

    n = len(temps)
    fig, axes = plt.subplots(
        n, 1,
        figsize=(10, 2.5 * n),
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]

    iters   = np.arange(N_iters) + discard
    palette = sns.color_palette('coolwarm', n)

    for i, t in enumerate(temps):
        axes[i].plot(iters, log_like[:, t, :], alpha=0.5, lw=0.7, color=palette[i])
        axes[i].set_ylabel(r'$\ln\mathcal{L}$', fontsize=10)
        axes[i].set_title(f'T = {t}', fontsize=9, loc='right', pad=3)
        axes[i].tick_params(labelsize=8)

    axes[-1].set_xlabel('Iteration', fontsize=11)
    fig.suptitle(title or 'Log-likelihood traces', fontsize=12, y=1.01)
    plt.tight_layout()
    return fig


# Auto-correlation thinning

def cut_samples_autocorr(
    reader: ErynHDFBackend,
    samples_per_temp: List[np.ndarray],
    discard: int = 0,
) -> List[np.ndarray]:
    """
    Thin samples by the max integrated auto-correlation time.

    Returns the original list unchanged (with a warning) when the
    auto-correlation time is not well-defined.
    """
    try:
        tau = reader.get_autocorr_time(discard=discard)['model_0'][0]
        if np.any(~np.isfinite(tau)):
            raise ValueError("Auto-correlation time contains NaN / Inf.")
        thin = int(np.ceil(np.max(tau)))
        print(f"Auto-correlation time per param: {np.round(tau, 1)}")
        print(f"Thinning by factor {thin}.")
        return [s[::thin] for s in samples_per_temp]
    except Exception as exc:
        warnings.warn(
            f"Auto-correlation time not well-defined — skipping thinning. "
            f"Reason: {exc}"
        )
        return samples_per_temp


#  Lost-walker filtering 

def filter_lost_walkers(
    reader: ErynHDFBackend,
    discard: int = 0,
    lower_bound: float = -30.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove walkers whose minimum cold-chain log-likelihood falls below
    `lower_bound` (indicating a stuck or lost walker).

    Only the temperature-0 chain is filtered and returned.
    After filtering, a new log-likelihood plot is displayed.

    Returns
    -------
    clean_samples    : ndarray (N_kept * N_iters, N_params)
    kept_walker_idx  : ndarray of kept walker indices
    ll_clean         : ndarray (N_iters, N_kept)
    """
    chain    = reader.get_chain(discard=discard)['model_0']
    log_like = reader.get_log_like(discard=discard)
    N_iters, N_temps, N_walkers = chain.shape[:3]
    N_params = chain.shape[-1]

    ll_cold = log_like[:, 0, :]   # (N_iters, N_walkers)
    kept = np.array([
        w for w in range(N_walkers)
        if np.min(ll_cold[:, w]) > lower_bound
    ])

    if len(kept) == 0:
        warnings.warn(
            "All walkers removed — lower_bound may be too tight. Keeping all."
        )
        kept = np.arange(N_walkers)

    n_removed = N_walkers - len(kept)
    print(f"Removed {n_removed}/{N_walkers} lost walkers "
          f"(lower_bound={lower_bound:.1f}).")

    clean_samples = chain[:, 0, kept].reshape(-1, N_params)
    ll_clean      = ll_cold[:, kept]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ll_clean, alpha=0.7, lw=0.8)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Log likelihood')
    ax.set_title(f'Log likelihood (T=0) after removing {n_removed} lost walkers')
    plt.tight_layout()
    plt.show()

    return clean_samples, kept, ll_clean


#  Corner plot

def _extract_flat_samples(
    entry_samples,
    active_names: List[str],
    keep_idx: List[int],
    temp: int = 0,
) -> np.ndarray:
    """
    Return a (N_samples, N_active_params) array from either sample format.

    Parameters
    ----------
    entry_samples : ndarray (N, P)  *or*  {param_name: ndarray(N_temps, N_iters, N_walkers)}
    active_names  : ordered list of parameter names to extract
    keep_idx      : positional indices of active_names within the legacy ndarray columns
    temp          : temperature index to use when entry_samples is a named dict
    """
    if isinstance(entry_samples, dict):
        cols = [entry_samples[name][temp].ravel() for name in active_names]
        return np.column_stack(cols)
    return np.array(entry_samples)[:, keep_idx]


def corner_plot(
    samples_dict: Dict[str, Dict],
    true_vals: np.ndarray,
    param_names: List[str],
    inj_model: str = '',
    rec_model: str = '',
    exclude_sky: bool = False,
    temp: int = 0,
    plot_name: Optional[str] = None,
):
    """
    Overlay multiple posteriors on a single corner figure.

    Parameters
    ----------
    samples_dict : {label: {'color': str, 'samples': s}}
                   where ``s`` is either:
                   - ndarray (N, P) aligned column-wise with ``param_names``
                   - dict {param_name: ndarray (N_temps, N_iters, N_walkers)}
                     as returned by ``load_samples_named``
    true_vals    : injected values aligned with ``param_names``
    param_names  : reference parameter list (defines order and true-value alignment)
    inj_model    : injection model name shown in the title
    rec_model    : recovery model name shown in the title
    exclude_sky  : drop sky-position params (theta_S, phi_S, theta_K, phi_K)
    temp         : temperature index to use when samples are in named-dict format
    plot_name    : if given, save the figure here

    Notes
    -----
    When mixing 1PA (has chi2) and 0PA (no chi2) samples, pass each run's
    samples as a named dict from ``load_samples_named``.  The function takes
    the *intersection* of available parameter names across all entries so that
    each posterior is plotted on the correct axis regardless of its model's
    parameter ordering.
    """
    # Determine the set of params available in every entry.
    # Named-dict entries declare their own keys; legacy ndarray entries are
    # assumed to cover all of param_names.
    common = set(param_names)
    for entry in samples_dict.values():
        if isinstance(entry['samples'], dict):
            common &= set(entry['samples'].keys())

    keep_idx = [i for i, n in enumerate(param_names)
                if n in common and not (exclude_sky and n in SKY_PARAMS)]

    active_names  = [param_names[i] for i in keep_idx]
    active_true   = np.array([true_vals[i] for i in keep_idx])
    active_labels = [PARAM_LABELS.get(n, n) for n in active_names]
    N = len(active_names)

    if N == 0:
        warnings.warn("No parameters left to plot after exclusions.")
        return None

    # Font size scales with figure dimension
    fig_size  = max(4.0, N * 2.5)
    base_font = max(10, int(fig_size * 1.3))

    # Unified range across all datasets (robust percentiles)
    all_samples = np.concatenate(
        [_extract_flat_samples(e['samples'], active_names, keep_idx, temp)
         for e in samples_dict.values()],
        axis=0,
    )
    unified_range = []
    for i in range(N):
        col = all_samples[:, i]
        std = np.std(col)
        if std == 0:
            c = active_true[i] if np.isfinite(active_true[i]) else 0.0
            w = max(0.1 * abs(c), 1e-6)
            unified_range.append((c - w, c + w))
        else:
            lo, hi = np.percentile(col, [0.002, 99.998])
            unified_range.append((lo, hi))

    base_kwargs = dict(
        bins=30,
        plot_datapoints=False,
        smooth=True,
        smooth1d=True,
        labels=active_labels,
        levels=(1 - np.exp(-0.5), 1 - np.exp(-2.0), 1 - np.exp(-4.5)),
        label_kwargs=dict(fontsize=base_font),
        max_n_ticks=3,
        show_titles=False,
        labelpad=0.1,
        range=unified_range,
        quiet=True,
    )

    figure       = None
    legend_handles = []

    for label, entry in samples_dict.items():
        color   = entry['color']
        samples = _extract_flat_samples(entry['samples'], active_names, keep_idx, temp)

        try:
            figure = corner.corner(
                samples,
                fig=figure,
                color=color,
                alpha=0.7,
                **base_kwargs,
            )
        except Exception as exc:
            warnings.warn(f"corner.corner failed for '{label}': {exc}")
            continue

        legend_handles.append(
            mlines.Line2D([], [], color=color, alpha=0.7, label=label)
        )

    if figure is None:
        return None

    axes = np.array(figure.axes).reshape((N, N))

    for i in range(N):
        axes[i, i].axvline(active_true[i], color='k', lw=1.2)
    for yi in range(N):
        for xi in range(yi):
            axes[yi, xi].axhline(active_true[yi], color='k', lw=0.8)
            axes[yi, xi].axvline(active_true[xi], color='k', lw=0.8)
            axes[yi, xi].plot(active_true[xi], active_true[yi], 'sk', ms=3)

    tick_fs = max(5, base_font - 8)
    for ax in figure.get_axes():
        ax.tick_params(axis='both', labelsize=tick_fs)

    legend_handles.append(mlines.Line2D([], [], color='k', lw=1.2, label='True value'))

    title_parts = [p for p in [
        f'Injection: {inj_model}' if inj_model else '',
        f'Recovery: {rec_model}'  if rec_model  else '',
    ] if p]
    title = '  |  '.join(title_parts)

    # Place legend in the unused top-right corner of the lower-triangular grid.
    # For N==1 there are no empty panels, so fall back to a figure-level legend.
    if N > 1:
        legend_ax = axes[0, N - 1]
        legend_ax.legend(
            handles=legend_handles,
            fontsize=max(8, base_font - 2),
            frameon=True,
            loc='center',
        )
        legend_ax.axis('off')
    else:
        figure.legend(
            handles=legend_handles,
            fontsize=max(8, base_font - 2),
            frameon=True,
            loc='upper right',
        )

    if title:
        figure.suptitle(title, fontsize=base_font, y=1.01)

    figure.set_size_inches(fig_size, fig_size)

    if plot_name:
        figure.savefig(plot_name, bbox_inches='tight')

    return figure


# Sky position plot

def ecliptic_to_icrs(
    theta_S: np.ndarray,
    phi_S: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert EMRI sky parameters (ecliptic colatitude theta_S, ecliptic
    longitude phi_S) to equatorial (RA, Dec) in radians.

    Inverse of icrs_to_ecliptic: lambda = phi_S, beta = pi/2 - theta_S.
    RA is wrapped to [-pi, pi] for Mollweide axes.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    lam  = np.asarray(phi_S)
    beta = np.pi / 2.0 - np.asarray(theta_S)

    ecl  = SkyCoord(lon=lam * u.rad, lat=beta * u.rad,
                    frame='barycentrictrueecliptic')
    icrs = ecl.icrs
    ra   = icrs.ra.wrap_at(180 * u.deg).rad
    dec  = icrs.dec.rad
    return ra, dec


def _hdr_bounds(
    x: np.ndarray, y: np.ndarray, frac: float = 0.99
) -> Tuple[float, float, float, float]:
    """Return (xmin, xmax, ymin, ymax) of the HDR `frac` region via KDE."""
    from scipy.stats import gaussian_kde

    kde       = gaussian_kde(np.vstack([x, y]))
    z         = kde(np.vstack([x, y]))
    threshold = np.percentile(z, (1.0 - frac) * 100.0)
    mask      = z >= threshold
    return float(x[mask].min()), float(x[mask].max()), \
           float(y[mask].min()), float(y[mask].max())


def _sky_columns(
    entry_samples,
    param_names: List[str],
    temp: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract flat (N,) arrays for theta_S and phi_S from either sample format."""
    if isinstance(entry_samples, dict):
        return (
            entry_samples['theta_S'][temp].ravel(),
            entry_samples['phi_S'][temp].ravel(),
        )
    i_th = param_names.index('theta_S')
    i_ph = param_names.index('phi_S')
    arr  = np.asarray(entry_samples)
    return arr[:, i_th], arr[:, i_ph]


def plot_sky_position(
    samples_dict: Dict[str, Dict],
    param_names: List[str],
    true_theta_S: Optional[float] = None,
    true_phi_S:   Optional[float] = None,
    temp: int = 0,
    plot_name: Optional[str] = None,
):
    """
    Mollweide sky map with a zoomed Cartesian inset on the 99 % HDR region.

    Parameters
    ----------
    samples_dict : {label: {'color': str, 'samples': s}}
                   where ``s`` is ndarray (N, P) or named dict from
                   ``load_samples_named``
    param_names  : names of the sampled parameters (used for legacy ndarray lookup)
    true_theta_S : injected ecliptic colatitude (rad), optional
    true_phi_S   : injected ecliptic longitude (rad), optional
    temp         : temperature index to use for named-dict samples
    """
    # Check that sky params are available in at least the first entry
    first_samples = next(iter(samples_dict.values()))['samples']
    if isinstance(first_samples, dict):
        if 'theta_S' not in first_samples or 'phi_S' not in first_samples:
            warnings.warn("theta_S or phi_S not in samples dict — sky plot skipped.")
            return None
    else:
        if 'theta_S' not in param_names or 'phi_S' not in param_names:
            warnings.warn("theta_S or phi_S not in param_names — sky plot skipped.")
            return None

    fig = plt.figure(figsize=(14, 7))
    ax_main = fig.add_subplot(111, projection='mollweide')
    ax_main.grid(True, alpha=0.3)

    all_ra: List[np.ndarray] = []
    all_dec: List[np.ndarray] = []
    legend_handles = []

    for label, entry in samples_dict.items():
        color    = entry['color']
        th_S, ph_S = _sky_columns(entry['samples'], param_names, temp)
        ra, dec = ecliptic_to_icrs(th_S, ph_S)
        all_ra.append(ra)
        all_dec.append(dec)

        ax_main.scatter(ra, dec, s=0.8, alpha=0.25, color=color, rasterized=True)
        legend_handles.append(
            mlines.Line2D([], [], color=color, marker='o', linestyle='None',
                          markersize=5, label=label)
        )

    if true_theta_S is not None and true_phi_S is not None:
        ra_true, dec_true = ecliptic_to_icrs(
            np.array([true_theta_S]), np.array([true_phi_S])
        )
        ax_main.scatter(ra_true, dec_true, marker='*', s=250, color='k',
                        zorder=6)
        legend_handles.append(
            mlines.Line2D([], [], color='k', marker='*', linestyle='None',
                          markersize=10, label='True value')
        )

    ax_main.set_xlabel('RA (rad)', labelpad=10)
    ax_main.set_ylabel('Dec (rad)')
    ax_main.legend(handles=legend_handles, loc='lower left', fontsize=10,
                   framealpha=0.8)

    # ── Inset: 99 % HDR zoom (Cartesian )
    if all_ra:
        ra_all  = np.concatenate(all_ra)
        dec_all = np.concatenate(all_dec)

        try:
            x0, x1, y0, y1 = _hdr_bounds(ra_all, dec_all, frac=0.99)
            mx = max(0.05 * abs(x1 - x0), 0.005)
            my = max(0.05 * abs(y1 - y0), 0.005)

            # Place inset in figure coordinates (avoids GeoAxes issues)
            ax_ins = fig.add_axes([0.62, 0.12, 0.30, 0.38])

            for label, entry in samples_dict.items():
                color   = entry['color']
                th_S_i, ph_S_i = _sky_columns(entry['samples'], param_names, temp)
                ra, dec = ecliptic_to_icrs(th_S_i, ph_S_i)
                ax_ins.scatter(ra, dec, s=1.0, alpha=0.25,
                               color=color, rasterized=True)

                mask = (
                    (ra  > x0 - mx) & (ra  < x1 + mx) &
                    (dec > y0 - my) & (dec < y1 + my)
                )
                if mask.sum() > 50:
                    try:
                        from scipy.stats import gaussian_kde
                        kde = gaussian_kde(np.vstack([ra[mask], dec[mask]]))
                        xi  = np.linspace(x0 - mx, x1 + mx, 80)
                        yi  = np.linspace(y0 - my, y1 + my, 80)
                        Xi, Yi = np.meshgrid(xi, yi)
                        Zi = kde(
                            np.vstack([Xi.ravel(), Yi.ravel()])
                        ).reshape(Xi.shape)
                        z_pts = kde(np.vstack([ra[mask], dec[mask]]))
                        fracs  = [0.50, 1 - np.exp(-0.5), 1 - np.exp(-2.0)]
                        levels = sorted(
                            np.percentile(z_pts, (1.0 - f) * 100) for f in fracs
                        )
                        ax_ins.contour(Xi, Yi, Zi, levels=levels,
                                       colors=color, linewidths=1.2, alpha=0.85)
                    except Exception:
                        pass

            if true_theta_S is not None:
                ax_ins.scatter(ra_true, dec_true, marker='*', s=200,
                               color='k', zorder=6)

            ax_ins.set_xlim(x0 - mx, x1 + mx)
            ax_ins.set_ylim(y0 - my, y1 + my)
            ax_ins.set_xlabel('RA (rad)', fontsize=8)
            ax_ins.set_ylabel('Dec (rad)', fontsize=8)
            ax_ins.tick_params(labelsize=7)
            ax_ins.set_title('99 % HDR region', fontsize=9)
            ax_ins.grid(True, alpha=0.3, linewidth=0.5)

            # Draw a rectangle on the Mollweide plot to indicate the zoom region
            rect = mpatches.Rectangle(
                (x0 - mx, y0 - my),
                (x1 + mx) - (x0 - mx),
                (y1 + my) - (y0 - my),
                linewidth=1.2, edgecolor='gray', facecolor='none',
                linestyle='--', transform=ax_main.transData,
            )
            ax_main.add_patch(rect)

        except Exception as exc:
            warnings.warn(f"Could not create HDR inset: {exc}")

    plt.suptitle('Sky position — equatorial coordinates', fontsize=13)
    if plot_name:
        fig.savefig(plot_name, bbox_inches='tight')
    plt.show()
    return fig


# Chain convergence

def plot_chain_convergence(
    reader: ErynHDFBackend,
    param_names: List[str],
    discard: int = 0,
    temp: int = 0,
    plot_name: Optional[str] = None,
):
    """
    Per-parameter MCMC trace plot showing all walkers at temperature `temp`.
    """
    chain  = reader.get_chain(discard=discard)['model_0']
    N_iters, N_temps, N_walkers = chain.shape[:3]
    N_params = chain.shape[-1]
    traces = chain[:, temp, :, 0, :]   # (N_iters, N_walkers, N_params)

    labels = [PARAM_LABELS.get(n, n) for n in param_names]
    ncols  = min(3, N_params)
    nrows  = int(np.ceil(N_params / ncols))
    iters  = np.arange(N_iters) + discard

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5 * ncols, 3 * nrows),
        sharex=True, squeeze=False,
    )
    walker_color = sns.color_palette('Blues_d', 1)[0]

    for k in range(N_params):
        ax = axes[k // ncols][k % ncols]
        ax.plot(iters, traces[:, :, k], alpha=0.35, lw=0.6, color=walker_color)
        ax.set_ylabel(labels[k], fontsize=12)
        ax.tick_params(labelsize=9)

    for k in range(N_params, nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)

    for ax in axes[-1]:
        ax.set_xlabel('Iteration', fontsize=11)

    fig.suptitle(f'Chain traces  (T={temp})', fontsize=14, y=1.01)
    plt.tight_layout()
    if plot_name:
        fig.savefig(plot_name, bbox_inches='tight')
    plt.show()
    return fig


# Covariance evolution 

def plot_covariance_evolution(
    reader: ErynHDFBackend,
    param_names: List[str],
    discard: int = 0,
    temp: int = 0,
    step: int = 50,
    plot_name: Optional[str] = None,
):
    """
    Running marginal variance (diagonal of the sample covariance matrix) as a
    function of iteration, at temperature `temp`.

    `step` controls the evaluation stride; smaller → smoother but slower.
    """
    chain  = reader.get_chain(discard=discard)['model_0']
    N_iters, N_temps, N_walkers = chain.shape[:3]
    N_params = chain.shape[-1]
    traces = chain[:, temp, :, 0, :]   # (N_iters, N_walkers, N_params)

    labels      = [PARAM_LABELS.get(n, n) for n in param_names]
    checkpoints = np.arange(step, N_iters + 1, step)
    running_var = np.zeros((len(checkpoints), N_params))

    for ci, end in enumerate(checkpoints):
        flat = traces[:end].reshape(-1, N_params)
        running_var[ci] = np.var(flat, axis=0, ddof=1)

    iter_axis = checkpoints + discard
    palette   = sns.color_palette('husl', N_params)

    ncols = min(3, N_params)
    nrows = int(np.ceil(N_params / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5 * ncols, 3 * nrows),
        sharex=True, squeeze=False,
    )

    for k in range(N_params):
        ax = axes[k // ncols][k % ncols]
        ax.plot(iter_axis, running_var[:, k], color=palette[k], lw=1.5)
        ax.axhline(running_var[-1, k], color='k', ls='--', lw=0.8, alpha=0.5)
        ax.set_ylabel(f'Var({labels[k]})', fontsize=10)
        ax.tick_params(labelsize=9)

    for k in range(N_params, nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)

    for ax in axes[-1]:
        ax.set_xlabel('Iteration', fontsize=11)

    fig.suptitle(f'Running marginal variance  (T={temp})', fontsize=14, y=1.01)
    plt.tight_layout()
    if plot_name:
        fig.savefig(plot_name, bbox_inches='tight')
    plt.show()
    return fig


#  Gelman–Rubin convergence test

def plot_gelman_rubin(
    reader: ErynHDFBackend,
    param_names: List[str],
    discard: int = 0,
    temp: int = 0,
    step: int = 50,
    convergence_threshold: float = 1.01,
    plot_name: Optional[str] = None,
):
    """
    Running Gelman–Rubin R-hat statistic per parameter.

    R-hat \sim 1 indicates convergence. The dashed line marks
    `convergence_threshold` (default 1.01).

    Returns
    -------
    rhat : ndarray (len(checkpoints), N_params)
    """
    chain  = reader.get_chain(discard=discard)['model_0']
    N_iters, N_temps, N_walkers = chain.shape[:3]
    N_params = chain.shape[-1]
    # traces: (N_iters, N_walkers, N_params)
    traces = chain[:, temp, :, 0, :]

    labels      = [PARAM_LABELS.get(n, n) for n in param_names]
    checkpoints = np.arange(2 * step, N_iters + 1, step)
    rhat        = np.full((len(checkpoints), N_params), np.nan)

    for ci, end in enumerate(checkpoints):
        sub = traces[:end]                  # (end, N_walkers, N_params)
        N   = sub.shape[0]
        M   = N_walkers
        # within-chain variance
        s2   = np.var(sub, axis=0, ddof=1)         # (M, P)
        W    = s2.mean(axis=0)                      # (P,)
        # between-chain variance
        mu_m = sub.mean(axis=0)                     # (M, P)
        mu   = mu_m.mean(axis=0)                    # (P,)
        B    = N / (M - 1) * np.sum((mu_m - mu) ** 2, axis=0)
        var_plus = ((N - 1) / N) * W + B / N
        with np.errstate(invalid='ignore', divide='ignore'):
            rhat[ci] = np.sqrt(np.where(W > 0, var_plus / W, np.nan))

    iter_axis = checkpoints + discard
    palette   = sns.color_palette('husl', N_params)

    ncols = min(3, N_params)
    nrows = int(np.ceil(N_params / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5 * ncols, 3 * nrows),
        sharex=True, squeeze=False,
    )

    for k in range(N_params):
        ax = axes[k // ncols][k % ncols]
        ax.plot(iter_axis, rhat[:, k], color=palette[k], lw=1.5)
        ax.axhline(convergence_threshold, color='k', ls='--', lw=0.9,
                   alpha=0.6, label=f'R-hat = {convergence_threshold}')
        ax.axhline(1.0, color='gray', ls=':', lw=0.8, alpha=0.5)
        ax.set_ylabel(f'R-hat  {labels[k]}', fontsize=10)
        ax.tick_params(labelsize=9)

    for k in range(N_params, nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)

    for ax in axes[-1]:
        ax.set_xlabel('Iteration', fontsize=11)

    fig.suptitle(f'Gelman–Rubin R-hat  (T={temp})', fontsize=14, y=1.01)
    plt.tight_layout()
    if plot_name:
        fig.savefig(plot_name, bbox_inches='tight')
    plt.show()
    return rhat


# Seaborn diagnostics 

def plot_seaborn_diagnostics(
    samples: np.ndarray,
    param_names: List[str],
    title: str = '',
    max_pair_params: int = 6,
    plot_name: Optional[str] = None,
):
    """
    Seaborn-based statistical diagnostics:
      - KDE pair-plot (first `max_pair_params` parameters for legibility)
      - Violin plots across all parameters

    Parameters
    ----------
    samples         : ndarray (N, P) aligned with `param_names`
    param_names     : names of the sampled parameters
    max_pair_params : maximum number of params shown in the pair plot
    """
    import pandas as pd

    labels = [PARAM_LABELS.get(n, n) for n in param_names]
    df     = pd.DataFrame(samples, columns=labels)

    #  Pair plot
    pair_cols = labels[:max_pair_params]
    g = sns.pairplot(
        df[pair_cols],
        diag_kind='kde',
        plot_kws=dict(alpha=0.25, s=4, rasterized=True),
        diag_kws=dict(fill=True, alpha=0.5),
    )
    g.fig.suptitle(
        f'Pair plot{": " + title if title else ""}',
        y=1.01, fontsize=13,
    )
    plt.tight_layout()
    if plot_name:
        g.fig.savefig(
            plot_name.replace('.', '_pairplot.', 1), bbox_inches='tight'
        )
    plt.show()

    #  Violin plot
    # Normalize columns (z-score) so all fit on a single axis
    df_norm = (df - df.mean()) / df.std().replace(0, 1)
    df_melted = df_norm.melt(var_name='Parameter', value_name='z-score')

    fig, ax = plt.subplots(figsize=(max(6, 1.8 * len(labels)), 5))
    palette = sns.color_palette('husl', len(labels))
    sns.violinplot(
        data=df_melted, x='Parameter', y='z-score',
        palette=palette, inner='quartile', linewidth=0.8, ax=ax,
    )
    ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    ax.set_xlabel('')
    ax.set_ylabel('Normalised value (z-score)', fontsize=11)
    ax.tick_params(axis='x', labelrotation=30, labelsize=9)
    fig.suptitle(
        f'Violin plots{": " + title if title else ""}', fontsize=13
    )
    plt.tight_layout()
    if plot_name:
        fig.savefig(
            plot_name.replace('.', '_violin.', 1), bbox_inches='tight'
        )
    plt.show()
    return fig
