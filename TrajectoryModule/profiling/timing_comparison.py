"""
Timing comparison: Trajectory1PAT1R vs KerrEccEqFlux

Benchmarks wall-clock timing of the 1PA trajectory model against
the adiabatic KerrEccEqFlux model on their overlapping parameter space
(circular, equatorial, low primary spin).
"""

import time
import statistics

from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode import Trajectory1PAT1R, KerrEccEqFlux

# ── Constants ──────────────────────────────────────────────────────────
N_REPEATS = 10
N_WARMUP = 3
DEFAULT_T = 2.0      # years
DEFAULT_DT = 10.0    # seconds
DEFAULT_ERR = 1e-10


# ── Helpers ────────────────────────────────────────────────────────────

def time_model(traj, args, kwargs):
    """Run warmup calls, then timed repeats. Return (median_seconds, n_points)."""
    # Warmup (handles numba JIT)
    for _ in range(N_WARMUP):
        try:
            result = traj(*args, **kwargs)
        except ValueError:
            print(*args)
            raise ValueError()

    times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        try:
            result = traj(*args, **kwargs)
        except ValueError:
            print(*args)
        
        t1 = time.perf_counter()
        times.append(t1 - t0)

    n_points = len(result[0])  # length of time array
    return statistics.median(times), n_points


def print_table(title, col_headers, rows):
    """Print a formatted table."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")

    # Compute column widths
    widths = [len(h) for h in col_headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)))

    # Header
    header = " | ".join(h.ljust(widths[i]) for i, h in enumerate(col_headers))
    print(header)
    print("-+-".join("-" * w for w in widths))

    # Rows
    for row in rows:
        line = " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(row))
        print(line)

    print()


# ── Suite 1: Head-to-head comparison ──────────────────────────────────

def run_comparison(traj_1pa, traj_flux):
    """Compare timing across 10 physics configurations."""
    configs = [
        # (label, m1, m2, a, p0, chi2)
        ("EMRI, no spin, p0=10",    1e6, 10,  0.0,  10.0, 0.0),
        ("EMRI, no spin, p0=20",    1e6, 10,  0.0,  20.0, 0.0),
        ("EMRI, chi1=0.1, p0=10",   1e6, 10,  0.1,  10.0, 0.0),
        ("EMRI, chi1=0.2, p0=10",   1e6, 10,  0.2,  10.0, 0.0),
        ("EMRI, chi1=-0.1, p0=10",  1e6, 10, -0.1,  10.0, 0.0),
        ("IMRI, no spin, p0=10",    1e4, 100, 0.0,  10.0, 0.0),
        ("IMRI, chi1=0.1, p0=10",   1e4, 100, 0.1,  10.0, 0.0),
        ("EMRI, chi2=0.5, p0=10",   1e6, 10,  0.0,  10.0, 0.5),
        ("Near separatrix, p0=6.5", 1e6, 10,  0.0,   6.5, 0.0),
        ("Wide orbit, p0=28",       1e6, 10,  0.0,  28.0, 0.0),
    ]

    kwargs = dict(T=DEFAULT_T, dt=DEFAULT_DT, err=DEFAULT_ERR)
    rows = []

    for label, m1, m2, a, p0, chi2 in configs:
        # 1PAT1R: (m1, m2, a, p0, e0, x0, chi2)
        args_1pa = (m1, m2, a, p0, 0.0, 1.0, chi2)
        t_1pa, n_1pa = time_model(traj_1pa, args_1pa, kwargs)

        # KerrEccEqFlux: (m1, m2, a, p0, e0, x0)  — no chi2
        args_flux = (m1, m2, a, p0, 0.0, 1.0)
        t_flux, n_flux = time_model(traj_flux, args_flux, kwargs)

        ratio = t_1pa / t_flux if t_flux > 0 else float("inf")
        rows.append((
            label,
            f"{t_1pa * 1e3:.2f}",
            n_1pa,
            f"{t_flux * 1e3:.2f}",
            n_flux,
            f"{ratio:.2f}x",
        ))

    print_table(
        "Head-to-head comparison (e0=0, xI0=1, T=1yr, dt=10s, err=1e-10)",
        ["Config", "1PAT1R (ms)", "n_pts", "KerrFlux (ms)", "n_pts", "Ratio"],
        rows,
    )


# ── Suite 2: Tolerance sweep ──────────────────────────────────────────

def run_tolerance_sweep(traj_1pa, traj_flux):
    """Vary integration tolerance."""
    errs = [1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-14]
    m1, m2, a, p0, chi2 = 1e6, 10, 0.0, 10.0, 0.0
    rows = []

    for err in errs:
        kwargs = dict(T=DEFAULT_T, dt=DEFAULT_DT, err=err)

        args_1pa = (m1, m2, a, p0, 0.0, 1.0, chi2)
        t_1pa, n_1pa = time_model(traj_1pa, args_1pa, kwargs)

        args_flux = (m1, m2, a, p0, 0.0, 1.0)
        t_flux, n_flux = time_model(traj_flux, args_flux, kwargs)

        ratio = t_1pa / t_flux if t_flux > 0 else float("inf")
        rows.append((
            f"{err:.0e}",
            f"{t_1pa * 1e3:.2f}",
            n_1pa,
            f"{t_flux * 1e3:.2f}",
            n_flux,
            f"{ratio:.2f}x",
        ))

    print_table(
        "Tolerance sweep (m1=1e6, m2=10, a=0, p0=10, T=1yr)",
        ["err", "1PAT1R (ms)", "n_pts", "KerrFlux (ms)", "n_pts", "Ratio"],
        rows,
    )


# ── Suite 3: Observation time sweep ──────────────────────────────────

def run_observation_time_sweep(traj_1pa, traj_flux):
    """Vary observation time."""
    T_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    m1, m2, a, p0, chi2 = 1e6, 10, 0.0, 15.0, 0.0
    rows = []

    for T in T_values:
        kwargs = dict(T=T, dt=DEFAULT_DT, err=DEFAULT_ERR)

        args_1pa = (m1, m2, a, p0, 0.0, 1.0, chi2)
        t_1pa, n_1pa = time_model(traj_1pa, args_1pa, kwargs)

        args_flux = (m1, m2, a, p0, 0.0, 1.0)
        t_flux, n_flux = time_model(traj_flux, args_flux, kwargs)

        ratio = t_1pa / t_flux if t_flux > 0 else float("inf")
        rows.append((
            f"{T:.1f}",
            f"{t_1pa * 1e3:.2f}",
            n_1pa,
            f"{t_flux * 1e3:.2f}",
            n_flux,
            f"{ratio:.2f}x",
        ))

    print_table(
        "Observation time sweep (m1=1e6, m2=10, a=0, p0=15, err=1e-10)",
        ["T (yr)", "1PAT1R (ms)", "n_pts", "KerrFlux (ms)", "n_pts", "Ratio"],
        rows,
    )


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("Instantiating trajectory models...")
    traj_1pa = EMRIInspiral(func=Trajectory1PAT1R)
    traj_flux = EMRIInspiral(func=KerrEccEqFlux)
    print("Done.\n")

    run_comparison(traj_1pa, traj_flux)
    run_tolerance_sweep(traj_1pa, traj_flux)
    run_observation_time_sweep(traj_1pa, traj_flux)


if __name__ == "__main__":
    main()
