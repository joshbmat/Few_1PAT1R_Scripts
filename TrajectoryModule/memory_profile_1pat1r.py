"""Profile the 1PAT1R trajectory using memray."""

from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode.IPAT1R import Trajectory1PAT1R

traj_module = EMRIInspiral(func=Trajectory1PAT1R)

m1 = 1e4
m2 = 1e1
chi1 = 0.1
p0 = 15.0
e0 = 0.0
xI0 = 1.0
chi2 = 0.5

t, p, e, x, Phi_phi, Phi_theta, Phi_r, delta_M, delta_a = traj_module(
    m1, m2, chi1, p0, e0, xI0, chi2, T=1.0
)

print(f"Trajectory computed: {len(t)} points, p range [{p[-1]:.4f}, {p[0]:.4f}]")
