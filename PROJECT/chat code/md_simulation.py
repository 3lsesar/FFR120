"""
md_simulation.py
=================

This module contains the code used to simulate granular flow in a two-dimensional silo
with inclined walls and an orifice at the base. The model treats the grains as
soft disks interacting via a linear spring and Coulomb friction. The base of the
silo remains closed for an initial compaction phase and is then opened at a
specified time, allowing particles to exit through the orifice if they reach the
base within the opening. A jam is detected when no grains exit for a period
``stall_time`` after the orifice is opened.

Functions provided:

* ``bottom_height``: Returns the vertical coordinate of the inclined floor as a
  function of the horizontal position ``x`` and the orifice width ``D``.
* ``simulate_md``: Runs the dynamic simulation for a given set of parameters
  (number of particles, radius, orifice width, friction coefficient, etc.) and
  returns whether a jam occurred and how many particles were discharged.
* ``experiment_vs_D``: Sweeps over a range of orifice widths ``D`` for a fixed
  friction coefficient and returns jam probabilities and average discharged
  counts over multiple replicas.
* ``experiment_vs_mu``: Sweeps over different friction coefficients for a fixed
  orifice width, returning jam probabilities and mean discharged counts.
* ``generate_snapshot``: Runs a simulation with the orifice closed, then opened
  at ``t_open``, and returns a list of circle patches representing particle
  positions when the jam is detected or at final time.

These functions were used in the final experiments and figures of the report.
You can use this module to build your own notebook or script to reproduce
simulations and plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Global physical parameters
g = -9.81  # gravity (units of length/time^2)
W = 1.0    # silo width
H = 1.5    # silo height

def bottom_height(x: np.ndarray, D: float, slope: float = 1.0) -> np.ndarray:
    """Return the bottom surface height at positions ``x`` for a given
    orifice width ``D`` and slope.

    Outside the orifice ``|x| > D/2``, the floor rises linearly with
    gradient ``slope``. Within the orifice region ``|x| <= D/2``, the
    floor is at height zero.

    Parameters
    ----------
    x : array_like
        Horizontal positions where to evaluate the floor height.
    D : float
        Width of the orifice (dimensionless units).
    slope : float, optional
        Slope of the inclined floor outside the orifice.

    Returns
    -------
    np.ndarray
        Floor heights at the corresponding ``x`` positions.
    """
    half = D / 2.0
    return np.where(np.abs(x) <= half, 0.0, slope * (np.abs(x) - half))


def simulate_md(
    N: int = 30,
    r: float = 0.02,
    D: float = 0.1,
    mu: float = 0.6,
    slope: float = 1.0,
    t_open: float = 0.5,
    stall_time: float = 1.0,
    dt: float = 0.001,
    max_time: float = 2.0,
    seed: int | None = None,
    noise: bool = False,
) -> tuple[bool, int]:
    """Simulate granular flow with friction and report jam and discharged count.

    The simulation places ``N`` particles of radius ``r`` (with optional
    Gaussian noise) randomly near the top of the silo. Particles interact via
    linear spring forces and Coulomb friction. The silo walls are vertical
    and the floor is inclined outside the orifice of width ``D``. The orifice
    remains closed until ``t_open``, after which particles that reach ``y <= 0``
    within the horizontal bounds ``|x| < D/2`` are removed from the system. If
    no particles are removed for ``stall_time`` after the opening and
    particles remain, the system is considered jammed.

    Parameters
    ----------
    N : int, optional
        Number of particles to simulate.
    r : float, optional
        Nominal radius of each particle.
    D : float, optional
        Width of the orifice.
    mu : float, optional
        Coefficient of friction for Coulomb tangential forces.
    slope : float, optional
        Slope of the inclined floor.
    t_open : float, optional
        Time at which the orifice opens.
    stall_time : float, optional
        Time without discharge used to detect a jam.
    dt : float, optional
        Integration time step.
    max_time : float, optional
        Maximum simulation time. If reached before jam detection, a jam is
        assumed if particles remain.
    seed : int or None, optional
        Random seed for reproducibility. Different seeds lead to different
        initial positions.
    noise : bool, optional
        If True, draws each particle radius from a normal distribution with
        mean ``r`` and standard deviation ``0.002``.

    Returns
    -------
    jam : bool
        True if a jam was detected before all particles exited; False if all
        particles were removed without jamming.
    removed_count : int
        Number of particles that exited the silo by the end of the simulation.
    """
    rng = np.random.default_rng(seed)
    # Assign radii (polydisperse if noise)
    if noise:
        radii = np.clip(r + rng.normal(scale=0.002, size=N), 0.005, None)
    else:
        radii = np.full(N, r)
    # Initial positions: random horizontal positions and random heights in [H-2r, H]
    positions = np.zeros((N, 2))
    positions[:, 0] = rng.uniform(-W/2 + radii, W/2 - radii)
    positions[:, 1] = H - 2 * radii * rng.random(N)
    # Velocities start at zero
    velocities = np.zeros((N, 2))
    # Spring constant for normal contacts
    k = 10000.0
    removed_count = 0
    last_removed_time = 0.0
    time = 0.0
    jam = False
    active_mask = np.ones(N, dtype=bool)
    while time < max_time:
        # Compute forces
        forces = np.zeros((N, 2))
        forces[active_mask, 1] += g  # gravity
        active_idx = np.where(active_mask)[0]
        # Particle-particle interactions
        for i in range(len(active_idx)):
            a = active_idx[i]
            for j in range(i + 1, len(active_idx)):
                b = active_idx[j]
                delta = positions[b] - positions[a]
                dist = np.linalg.norm(delta)
                overlap = radii[a] + radii[b] - dist
                if overlap > 0:
                    # Compute normal and tangential forces
                    n = delta / (dist + 1e-8)
                    fn = k * overlap
                    dv = velocities[a] - velocities[b]
                    vt = dv - np.dot(dv, n) * n
                    f_t_mag = np.linalg.norm(vt)
                    if f_t_mag > 1e-8:
                        ft_dir = -vt / (f_t_mag + 1e-8)
                        ft = mu * fn * ft_dir
                    else:
                        ft = np.zeros(2)
                    forces[a] -= (fn * n + ft)
                    forces[b] += (fn * n + ft)
        # Particle-wall and particle-floor interactions
        for idx in active_idx:
            x, y = positions[idx]
            rad = radii[idx]
            # Vertical walls
            if x - rad < -W/2:
                overlap = -W/2 - (x - rad)
                forces[idx][0] += k * overlap
            if x + rad > W/2:
                overlap = (x + rad) - W/2
                forces[idx][0] -= k * overlap
            # Inclined floor
            bh = bottom_height(x, D, slope)
            if y - rad < bh:
                overlap = bh - (y - rad)
                # Normal: vertical inside the opening, inclined outside
                if abs(x) <= D / 2:
                    normal = np.array([0.0, 1.0])
                else:
                    sign = 1.0 if x >= 0 else -1.0
                    slope_vec = np.array([sign, slope])
                    normal = np.array([-slope_vec[1], slope_vec[0]])
                    normal /= np.linalg.norm(normal)
                fn = k * overlap
                v_rel = velocities[idx]
                vt = v_rel - np.dot(v_rel, normal) * normal
                f_t_mag = np.linalg.norm(vt)
                if f_t_mag > 1e-8:
                    ft_dir = -vt / (f_t_mag + 1e-8)
                    ft = mu * fn * ft_dir
                else:
                    ft = np.zeros(2)
                forces[idx] += fn * normal + ft
        # Update velocities and positions
        velocities[active_mask] += forces[active_mask] * dt
        positions[active_mask] += velocities[active_mask] * dt
        time += dt
        # Remove particles that exit after orifice opens
        if time > t_open:
            to_remove = []
            for idx in active_idx:
                x, y = positions[idx]
                if y - radii[idx] <= 0 and abs(x) < D / 2:
                    to_remove.append(idx)
            if to_remove:
                for idx in to_remove:
                    active_mask[idx] = False
                removed_count += len(to_remove)
                last_removed_time = time
        # Jam detection: stall time after first discharge
        if time > t_open and (time - last_removed_time) > stall_time:
            if active_mask.sum() == 0:
                jam = False
            else:
                jam = True
            break
    if time >= max_time and active_mask.sum() > 0:
        jam = True
    return jam, removed_count


def experiment_vs_D(
    mu: float = 0.6,
    D_range: np.ndarray | list = np.arange(0.06, 0.31, 0.03),
    replicates: int = 3,
    N: int = 30,
    r: float = 0.02,
    noise: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute jam probability and mean discharged particles as a function of D.

    Runs ``replicates`` simulations for each orifice width ``D`` in ``D_range``
    and computes the fraction of jams and the mean number of discharged discs.
    """
    jam_probs = []
    removed_means = []
    for D in D_range:
        jam_count = 0
        removed_list = []
        for rep in range(replicates):
            jam, removed = simulate_md(N=N, r=r, D=D, mu=mu, noise=noise, seed=rep)
            jam_count += int(jam)
            removed_list.append(removed)
        jam_probs.append(jam_count / replicates)
        removed_means.append(np.mean(removed_list))
    return np.array(jam_probs), np.array(removed_means)


def experiment_vs_mu(
    D: float,
    mu_values: list[float],
    replicates: int = 3,
    N: int = 30,
    r: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute jam probability and mean discharged particles as a function of mu.

    Runs ``replicates`` simulations for each friction coefficient in ``mu_values``
    at a fixed orifice width ``D``.
    """
    jam_probs = []
    removed_means = []
    for mu_val in mu_values:
        jam_count = 0
        removed_list = []
        for rep in range(replicates):
            jam, removed = simulate_md(N=N, r=r, D=D, mu=mu_val, seed=rep)
            jam_count += int(jam)
            removed_list.append(removed)
        jam_probs.append(jam_count / replicates)
        removed_means.append(np.mean(removed_list))
    return np.array(jam_probs), np.array(removed_means)


def generate_snapshot(
    N: int = 30,
    r: float = 0.02,
    D: float = 0.1,
    mu: float = 0.6,
    slope: float = 1.0,
    t_open: float = 0.5,
    stall_time: float = 1.0,
    dt: float = 0.001,
    max_time: float = 2.0,
    seed: int | None = None,
    noise: bool = False,
) -> list[Circle]:
    """Simulate the system and return particle circles at jam or end time.

    This helper runs ``simulate_md`` but, instead of returning jam/removed
    statistics, it records particle positions at the moment a jam is
    detected (or at the end of the simulation). It returns a list of
    ``matplotlib.patches.Circle`` objects with proper radii that can be
    directly added to a plot.
    """
    rng = np.random.default_rng(seed)
    # Set up radii and initial positions
    if noise:
        radii = np.clip(r + rng.normal(scale=0.002, size=N), 0.005, None)
    else:
        radii = np.full(N, r)
    positions = np.zeros((N, 2))
    positions[:, 0] = rng.uniform(-W/2 + radii, W/2 - radii)
    positions[:, 1] = H - 2 * radii * rng.random(N)
    velocities = np.zeros((N, 2))
    k = 10000.0
    time = 0.0
    last_removed_time = 0.0
    jam_detected = False
    active_mask = np.ones(N, dtype=bool)
    # Simulate until jam or max_time
    while time < max_time:
        forces = np.zeros((N, 2))
        forces[active_mask, 1] += g
        active_idx = np.where(active_mask)[0]
        for i in range(len(active_idx)):
            a = active_idx[i]
            for j in range(i + 1, len(active_idx)):
                b = active_idx[j]
                delta = positions[b] - positions[a]
                dist = np.linalg.norm(delta)
                overlap = radii[a] + radii[b] - dist
                if overlap > 0:
                    n = delta / (dist + 1e-8)
                    fn = k * overlap
                    dv = velocities[a] - velocities[b]
                    vt = dv - np.dot(dv, n) * n
                    f_t_mag = np.linalg.norm(vt)
                    if f_t_mag > 1e-8:
                        ft_dir = -vt / (f_t_mag + 1e-8)
                        ft = mu * fn * ft_dir
                    else:
                        ft = np.zeros(2)
                    forces[a] -= (fn * n + ft)
                    forces[b] += (fn * n + ft)
        for idx in active_idx:
            x, y = positions[idx]
            rad = radii[idx]
            # Walls
            if x - rad < -W/2:
                overlap = -W/2 - (x - rad)
                forces[idx][0] += k * overlap
            if x + rad > W/2:
                overlap = (x + rad) - W/2
                forces[idx][0] -= k * overlap
            # Inclined floor
            bh = bottom_height(x, D, slope)
            if y - rad < bh:
                overlap = bh - (y - rad)
                if abs(x) <= D / 2:
                    normal = np.array([0.0, 1.0])
                else:
                    sign = 1.0 if x >= 0 else -1.0
                    slope_vec = np.array([sign, slope])
                    normal = np.array([-slope_vec[1], slope_vec[0]]) / np.linalg.norm(slope_vec)
                fn = k * overlap
                v_rel = velocities[idx]
                vt = v_rel - np.dot(v_rel, normal) * normal
                f_t_mag = np.linalg.norm(vt)
                if f_t_mag > 1e-8:
                    ft_dir = -vt / (f_t_mag + 1e-8)
                    ft = mu * fn * ft_dir
                else:
                    ft = np.zeros(2)
                forces[idx] += fn * normal + ft
        # Integrate
        velocities[active_mask] += forces[active_mask] * dt
        positions[active_mask] += velocities[active_mask] * dt
        time += dt
        if time > t_open:
            to_remove = []
            for idx in active_idx:
                x, y = positions[idx]
                if y - radii[idx] <= 0 and abs(x) < D / 2:
                    to_remove.append(idx)
            if to_remove:
                for idx in to_remove:
                    active_mask[idx] = False
                last_removed_time = time
        if time > t_open and (time - last_removed_time) > stall_time:
            jam_detected = active_mask.sum() > 0
            break
    # Build circle patches
    patches = []
    for idx in np.where(active_mask)[0]:
        x, y = positions[idx]
        patches.append(Circle((x, y), radii[idx]))
    return patches