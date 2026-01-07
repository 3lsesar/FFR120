import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================
N = 50  # Number of particles
R_mean = 0.06  # Mean particle radius (m)
m = 1.0  # Particle mass (kg)
g = 9.81  # Gravity (m/s^2)
dt = 0.0001  # Time step (s)
D = 3 * R_mean  # Funnel opening width (m) (orifice diameter)
substeps = 50  # Integration substeps per frame

# Contact force parameters
kn = 5000  # Normal stiffness (N/m)
kt = 4000   # Tangential stiffness (N/m)
gamma_n = 5  # Normal damping coefficient
mu = 0.6  # Friction coefficient
restitution = 0.5  # Wall restitution coefficient

# Brownian motion parameters
noise_amplitude = 1.0  # Amplitude of noise force (N)

# Jamming detection parameters
t_jam = 3.0  # Time threshold for jamming (seconds)

# GLOBAL STATE VARIABLES
x = np.zeros(N)
y = np.zeros(N)
vx = np.zeros(N)
vy = np.zeros(N)
R = np.zeros(N)  # Individual particle radius
active = np.ones(N, dtype=bool)

# Tracking variables
contact_xi = {}
time_since_exit = 0.0
total_time = 0.0
particles_exited = 0
is_jammed = False

# Funnel walls: each wall is (x1, y1, x2, y2)
walls = [
    (-0.5, 0.5, -0.5*D, 0),
    (0.5, 0.5, 0.5*D, 0),
]

# HELPER FUNCTIONS
def wall_x(x1, y1, x2, y2, y):
    """Calculate x position on wall at given y"""
    if y2 - y1 == 0:
        return x1
    return x1 + (x2 - x1) * (y - y1) / (y2 - y1)

def initialize_particles_random():
    """Randomly initialize particles above the funnel"""
    global x, y, vx, vy, R, active, N
    
    # Assign radii
    if enable_size_noise:
        R[:] = R_mean * (1 + 0.1 * (np.random.rand(N) - 0.5))
    else:
        R[:] = R_mean
    
    # Get funnel boundaries
    y_max = 0.48
    y_min = 0.05
    
    placed = 0
    max_attempts = 10000
    attempts = 0
    
    while placed < N and attempts < max_attempts:
        attempts += 1
        
        yi = np.random.uniform(y_min, y_max)
        x_left = wall_x(-0.5, 0.5, -0.5*D, 0, yi)
        x_right = wall_x(0.5, 0.5, 0.5*D, 0, yi)
        
        margin = R[placed] * 1.1
        x_left += margin
        x_right -= margin
        
        if x_left >= x_right:
            continue
        
        xi = np.random.uniform(x_left, x_right)
        
        # Check overlap with existing particles
        overlap = False
        for j in range(placed):
            dist = np.sqrt((xi - x[j])**2 + (yi - y[j])**2)
            if dist < (R[placed] + R[j]) * 1.1:
                overlap = True
                break
        
        if not overlap:
            x[placed] = xi
            y[placed] = yi
            vx[placed] = (np.random.random() - 0.5) * 0.05
            vy[placed] = 0.0
            placed += 1
    
    if placed < N:
        print(f"Warning: Only placed {placed}/{N} particles")

def point_to_segment_distance(px, py, x1, y1, x2, y2):
    """Calculate distance and closest point from point to line segment"""
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return np.sqrt((px - x1)**2 + (py - y1)**2), x1, y1
    
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx**2 + dy**2)))
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    dist = np.sqrt((px - closest_x)**2 + (py - closest_y)**2)
    
    return dist, closest_x, closest_y

def compute_forces():
    """Compute all forces on all particles"""
    fx = np.zeros(N)
    fy = np.zeros(N)
    
    # Gravity
    fy[active] = -m * g

    # Brownian motion
    if enable_noise:
        fx[active] += noise_amplitude * np.random.randn(np.sum(active))
        fy[active] += noise_amplitude * np.random.randn(np.sum(active))
    
    # Particle-particle contact forces
    active_indices = np.where(active)[0]
    for idx_i, i in enumerate(active_indices):
        for j in active_indices[idx_i + 1:]:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx**2 + dy**2)
            delta = R[i] + R[j] - dist
            
            if delta > 0 and dist > 1e-12:
                n_x = dx / dist
                n_y = dy / dist
                
                dvx = vx[i] - vx[j]
                dvy = vy[i] - vy[j]
                v_n = dvx * n_x + dvy * n_y
                
                F_n_mag = kn * delta - gamma_n * v_n
                F_n_x = F_n_mag * n_x
                F_n_y = F_n_mag * n_y
                
                t_x = -n_y
                t_y = n_x
                v_t = dvx * t_x + dvy * t_y
                
                contact_key = (min(i, j), max(i, j))
                if contact_key not in contact_xi:
                    contact_xi[contact_key] = 0.0
                
                contact_xi[contact_key] += v_t * dt
                xi = contact_xi[contact_key]
                
                F_t_elastic = kt * abs(xi)
                F_t_coulomb = mu * abs(F_n_mag)
                F_t_mag = -min(F_t_elastic, F_t_coulomb) * np.sign(xi)
                F_t_x = F_t_mag * t_x
                F_t_y = F_t_mag * t_y
                
                fx[i] += F_n_x + F_t_x
                fy[i] += F_n_y + F_t_y
                fx[j] -= F_n_x + F_t_x
                fy[j] -= F_n_y + F_t_y
            else:
                contact_key = (min(i, j), max(i, j))
                if contact_key in contact_xi:
                    del contact_xi[contact_key]
    
    # Particle-wall contact forces
    for i in active_indices:
        for wall_idx, wall in enumerate(walls):
            x1, y1, x2, y2 = wall
            dist, cx, cy = point_to_segment_distance(x[i], y[i], x1, y1, x2, y2)
            delta = R[i] - dist
            
            if delta > 0:
                n_x = x[i] - cx
                n_y = y[i] - cy
                n_mag = np.sqrt(n_x**2 + n_y**2)
                
                if n_mag > 0:
                    n_x /= n_mag
                    n_y /= n_mag
                    
                    v_n = vx[i] * n_x + vy[i] * n_y
                    F_n_mag = kn * delta - gamma_n * v_n
                    F_n_x = F_n_mag * n_x
                    F_n_y = F_n_mag * n_y
                    
                    t_x = -n_y
                    t_y = n_x
                    v_t = vx[i] * t_x + vy[i] * t_y
                    
                    contact_key = ('wall', i, wall_idx)
                    if contact_key not in contact_xi:
                        contact_xi[contact_key] = 0.0
                    
                    contact_xi[contact_key] += v_t * dt
                    xi = contact_xi[contact_key]
                    
                    F_t_elastic = kt * abs(xi)
                    F_t_coulomb = mu * abs(F_n_mag)
                    F_t_mag = -min(F_t_elastic, F_t_coulomb) * np.sign(xi)
                    F_t_x = F_t_mag * t_x
                    F_t_y = F_t_mag * t_y
                    
                    fx[i] += F_n_x + F_t_x
                    fy[i] += F_n_y + F_t_y
    
    return fx, fy

def enforce_wall_constraints():
    for i in np.where(active)[0]:
        for wall in walls:
            x1, y1, x2, y2 = wall
            dist, cx, cy = point_to_segment_distance(x[i], y[i], x1, y1, x2, y2)

            if dist < R[i]:
                nx = x[i] - cx
                ny = y[i] - cy
                n_mag = np.hypot(nx, ny)
                if n_mag == 0:
                    continue
                nx /= n_mag
                ny /= n_mag

                penetration = R[i] - dist
                x[i] += nx * penetration
                y[i] += ny * penetration

                vn = vx[i] * nx + vy[i] * ny
                if vn < 0:
                    vx[i] -= (1 + restitution) * vn * nx
                    vy[i] -= (1 + restitution) * vn * ny

def enforce_particle_constraints():
    """Hard-sphere collision enforcement between particles"""
    active_indices = np.where(active)[0]
    
    for idx_i, i in enumerate(active_indices):
        for j in active_indices[idx_i + 1:]:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.hypot(dx, dy)
            overlap = R[i] + R[j] - dist
            if overlap > 0 and dist > 1e-12:
                nx = dx / dist
                ny = dy / dist

                x[i] += 0.5 * overlap * nx
                y[i] += 0.5 * overlap * ny
                x[j] -= 0.5 * overlap * nx
                y[j] -= 0.5 * overlap * ny

                vn = (vx[i]-vx[j])*nx + (vy[i]-vy[j])*ny
                if vn < 0:
                    delta_v = -(1+restitution)*vn
                    vx[i] += 0.5 * delta_v * nx
                    vy[i] += 0.5 * delta_v * ny
                    vx[j] -= 0.5 * delta_v * nx
                    vy[j] -= 0.5 * delta_v * ny

def update(frame):
    """Update simulation by one frame"""
    global x, y, vx, vy, active, contact_xi
    global time_since_exit, total_time, particles_exited, is_jammed
    
    # Perform multiple substeps for numerical stability
    for _ in range(substeps):
        fx, fy = compute_forces()
        
        vx[active] += (fx[active] / m) * dt
        vy[active] += (fy[active] / m) * dt
        
        x[active] += vx[active] * dt
        y[active] += vy[active] * dt

        enforce_wall_constraints()
        enforce_particle_constraints()
        
        total_time += dt
        time_since_exit += dt

    # Check for particles exiting
    newly_inactive = (y < -0.2) & active
    if np.any(newly_inactive):
        particles_exited += np.sum(newly_inactive)
        time_since_exit = 0.0
    
    active[newly_inactive] = False
    
    # Jamming detection
    if not is_jammed and time_since_exit > t_jam and np.sum(active) > 0:
        is_jammed = True
        if enable_animation:
            print(f"JAMMED at t={total_time:.2f}s ({particles_exited}/{N} exited)")
    
    # Clean up contact tracking
    keys_to_remove = []
    for key in contact_xi:
        if isinstance(key, tuple) and len(key) == 2:
            i, j = key
            if not active[i] or not active[j]:
                keys_to_remove.append(key)
        elif key[0] == 'wall':
            _, i, _ = key
            if not active[i]:
                keys_to_remove.append(key)
    
    for key in keys_to_remove:
        del contact_xi[key]
    
    # Update plot (only if animation enabled)
    if enable_animation:
        positions = np.column_stack([x[active], y[active]])
        scatter.set_offsets(positions)
        
        sizes = (R[active] * 800)**2
        scatter.set_sizes(sizes)
        
        if active.sum() > 0:
            speeds = np.sqrt(vx[active]**2 + vy[active]**2)
            max_speed = max(speeds.max(), 1.0)
            colors = plt.cm.plasma(speeds / max_speed)
            scatter.set_color(colors)
        
        status = "JAMMED" if is_jammed else "Running"
        title.set_text(f'Active: {active.sum()}/{N} | Exited: {particles_exited} | Time: {total_time:.2f}s | {status}')
        
        return scatter, title
    
    return [],

# BATCH MODE FUNCTIONS
def run_single_simulation(N_sim, R_sim, D_sim, mu_sim, size_noise_sim):
    """Run a single simulation without animation"""
    global N, R_mean, D, mu, enable_size_noise
    global x, y, vx, vy, R, active, contact_xi
    global time_since_exit, total_time, particles_exited, is_jammed, walls
    
    # Set parameters
    N = N_sim
    R_mean = R_sim
    D = D_sim
    mu = mu_sim
    enable_size_noise = size_noise_sim
    
    # Reset arrays
    x = np.zeros(N)
    y = np.zeros(N)
    vx = np.zeros(N)
    vy = np.zeros(N)
    R = np.zeros(N)
    active = np.ones(N, dtype=bool)
    contact_xi = {}
    
    # Reset tracking
    time_since_exit = 0.0
    total_time = 0.0
    particles_exited = 0
    is_jammed = False
    
    # Update walls
    walls = [
        (-0.5, 0.5, -0.5*D, 0),
        (0.5, 0.5, 0.5*D, 0),
    ]
    
    # Initialize
    initialize_particles_random()
    
    # Run simulation
    max_steps = 50000
    step = 0
    while (np.sum(active) > 0 and not is_jammed) and step < max_steps:
        update(None)
        step += 1
    
    return {
        'D': D,
        'mu': mu,
        'jammed': is_jammed,
        'time': total_time,
        'exited': particles_exited,
        'N': N
    }

def run_batch_simulations():
    """Run multiple simulations and collect statistics"""
    results = []
    
    print("\n" + "="*60)
    print("BATCH SIMULATION MODE")
    print("="*60)
    print(f"Brownian motion: {'ENABLED' if enable_noise else 'DISABLED'}")
    print(f"Size variation: {'ENABLED' if enable_size_noise else 'DISABLED'}")
    print("="*60)
    
    for D_val in batch_D_values:
        for mu_val in batch_mu_values:
            print(f"\nRunning {batch_n_sims} simulations with D={D_val:.3f}m, μ={mu_val:.2f}")
            
            for sim_idx in range(batch_n_sims):
                result = run_single_simulation(N, R_mean, D_val, mu_val, enable_size_noise)
                results.append(result)
                
                status = "JAMMED" if result['jammed'] else "Complete"
                print(f"  Sim {sim_idx+1:2d}: {status:8s} | t={result['time']:6.2f}s | Exited: {result['exited']:2d}/{result['N']}")
            
            jam_count = sum(1 for r in results[-batch_n_sims:] if r['jammed'])
            jam_prob = jam_count / batch_n_sims
            print(f"  → Jamming probability: {jam_prob:.1%}")
    
    return results

# =============================================================================
# MAIN EXECUTION

# TOGGLES
enable_animation = True  # Set to False for batch mode (multiple runs with statistics)
enable_noise = True      # Toggle Brownian motion on/off
enable_size_noise = False  # Toggle particle size variation

# Batch mode parameters (only used when enable_animation = False)
batch_n_sims = 10  # Number of simulations per parameter combination
batch_D_values = [0.06, 0.12, 0.18, 0.24]  # Orifice widths to test
batch_mu_values = [0.2, 0.4, 0.6]  # Friction coefficients to test

if __name__ == "__main__":
    
    if enable_animation:
        # ANIMATION MODE - Single run with visualization
        print("\n" + "="*60)
        print("ANIMATION MODE")
        print("="*60)
        print(f"Brownian motion: {'ENABLED' if enable_noise else 'DISABLED'}")
        print(f"Size variation: {'ENABLED' if enable_size_noise else 'DISABLED'}")
        print("Set enable_animation = False for batch mode")
        print("="*60 + "\n")
        
        initialize_particles_random()
        
        # Set up plot
        fig, ax = plt.subplots(figsize=(8, 10))
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.2, 1.3)
        ax.set_aspect('equal')
        ax.set_facecolor('#f0f0f0')

        # Draw walls
        for wall in walls:
            ax.plot([wall[0], wall[2]], [wall[1], wall[3]], 'k-', linewidth=3)

        # Create scatter plot
        scatter = ax.scatter([], [], s=(R_mean * 800)**2, c='blue', alpha=0.8, 
                        edgecolors='black', linewidths=0.5)
        title = ax.set_title('', fontsize=12, pad=10)

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True, alpha=0.3)

        # Run animation
        anim = animation.FuncAnimation(fig, update, frames=3000, interval=1, blit=True)

        plt.tight_layout()
        plt.show()
        
    else:
        # BATCH MODE - Multiple runs with statistics
        results = run_batch_simulations()

        # Save results to file
        with open("simulation_data.txt", "w") as f:
            f.write("results = " + repr(results) + "\n")
