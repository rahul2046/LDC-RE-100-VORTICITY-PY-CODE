import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Parameters ---
N_POINTS = 129  # Ghia used 129x129 grid for Re=1000
DOMAIN_SIZE = 1.0
N_ITERATIONS = 30000  # Increased for better convergence
TIME_STEP_LENGTH = 0.0001  # Reduced for stability
HORIZONTAL_VELOCITY_TOP = 1.0
DENSITY = 1.0
Re = 100
KINEMATIC_VISCOSITY = 1.0 / Re
N_PRESSURE_POISSON_ITERATIONS = 50
STABILITY_SAFETY_FACTOR = 0.25

def simulate_lid_driven_cavity():
    dx = DOMAIN_SIZE / (N_POINTS - 1)
    x = np.linspace(0.0, DOMAIN_SIZE, N_POINTS)
    y = np.linspace(0.0, DOMAIN_SIZE, N_POINTS)
    X, Y = np.meshgrid(x, y)

    u = np.zeros_like(X)
    v = np.zeros_like(X)
    p = np.zeros_like(X)

    # Pre-compute coefficients
    dt_nu = TIME_STEP_LENGTH * KINEMATIC_VISCOSITY
    dt_rho = TIME_STEP_LENGTH / DENSITY

    def central_diff_x(f):
        diff = np.zeros_like(f)
        diff[1:-1, 1:-1] = (f[1:-1, 2:] - f[1:-1, :-2]) / (2 * dx)
        return diff

    def central_diff_y(f):
        diff = np.zeros_like(f)
        diff[1:-1, 1:-1] = (f[2:, 1:-1] - f[:-2, 1:-1]) / (2 * dx)
        return diff

    def laplacian(f):
        diff = np.zeros_like(f)
        diff[1:-1, 1:-1] = (
            f[1:-1, :-2] + f[:-2, 1:-1] + f[1:-1, 2:] + f[2:, 1:-1] - 4 * f[1:-1, 1:-1]
        ) / dx**2
        return diff

    # Stability check
    max_dt = 0.25 * dx**2 / KINEMATIC_VISCOSITY
    if TIME_STEP_LENGTH > STABILITY_SAFETY_FACTOR * max_dt:
        raise RuntimeError("Time step too large for stability.")

    # For convergence monitoring
    u_old = np.zeros_like(u)
    v_old = np.zeros_like(v)
    tolerance = 1e-6

    for it in tqdm(range(N_ITERATIONS), desc="Simulating"):
        u_old = u.copy()
        v_old = v.copy()

        # Intermediate velocity fields
        du_dx = central_diff_x(u)
        du_dy = central_diff_y(u)
        dv_dx = central_diff_x(v)
        dv_dy = central_diff_y(v)
        lap_u = laplacian(u)
        lap_v = laplacian(v)

        u_star = u + TIME_STEP_LENGTH * (-u * du_dx - v * du_dy) + dt_nu * lap_u
        v_star = v + TIME_STEP_LENGTH * (-u * dv_dx - v * dv_dy) + dt_nu * lap_v

        # Boundary conditions
        u_star[0, :] = 0.0       # Bottom wall
        u_star[:, 0] = 0.0        # Left wall
        u_star[:, -1] = 0.0       # Right wall
        u_star[-1, :] = HORIZONTAL_VELOCITY_TOP  # Top wall

        v_star[0, :] = 0.0        # Bottom wall
        v_star[:, 0] = 0.0        # Left wall
        v_star[:, -1] = 0.0       # Right wall
        v_star[-1, :] = 0.0       # Top wall

        # Pressure Poisson equation
        divergence = central_diff_x(u_star) + central_diff_y(v_star)
        rhs = divergence / TIME_STEP_LENGTH

        for _ in range(N_PRESSURE_POISSON_ITERATIONS):
            p_new = np.zeros_like(p)
            p_new[1:-1, 1:-1] = 0.25 * (
                p[1:-1, :-2] + p[:-2, 1:-1] + p[1:-1, 2:] + p[2:, 1:-1] - dx**2 * rhs[1:-1, 1:-1]
            )

            # Pressure BCs
            p_new[:, -1] = p_new[:, -2]  # Right
            p_new[0, :] = p_new[1, :]    # Bottom
            p_new[:, 0] = p_new[:, 1]    # Left
            p_new[-1, :] = p_new[-2, :]  # Top
            p_new[0, 0] = 0.0           # Reference pressure

            # Under-relaxation
            p = 0.7 * p + 0.3 * p_new

        # Velocity correction
        dp_dx = central_diff_x(p)
        dp_dy = central_diff_y(p)

        u = u_star - dt_rho * dp_dx
        v = v_star - dt_rho * dp_dy

        # Reapply velocity BCs
        u[0, :] = 0.0
        u[:, 0] = 0.0
        u[:, -1] = 0.0
        u[-1, :] = HORIZONTAL_VELOCITY_TOP
        v[0, :] = 0.0
        v[:, 0] = 0.0
        v[:, -1] = 0.0
        v[-1, :] = 0.0

        # Convergence check
        if it % 1000 == 0:
            du_max = np.max(np.abs(u - u_old))
            dv_max = np.max(np.abs(v - v_old))
            if du_max < tolerance and dv_max < tolerance:
                print(f"Converged after {it} iterations")
                break

    return u, v, p, X, Y, dx

def compute_vorticity(u, v, dx):
    dv_dx = np.zeros_like(v)
    du_dy = np.zeros_like(u)
    dv_dx[1:-1, 1:-1] = (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dx)
    du_dy[1:-1, 1:-1] = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dx)
    vorticity = dv_dx - du_dy

    # Boundary vorticity (important for accurate contours)
    # Top wall (moving lid)
    vorticity[-1, 1:-1] = (u[-1, 1:-1] - u[-2, 1:-1]) / dx - (v[-1, 2:] - v[-1, :-2]) / (2 * dx)
    # Bottom wall
    vorticity[0, 1:-1] = (u[1, 1:-1] - u[0, 1:-1]) / dx - (v[0, 2:] - v[0, :-2]) / (2 * dx)
    # Left wall
    vorticity[1:-1, 0] = (v[1:-1, 1] - v[1:-1, 0]) / dx - (u[2:, 0] - u[:-2, 0]) / (2 * dx)
    # Right wall
    vorticity[1:-1, -1] = (v[1:-1, -1] - v[1:-1, -2]) / dx - (u[2:, -1] - u[:-2, -1]) / (2 * dx)

    # Corner points (average of adjacent faces)
    vorticity[0, 0] = 0.5 * (vorticity[0, 1] + vorticity[1, 0])
    vorticity[0, -1] = 0.5 * (vorticity[0, -2] + vorticity[1, -1])
    vorticity[-1, 0] = 0.5 * (vorticity[-1, 1] + vorticity[-2, 0])
    vorticity[-1, -1] = 0.5 * (vorticity[-1, -2] + vorticity[-2, -1])

    return vorticity

def plot_vorticity(omega, X, Y):
    plt.figure(figsize=(8, 7))

    # Ghia's exact contour levels (-5 to 5 with 0.5 increments)
    levels = np.arange(-5, 5.1, 0.5)

    # Plotting with proper line styles
    cs_pos = plt.contour(X, Y, omega, levels=levels[levels > 0], colors='black', linewidths=1.0)
    cs_neg = plt.contour(X, Y, omega, levels=levels[levels < 0], colors='black', linewidths=1.0, linestyles='dashed')
    cs_zero = plt.contour(X, Y, omega, levels=[0], colors='black', linewidths=1.5)

    # Label every other contour for clarity
    plt.clabel(cs_pos, levels=levels[levels > 0][::2], inline=True, fontsize=8, fmt='%1.1f')
    plt.clabel(cs_neg, levels=levels[levels < 0][::2], inline=True, fontsize=8, fmt='%1.1f')

    plt.title('Vorticity Contours (Re=100) - Ghia et al. Comparison', pad=20)
    plt.xlabel('X/D')
    plt.ylabel('Y/D')
    plt.gca().set_aspect('equal')


    plt.grid(True, linestyle='--', linewidth=0.3, alpha=0.5)
    plt.tight_layout()
    plt.savefig('vorticity_Re1000.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Starting simulation...")
    u, v, p, X, Y, dx = simulate_lid_driven_cavity()
    print("Computing vorticity...")
    omega = compute_vorticity(u, v, dx)
    print("Plotting results...")
    plot_vorticity(omega, X, Y)
    print("Done! Results saved as vorticity_Re1000.png")
