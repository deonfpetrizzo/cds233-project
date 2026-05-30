import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from typing import List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# 1. Physics & Parameters 
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class QuadrotorParams:
    m: float = 1.0    
    I: float = 0.01   
    r: float = 0.15   
    g: float = 9.81   

def f(x: np.ndarray, params: QuadrotorParams) -> np.ndarray:
    _, _, _, xd, yd, thd = x
    return np.array([xd, yd, thd, 0.0, -params.g, 0.0])

def g_matrix(x: np.ndarray, params: QuadrotorParams) -> np.ndarray:
    _, _, theta, _, _, _ = x
    s, c = np.sin(theta), np.cos(theta)
    m, I, r = params.m, params.I, params.r
    G = np.zeros((6, 2))
    G[3, :] = [-s / m, -s / m]
    G[4, :] = [ c / m,  c / m]
    G[5, :] = [ r / I, -r / I]
    return G

# ─────────────────────────────────────────────────────────────────────────────
# 2. CBF-QP & Controllers (With High-Gain Attitude Fix!)
# ─────────────────────────────────────────────────────────────────────────────
def get_nominal_acceleration(state, target_pos, kp=3.0, kd=2.5):
    x, y, _, vx, vy, _ = state
    tx, ty = target_pos
    ax_nom = kp * (tx - x) - kd * vx
    ay_nom = kp * (ty - y) - kd * vy
    return np.array([ax_nom, ay_nom])

def map_accel_to_thrusts(state, accel_safe, params):
    x, y, theta, vx, vy, theta_dot = state
    ax, ay = accel_safe
    
    T_total = params.m * np.sqrt(ax**2 + (ay + params.g)**2)
    theta_d = np.arctan2(-ax, ay + params.g)
    
    # HIGH GAINS so the physical drone acts instantly like the math assumes
    kp_th, kd_th = 100.0, 15.0
    torque = params.I * (kp_th * (theta_d - theta) - kd_th * theta_dot)
    
    u1 = (T_total + torque / params.r) / 2.0
    u2 = (T_total - torque / params.r) / 2.0
    return np.array([u1, u2])

_KD_BRAKE = 2.5 

def solve_cbf_qp(states, u_nom, D_s=0.6, alpha1=3.0, gamma=3.0):
    N = states.shape[0]
    u = cp.Variable(2 * N)
    u_nom_flat = u_nom.flatten()
    objective = cp.Minimize(cp.sum_squares(u - u_nom_flat))
    constraints = []

    # N(N-1)/2 Pairwise Constraints
    for i in range(N):
        for j in range(i + 1, N):
            p_i, v_i = states[i, 0:2], states[i, 3:5]
            p_j, v_j = states[j, 0:2], states[j, 3:5]

            dp = p_i - p_j
            dv = v_i - v_j

            A_i = 2 * dp
            A_j = -2 * dp

            RHS = -2 * np.linalg.norm(dv)**2 \
                  - (gamma + alpha1) * (2 * np.dot(dp, dv)) \
                  - alpha1 * gamma * (np.linalg.norm(dp)**2 - D_s**2)

            constraints.append(A_i @ u[2*i : 2*i+2] + A_j @ u[2*j : 2*j+2] >= RHS)

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.OSQP)
        # Accept 'optimal_inaccurate' to prevent ghosting
        if prob.status in ['optimal', 'optimal_inaccurate']:
            return u.value.reshape((N, 2))
        else:
            return -_KD_BRAKE * states[:, 3:5]
    except Exception:
        return -_KD_BRAKE * states[:, 3:5]

# ─────────────────────────────────────────────────────────────────────────────
# 3. Merged OOP Architecture
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class QuadrotorInstance:
    x0:         np.ndarray
    target:     np.ndarray      
    params:     QuadrotorParams = field(default_factory=QuadrotorParams)
    color:      Optional[str]   = None
    t:          Optional[np.ndarray] = field(default=None, repr=False)
    X:          Optional[np.ndarray] = field(default=None, repr=False)

class MultiQuadrotorSim:
    def __init__(self, instances: List[QuadrotorInstance]):
        self.instances = instances

    def run_centralized(self, t_span=(0.0, 8.0), dt=0.04):
        N = len(self.instances)
        time_grid = np.arange(t_span[0], t_span[1] + dt, dt)
        
        for inst in self.instances:
            inst.t = time_grid
            inst.X = np.zeros((len(time_grid), 6))
            inst.X[0] = inst.x0
            
        print(f"--- Starting Drone Show ---")
        print(f"Agents: {N} | Active Collision Constraints: {int(N*(N-1)/2)}")
        
        for k in range(1, len(time_grid)):
            t_start = time_grid[k-1]
            t_end = time_grid[k]
            
            # Print progress bar because 300 constraints takes a moment
            if k % 25 == 0:
                print(f"Simulating time {t_end:.2f}s / {t_span[1]:.2f}s ...")
            
            current_states = np.array([inst.X[k-1] for inst in self.instances])
            
            u_nom = np.zeros((N, 2))
            for i, inst in enumerate(self.instances):
                u_nom[i] = get_nominal_acceleration(current_states[i], inst.target)
                
            # D_s=0.6 for mathematical barrier
            u_safe = solve_cbf_qp(current_states, u_nom, D_s=0.6)
            
            for i, inst in enumerate(self.instances):
                def dyn(t, x, accel=u_safe[i], p=inst.params):
                    u_th = map_accel_to_thrusts(x, accel, p)
                    return f(x, p) + g_matrix(x, p) @ u_th

                sol = solve_ivp(dyn, [t_start, t_end], current_states[i], method="RK45")
                inst.X[k] = sol.y[:, -1]
                
        print("Simulation Complete!")
        return self

    def animate(self, figsize=(10, 10), interval=40, trail_len=25, arm_scale=1.0):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect('equal')
        ax.set_xlabel('$x$ [m]')
        ax.set_ylabel('$y$ [m]')
        ax.set_title('Drone Show: 25-Agent Swarm Reconfiguration')
        
        # Dark background makes it look like a night show!
        ax.set_facecolor('#111111')
        fig.patch.set_facecolor('#111111')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.1)
        
        all_x = np.concatenate([inst.X[:, 0] for inst in self.instances])
        all_y = np.concatenate([inst.X[:, 1] for inst in self.instances])
        pad = 1.0
        ax.set_xlim(all_x.min() - pad, all_x.max() + pad)
        ax.set_ylim(all_y.min() - pad, all_y.max() + pad)

        time_text = ax.text(0.02, 0.96, '', transform=ax.transAxes, fontsize=12, va='top', color='white')

        vehicle_artists = []
        body_w, body_h, rotor_r = 0.12 * arm_scale, 0.03 * arm_scale, 0.05 * arm_scale

        for inst in self.instances:
            c = inst.color
            # Draw faded target pixel
            ax.plot(inst.target[0], inst.target[1], 's', color=c, markersize=12, alpha=0.2) 
            
            trail, = ax.plot([], [], '-', color=c, lw=1.5, alpha=0.6)
            
            # Bubble radius is D_s / 2 = 0.3
            bubble = plt.Circle((0,0), 0.3, color=c, fill=False, linestyle='--', alpha=0.3) 
            ax.add_patch(bubble)
            
            body = mpatches.FancyBboxPatch((-body_w, -body_h), 2*body_w, 2*body_h, boxstyle="round,pad=0.01", linewidth=1.2, edgecolor='white', facecolor=c)
            ax.add_patch(body)
            r_left = mpatches.Circle((0, 0), rotor_r, color='white', alpha=0.8, zorder=4)
            r_right = mpatches.Circle((0, 0), rotor_r, color='white', alpha=0.8, zorder=4)
            ax.add_patch(r_left)
            ax.add_patch(r_right)

            vehicle_artists.append(dict(trail=trail, body=body, r_left=r_left, r_right=r_right, bubble=bubble))

        def _update(frame):
            time_text.set_text(f't = {self.instances[0].t[frame]:.2f} s')
            for vi, (inst, arts) in enumerate(zip(self.instances, vehicle_artists)):
                X = inst.X
                px, py, theta = X[frame, 0], X[frame, 1], X[frame, 2]
                
                start = max(0, frame - trail_len)
                arts['trail'].set_data(X[start:frame + 1, 0], X[start:frame + 1, 1])
                arts['bubble'].center = (px, py)
                arts['body'].set_transform(mtransforms.Affine2D().rotate_around(0, 0, theta).translate(px, py) + ax.transData)
                
                r = inst.params.r * arm_scale
                lx, ly = px - r * np.cos(theta), py - r * np.sin(theta)
                rx, ry = px + r * np.cos(theta), py + r * np.sin(theta)
                arts['r_left'].set_center((lx, ly))
                arts['r_right'].set_center((rx, ry))

            return [a for arts in vehicle_artists for a in arts.values()] + [time_text]

        anim = FuncAnimation(fig, _update, frames=len(self.instances[0].t), interval=40, blit=False)
        plt.tight_layout()
        return anim

# ─────────────────────────────────────────────────────────────────────────────
# 4. Scenario: The Drone Show (Caltech "C")
# ─────────────────────────────────────────────────────────────────────────────
def run_drone_show_scenario():
    # 5x5 Pixel Art for Caltech 'C'
    # O = Orange, G = Dark Gray
    caltech_c = [
        ['G', 'O', 'O', 'O', 'G'],
        ['O', 'G', 'G', 'G', 'G'],
        ['O', 'G', 'G', 'G', 'G'],
        ['O', 'G', 'G', 'G', 'G'],
        ['G', 'O', 'O', 'O', 'G']
    ]
    color_map = {'O': '#FF6F00', 'G': '#555555'} # Caltech Orange and Gray
    
    # Generate ordered targets in the sky (Y=5 to 9)
    targets = []
    colors = []
    for row in range(5):
        for col in range(5):
            tx = col - 2.0         # X: -2 to 2
            ty = 9.0 - row         # Y: 5 to 9 (top-down)
            targets.append(np.array([tx, ty]))
            colors.append(color_map[caltech_c[row][col]])
            
    # Generate a scrambled grid on the ground (Y=0 to 4)
    starts = []
    for row in range(5):
        for col in range(5):
            sx = col - 2.0
            sy = 4.0 - row
            starts.append(np.array([sx, sy]))
            
    # Shuffle the starting positions so they HAVE to cross paths!
    np.random.seed(42) 
    np.random.shuffle(starts)
    
    # Create the 25 quadrotor instances
    instances = []
    for i in range(25):
        x0 = np.array([starts[i][0], starts[i][1], 0.0, 0.0, 0.0, 0.0])
        instances.append(QuadrotorInstance(x0=x0, target=targets[i], color=colors[i]))

    sim = MultiQuadrotorSim(instances)
    
    # Run the simulation! (t_end=8.0 gives them time to settle into the image)
    sim.run_centralized(t_span=(0.0, 15.0), dt=0.04)
    anim = sim.animate(arm_scale=1.0, trail_len=15)
    plt.show()

if __name__ == "__main__":
    run_drone_show_scenario()