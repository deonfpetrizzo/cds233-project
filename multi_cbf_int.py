import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# 1. Physics & Parameters (From Teammate's Code)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class QuadrotorParams:
    m: float = 1.0    # mass [kg]
    I: float = 0.01   # moment of inertia [kg·m²]
    r: float = 0.15   # half-span [m]
    g: float = 9.81   # gravity [m/s²]

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
# 2. CBF-QP & Controllers
# ─────────────────────────────────────────────────────────────────────────────
def get_nominal_acceleration(state, target_pos, kp=3.0, kd=2.5):
    x, y, _, vx, vy, _ = state
    tx, ty = target_pos
    ax_nom = kp * (tx - x) - kd * vx
    ay_nom = kp * (ty - y) - kd * vy
    return np.array([ax_nom, ay_nom])

def map_accel_to_thrusts(state, accel_safe, params):
    """ Converts safe 2D acceleration back into real rotor thrusts (u1, u2) """
    x, y, theta, vx, vy, theta_dot = state
    ax, ay = accel_safe
    
    T_total = params.m * np.sqrt(ax**2 + (ay + params.g)**2)
    theta_d = np.arctan2(-ax, ay + params.g)
    
    kp_th, kd_th = 20.0, 5.0
    torque = params.I * (kp_th * (theta_d - theta) - kd_th * theta_dot)
    
    u1 = (T_total + torque / params.r) / 2.0
    u2 = (T_total - torque / params.r) / 2.0
    return np.array([u1, u2])

def solve_cbf_qp(states, u_nom, D_s=0.6, alpha1=3.0, gamma=3.0):
    """ The Centralized Master QP for N agents """
    N = states.shape[0]
    u = cp.Variable(2 * N)
    u_nom_flat = u_nom.flatten()
    objective = cp.Minimize(cp.sum_squares(u - u_nom_flat))
    constraints = []
    
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
        # OSQP is fast but sometimes returns 'optimal_inaccurate' when stressed
        prob.solve(solver=cp.OSQP)
        if prob.status in ['optimal', 'optimal_inaccurate']:
            return u.value.reshape((N, 2))
        else:
            # If completely infeasible, BRAKE! Do not return u_nom.
            return -2.5 * states[:, 3:5] 
    except Exception:
        return -2.5 * states[:, 3:5] # Extreme braking if solver crashes

# ─────────────────────────────────────────────────────────────────────────────
# 3. Merged OOP Architecture
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class QuadrotorInstance:
    x0:         np.ndarray
    target:     np.ndarray      # <--- Added target coordinate!
    params:     QuadrotorParams = field(default_factory=QuadrotorParams)
    label:      str             = ""
    color:      Optional[str]   = None
    t:          Optional[np.ndarray] = field(default=None, repr=False)
    X:          Optional[np.ndarray] = field(default=None, repr=False)

class MultiQuadrotorSim:
    _DEFAULT_PALETTE = plt.cm.tab10(np.linspace(0, 0.9, 10))

    def __init__(self, instances: List[QuadrotorInstance]):
        self.instances = instances
        for i, inst in enumerate(self.instances):
            if inst.color is None:
                inst.color = self._DEFAULT_PALETTE[i % 10]
            if not inst.label:
                inst.label = f"Quad {i + 1}"

    def run_centralized(self, t_span=(0.0, 6.0), dt=0.02):
        """ 
        The core change! Solves for all drones simultaneously so they can see each other.
        """
        N = len(self.instances)
        time_grid = np.arange(t_span[0], t_span[1] + dt, dt)
        
        # Initialize histories
        for inst in self.instances:
            inst.t = time_grid
            inst.X = np.zeros((len(time_grid), 6))
            inst.X[0] = inst.x0
            
        print(f"Running Centralized CBF-QP for {N} agents...")
        
        for k in range(1, len(time_grid)):
            t_start = time_grid[k-1]
            t_end = time_grid[k]
            
            # 1. Gather current states
            current_states = np.array([inst.X[k-1] for inst in self.instances])
            
            # 2. Get Nominal Control
            u_nom = np.zeros((N, 2))
            for i, inst in enumerate(self.instances):
                u_nom[i] = get_nominal_acceleration(current_states[i], inst.target)
                
            # 3. Filter through CBF-QP
            # Change this line inside run_centralized()
            u_safe = solve_cbf_qp(current_states, u_nom, D_s=0.5)
            
            # 4. Integrate physics using scipy.solve_ivp for accuracy
            for i, inst in enumerate(self.instances):
                # Map safe acceleration to actual rotor thrusts
                u_thrusts = map_accel_to_thrusts(current_states[i], u_safe[i], inst.params)
                
                # Dynamics wrapper for solve_ivp
                def dyn(t, x):
                    return f(x, inst.params) + g_matrix(x, inst.params) @ u_thrusts
                
                sol = solve_ivp(dyn, [t_start, t_end], current_states[i], method="RK45")
                inst.X[k] = sol.y[:, -1]
                
        print("Simulation Complete!")
        return self

    # ─────────────────────────────────────────────────────────────────────────────
    # Teammate's Animation Code (Slightly modified to show targets & bubbles)
    # ─────────────────────────────────────────────────────────────────────────────
    def animate(self, figsize=(10, 7), interval=30, trail_len=80, arm_scale=1.0):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect('equal')
        ax.set_xlabel('$x$ [m]')
        ax.set_ylabel('$y$ [m]')
        ax.set_title('Multi-Agent CBF: $N$-Drone Intersection Dodge')
        ax.grid(True, alpha=0.25)
        
        # Set static axes limits
        all_x = np.concatenate([inst.X[:, 0] for inst in self.instances])
        all_y = np.concatenate([inst.X[:, 1] for inst in self.instances])
        pad = 1.0
        ax.set_xlim(all_x.min() - pad, all_x.max() + pad)
        ax.set_ylim(all_y.min() - pad, all_y.max() + pad)

        time_text = ax.text(0.02, 0.96, '', transform=ax.transAxes, fontsize=10, va='top')

        vehicle_artists = []
        body_w, body_h, rotor_r = 0.12 * arm_scale, 0.03 * arm_scale, 0.05 * arm_scale

        for inst in self.instances:
            c = inst.color
            ax.plot(inst.target[0], inst.target[1], '*', color=c, markersize=10, alpha=0.5) # Draw target
            
            trail, = ax.plot([], [], '-', color=c, lw=1.2, alpha=0.55)
            bubble = plt.Circle((0,0), 0.35, color=c, fill=False, linestyle='--', alpha=0.5) # D_s/2
            ax.add_patch(bubble)
            
            body = mpatches.FancyBboxPatch((-body_w, -body_h), 2*body_w, 2*body_h, boxstyle="round,pad=0.01", linewidth=1.2, edgecolor=c, facecolor=(*plt.cm.colors.to_rgb(c), 0.35))
            ax.add_patch(body)
            r_left = mpatches.Circle((0, 0), rotor_r, color=c, alpha=0.55, zorder=4)
            r_right = mpatches.Circle((0, 0), rotor_r, color=c, alpha=0.55, zorder=4)
            ax.add_patch(r_left)
            ax.add_patch(r_right)
            txt = ax.text(0, 0, inst.label, fontsize=7, ha='center', va='bottom', color=c, fontweight='bold', zorder=6)

            vehicle_artists.append(dict(trail=trail, body=body, r_left=r_left, r_right=r_right, txt=txt, bubble=bubble))

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
                arts['txt'].set_position((px, py + 0.12 * arm_scale))

            return [a for arts in vehicle_artists for a in arts.values()] + [time_text]

        anim = FuncAnimation(fig, _update, frames=len(self.instances[0].t), interval=20, blit=False)
        plt.tight_layout()
        return anim

# ─────────────────────────────────────────────────────────────────────────────
# 4. Scenario: N=4 Cross Intersection
# ─────────────────────────────────────────────────────────────────────────────
def run_four_drone_scenario():
    instances = []
    
    # Drone 0: Left going Right (Top Lane)
    instances.append(QuadrotorInstance(
        x0=np.array([-3.0, 2.1, 0.0, 0.0, 0.0, 0.0]),
        target=np.array([ 3.0, 2.1]), label="Q0 (L->R)", color='blue'))
        
    # Drone 1: Right going Left (Bottom Lane)
    instances.append(QuadrotorInstance(
        x0=np.array([ 3.0, 1.9, 0.0, 0.0, 0.0, 0.0]),
        target=np.array([-3.0, 1.9]), label="Q1 (R->L)", color='red'))
        
    # Drone 2: Bottom going Top (Right Lane)
    instances.append(QuadrotorInstance(
        x0=np.array([ 0.1, -1.0, 0.0, 0.0, 0.0, 0.0]),
        target=np.array([ 0.1,  5.0]), label="Q2 (B->T)", color='green'))
        
    # Drone 3: Top going Bottom (Left Lane)
    instances.append(QuadrotorInstance(
        x0=np.array([-0.1,  5.0, 0.0, 0.0, 0.0, 0.0]),
        target=np.array([-0.1, -1.0]), label="Q3 (T->B)", color='orange'))

    sim = MultiQuadrotorSim(instances)
    
    # D_s = 0.5 gives them a bit more room to squeeze through the 4-way intersection
    # To pass D_s properly, update the loop in run_centralized:
    # u_safe = solve_cbf_qp(current_states, u_nom, D_s=0.5) 
    
    sim.run_centralized(t_span=(0.0, 6.0), dt=0.02)
    anim = sim.animate(arm_scale=1.5, trail_len=50)
    plt.show()

if __name__ == "__main__":
    run_four_drone_scenario()