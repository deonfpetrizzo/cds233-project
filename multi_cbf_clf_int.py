import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
from dataclasses import dataclass, field
from typing import List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# 1. Physics & Parameters
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
# 2. CLF design for the planning model
# ─────────────────────────────────────────────────────────────────────────────
# The nominal controller and the CBF filter both act in ACCELERATION space: the
# planning model is a 2D double integrator  p_ddot = a,  and map_accel_to_thrusts
# below realizes the commanded acceleration through the inner attitude loop.
#
# We want to regulate the output  y = p - p_des  to zero. For a double integrator
# y has RELATIVE DEGREE 2, so any V that depends on position ONLY gives L_g V = 0
# and is useless in an acceleration-level QP. Following Lecture 24/26, we instead
# build the CLF on the OUTPUT DYNAMICS, i.e. on (output, output-derivative):
#
#     eta = [ p - p_des ,  v ]      eta_dot = A eta + B a
#     A = [[0, 1],[0, 0]]           B = [0, 1]^T          (per axis; x,y decouple)
#
# V(eta) = eta^T P eta, with P solving the CARE  A^T P + P A - P B R^-1 B^T P + Q = 0.
# This is a valid (RES-)CLF (the robotic-systems analogue of Prop. 26.1): along the
# LQR feedback, V_dot = -eta^T Q eta <= -(lambda_min(Q)/lambda_max(P)) V, so the CLF
# decrease condition is always feasible.
_A_DI = np.array([[0.0, 1.0], [0.0, 0.0]])
_B_DI = np.array([[0.0], [1.0]])

def _build_clf(q_pos: float = 1.0, q_vel: float = 2.0, r_ctrl: float = 1.0):
    """Solve the per-axis CARE once and cache the matrices used by the CLF-QP."""
    Q = np.diag([q_pos, q_vel])
    R = np.array([[r_ctrl]])
    P = solve_continuous_are(_A_DI, _B_DI, Q, R)
    M = _A_DI.T @ P + P @ _A_DI         # appears in  L_f V = eta^T M eta
    PB = (P @ _B_DI).flatten()          # appears in  L_g V = 2 eta^T (P B)
    lam_guaranteed = float(np.min(np.linalg.eigvalsh(Q)) / np.max(np.linalg.eigvalsh(P)))
    return P, M, PB, lam_guaranteed

_P, _M, _PB, _LAM_GUARANTEED = _build_clf()

def solve_clf_qp(state, target_pos, clf_rate=1.8, p_relax=1.0e3, a_max=20.0):
    """
    CLF-QP nominal controller (replaces the PID). Regulates (x, y) -> (x_des, y_des)
    with v -> 0, as a SOFT constraint (relaxation delta), so the downstream CBF-QP
    obstacle-avoidance filter can freely override it when safety requires.

        min_{a, delta}   1/2 ||a||^2  +  1/2 * p_relax * delta^2
        s.t.   L_f V + L_g V a  <=  -clf_rate * V  +  delta

    With a single CLF constraint the relaxed QP has a closed-form (KKT) solution.
    Writing  phi0 = L_f V + clf_rate * V   and   phi1 = (L_g V)^T:

        a = -mu * phi1,   mu = max(0, phi0) / (||phi1||^2 + 1/p_relax)

    i.e. the pointwise min-norm acceleration that enforces the desired decrease
    rate, relaxed by 1/p_relax. The relaxation also tames the transient blow-up of
    the min-norm law where L_g V momentarily vanishes (loss of control authority).
    `a_max` is an optional saturation modeling the quadrotor's finite tilt/thrust
    authority. Returns a 2D acceleration command [ax, ay].
    """
    x, y, _, vx, vy, _ = state
    tx, ty = target_pos

    eta_x = np.array([x - tx, vx])
    eta_y = np.array([y - ty, vy])

    V   = eta_x @ _P @ eta_x + eta_y @ _P @ eta_y          # V(eta) = eta^T P eta
    LfV = eta_x @ _M @ eta_x + eta_y @ _M @ eta_y          # L_f V
    LgV = np.array([2.0 * eta_x @ _PB, 2.0 * eta_y @ _PB]) # L_g V  (grad wrt a)

    phi0 = LfV + clf_rate * V
    mu = max(0.0, phi0) / (LgV @ LgV + 1.0 / p_relax)
    a = -mu * LgV

    if a_max is not None:
        n = np.linalg.norm(a)
        if n > a_max:
            a *= a_max / n
    return a

# ─────────────────────────────────────────────────────────────────────────────
# 3. Acceleration -> rotor thrusts, and the centralized CBF-QP safety filter
# ─────────────────────────────────────────────────────────────────────────────
def map_accel_to_thrusts(state, accel_safe, params):
    """Converts a safe 2D acceleration back into real rotor thrusts (u1, u2)."""
    x, y, theta, vx, vy, theta_dot = state
    ax, ay = accel_safe

    T_total = params.m * np.sqrt(ax**2 + (ay + params.g)**2)
    theta_d = np.arctan2(-ax, ay + params.g)

    kp_th, kd_th = 20.0, 5.0
    torque = params.I * (kp_th * (theta_d - theta) - kd_th * theta_dot)

    u1 = (T_total + torque / params.r) / 2.0
    u2 = (T_total - torque / params.r) / 2.0
    return np.array([u1, u2])

def solve_cbf_qp(states, u_nom, D_s=0.5, alpha1=3.0, gamma=3.0):
    """The centralized master QP that filters the CLF accelerations for N agents."""
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
        prob.solve(solver=cp.OSQP)
        if prob.status in ['optimal', 'optimal_inaccurate']:
            return u.value.reshape((N, 2))
        else:
            return -2.5 * states[:, 3:5]   # infeasible -> brake
    except Exception:
        return -2.5 * states[:, 3:5]       # solver crash -> brake

# ─────────────────────────────────────────────────────────────────────────────
# 4. OOP Architecture
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class QuadrotorInstance:
    x0:         np.ndarray
    target:     np.ndarray
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

    def run_centralized(self, t_span=(0.0, 6.0), dt=0.02,
                        clf_rate=1.8, p_relax=1.0e3, a_max=20.0, D_s=0.5):
        """
        Layered controller:
          (1) per-agent CLF-QP  -> nominal acceleration regulating to the target,
          (2) centralized CBF-QP -> filters those accelerations for collision safety.
        """
        N = len(self.instances)
        time_grid = np.arange(t_span[0], t_span[1] + dt, dt)

        for inst in self.instances:
            inst.t = time_grid
            inst.X = np.zeros((len(time_grid), 6))
            inst.X[0] = inst.x0

        print(f"Running CLF-QP (nominal) + CBF-QP (safety) for {N} agents...")
        print(f"  guaranteed CLF rate lambda_min(Q)/lambda_max(P) = {_LAM_GUARANTEED:.3f}, "
              f"using clf_rate = {clf_rate}")

        for k in range(1, len(time_grid)):
            t_start, t_end = time_grid[k-1], time_grid[k]

            # 1. Gather current states
            current_states = np.array([inst.X[k-1] for inst in self.instances])

            # 2. Nominal control from the CLF-QP (replaces the old PID)
            u_nom = np.zeros((N, 2))
            for i, inst in enumerate(self.instances):
                u_nom[i] = solve_clf_qp(current_states[i], inst.target,
                                        clf_rate=clf_rate, p_relax=p_relax, a_max=a_max)

            # 3. Filter through the CBF-QP safety filter
            u_safe = solve_cbf_qp(current_states, u_nom, D_s=D_s)

            # 4. Integrate physics
            for i, inst in enumerate(self.instances):
                u_thrusts = map_accel_to_thrusts(current_states[i], u_safe[i], inst.params)

                def dyn(t, x):
                    return f(x, inst.params) + g_matrix(x, inst.params) @ u_thrusts

                sol = solve_ivp(dyn, [t_start, t_end], current_states[i], method="RK45")
                inst.X[k] = sol.y[:, -1]

        print("Simulation Complete!")
        return self

    # ─────────────────────────────────────────────────────────────────────────
    # Animation (shows targets & safety bubbles)
    # ─────────────────────────────────────────────────────────────────────────
    def animate(self, figsize=(10, 7), interval=30, trail_len=80, arm_scale=1.0):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect('equal')
        ax.set_xlabel('$x$ [m]')
        ax.set_ylabel('$y$ [m]')
        ax.set_title('CLF-QP + CBF: $N$-Drone Intersection Dodge')
        ax.grid(True, alpha=0.25)

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
            ax.plot(inst.target[0], inst.target[1], '*', color=c, markersize=10, alpha=0.5)

            trail, = ax.plot([], [], '-', color=c, lw=1.2, alpha=0.55)
            bubble = plt.Circle((0, 0), 0.35, color=c, fill=False, linestyle='--', alpha=0.5)
            ax.add_patch(bubble)

            body = mpatches.FancyBboxPatch((-body_w, -body_h), 2*body_w, 2*body_h,
                                           boxstyle="round,pad=0.01", linewidth=1.2,
                                           edgecolor=c, facecolor=(*plt.cm.colors.to_rgb(c), 0.35))
            ax.add_patch(body)
            r_left = mpatches.Circle((0, 0), rotor_r, color=c, alpha=0.55, zorder=4)
            r_right = mpatches.Circle((0, 0), rotor_r, color=c, alpha=0.55, zorder=4)
            ax.add_patch(r_left)
            ax.add_patch(r_right)
            txt = ax.text(0, 0, inst.label, fontsize=7, ha='center', va='bottom',
                          color=c, fontweight='bold', zorder=6)

            vehicle_artists.append(dict(trail=trail, body=body, r_left=r_left,
                                        r_right=r_right, txt=txt, bubble=bubble))

        def _update(frame):
            time_text.set_text(f't = {self.instances[0].t[frame]:.2f} s')
            for inst, arts in zip(self.instances, vehicle_artists):
                X = inst.X
                px, py, theta = X[frame, 0], X[frame, 1], X[frame, 2]

                start = max(0, frame - trail_len)
                arts['trail'].set_data(X[start:frame + 1, 0], X[start:frame + 1, 1])
                arts['bubble'].center = (px, py)
                arts['body'].set_transform(
                    mtransforms.Affine2D().rotate_around(0, 0, theta).translate(px, py) + ax.transData)

                r = inst.params.r * arm_scale
                lx, ly = px - r * np.cos(theta), py - r * np.sin(theta)
                rx, ry = px + r * np.cos(theta), py + r * np.sin(theta)
                arts['r_left'].set_center((lx, ly))
                arts['r_right'].set_center((rx, ry))
                arts['txt'].set_position((px, py + 0.12 * arm_scale))

            return [a for arts in vehicle_artists for a in arts.values()] + [time_text]

        anim = FuncAnimation(fig, _update, frames=len(self.instances[0].t),
                             interval=20, blit=False)
        plt.tight_layout()
        return anim

# ─────────────────────────────────────────────────────────────────────────────
# 5. Scenario: N=4 Cross Intersection
# ─────────────────────────────────────────────────────────────────────────────
def run_four_drone_scenario():
    instances = [
        QuadrotorInstance(x0=np.array([-3.0, 2.1, 0.0, 0.0, 0.0, 0.0]),
                          target=np.array([ 3.0, 2.1]), label="Q0 (L->R)", color='blue'),
        QuadrotorInstance(x0=np.array([ 3.0, 1.9, 0.0, 0.0, 0.0, 0.0]),
                          target=np.array([-3.0, 1.9]), label="Q1 (R->L)", color='red'),
        QuadrotorInstance(x0=np.array([ 0.1, -1.0, 0.0, 0.0, 0.0, 0.0]),
                          target=np.array([ 0.1,  5.0]), label="Q2 (B->T)", color='green'),
        QuadrotorInstance(x0=np.array([-0.1,  5.0, 0.0, 0.0, 0.0, 0.0]),
                          target=np.array([-0.1, -1.0]), label="Q3 (T->B)", color='orange'),
    ]

    sim = MultiQuadrotorSim(instances)
    sim.run_centralized(t_span=(0.0, 6.0), dt=0.02,
                        clf_rate=1.8, p_relax=1.0e3, a_max=20.0, D_s=0.5)
    anim = sim.animate(arm_scale=1.5, trail_len=50)
    plt.show()

if __name__ == "__main__":
    run_four_drone_scenario()