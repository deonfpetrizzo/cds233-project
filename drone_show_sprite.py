import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
from PIL import Image
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import time

from matplotlib.animation import FFMpegWriter


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




def _build_lqr(q_pos=9.0, q_vel=6.0, r_ctrl=1.0):
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    Q = np.diag([q_pos, q_vel])
    R = np.array([[r_ctrl]])
    P = solve_continuous_are(A, B, Q, R)
    K = (np.linalg.solve(R, B.T @ P)).flatten() 
    return K

_K = _build_lqr()

def lqr_control(state, target_pos, a_max=None):
    """Per-agent LQR nominal: regulates (x,y)->(x_des,y_des), v->0. Returns [ax,ay]."""
    x, y, _, vx, vy, _ = state
    tx, ty = target_pos
    eta_x = np.array([x - tx, vx])
    eta_y = np.array([y - ty, vy])

    a = np.array([-_K @ eta_x, -_K @ eta_y])

    if a_max is not None:
        n = np.linalg.norm(a)
        if n > a_max:
            a *= a_max / n
    return a

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


class WarmStartCBF:
    """Warm-started, parameterized centralized CBF-QP. Build once per agent count;
    call each timestep with the current states and nominal accelerations."""
    def __init__(self, N, D_s=0.6, D_sense=2.0, alpha1=3.0, gamma=3.0, epsilon=None):
        self.N = N
        self.D_s = D_s
        self.D_sense = D_sense
        self.alpha1 = alpha1
        self.gamma = gamma
        self.epsilon = epsilon

        pairs = np.array([(i, j) for i in range(N) for j in range(i + 1, N)], dtype=int)
        self.I = pairs[:, 0]
        self.J = pairs[:, 1]
        self.npr = len(pairs)

        self.u    = cp.Variable((N, 2))
        self.un_p = cp.Parameter((N, 2))           
        self.dp_p = cp.Parameter((self.npr, 2))     
        self.rhs_p = cp.Parameter(self.npr)        

        cons = [self.dp_p[p] @ self.u[self.I[p]] - self.dp_p[p] @ self.u[self.J[p]]
                >= self.rhs_p[p] for p in range(self.npr)]
        self.prob = cp.Problem(cp.Minimize(cp.sum_squares(self.u - self.un_p)), cons)

    def __call__(self, states, u_nom):
        N = self.N
        Pp = states[:, 0:2]
        Vv = states[:, 3:5]

        dp = Pp[self.I] - Pp[self.J]                 
        dv = Vv[self.I] - Vv[self.J]
        dist2 = np.sum(dp * dp, axis=1)
        rhs = (-2.0 * np.sum(dv * dv, axis=1)
               - (self.gamma + self.alpha1) * (2.0 * np.sum(dp * dv, axis=1))
               - self.alpha1 * self.gamma * (dist2 - self.D_s**2))

        # ISSf robustness: tighten by (1/epsilon)||Lg h_e||^2 = (8/epsilon)||dp||^2
        if self.epsilon is not None:
            rhs = rhs + (8.0 / self.epsilon) * dist2

        # Radius gate: out-of-range pairs become inert (zero coeff, rhs = -1)
        active = dist2 <= self.D_sense**2
        self.dp_p.value  = np.where(active[:, None], 2.0 * dp, 0.0)
        self.rhs_p.value = np.where(active, rhs, -1.0)
        self.un_p.value  = u_nom

        try:
            self.prob.solve(solver=cp.OSQP, warm_start=True)
            if self.prob.status in ['optimal', 'optimal_inaccurate'] and self.u.value is not None:
                return self.u.value.copy()
            return -_KD_BRAKE * states[:, 3:5]
        except Exception:
            return -_KD_BRAKE * states[:, 3:5]


def solve_cbf_qp(states, u_nom, D_s=0.6, D_sense=2.0, alpha1=3.0, gamma=3.0, flag=True, epsilon=None):
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

            dist = np.linalg.norm(dp)
            if flag and dist > D_sense:
                continue

            A_i = 2 * dp
            A_j = -2 * dp

            RHS = -2 * np.linalg.norm(dv)**2 \
                  - (gamma + alpha1) * (2 * np.dot(dp, dv)) \
                  - alpha1 * gamma * (np.linalg.norm(dp)**2 - D_s**2)

            # ISSf robustness: tighten by (1/epsilon)||Lg h_e||^2 = (8/epsilon)||dp||^2
            if epsilon is not None:
                RHS += (8.0 / epsilon) * dist**2

            constraints.append(A_i @ u[2*i : 2*i+2] + A_j @ u[2*j : 2*j+2] >= RHS)

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.OSQP)
        if prob.status in ['optimal', 'optimal_inaccurate']:
            return u.value.reshape((N, 2))
        else:
            return -_KD_BRAKE * states[:, 3:5]
    except Exception:
        return -_KD_BRAKE * states[:, 3:5]




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

    def run_centralized(self, t_span=(0.0, 16.0), dt=0.04, a_max=None,
                        D_s=0.6, D_sense=2.0, warm_start=True, epsilon=None,
                        converge_pos_tol=0.05, converge_vel_tol=0.05, converge_hold=5):
        """Runs until either t_span[1] OR all agents converge to their targets.
        Convergence: every agent within `converge_pos_tol` (m) of its target with
        speed below `converge_vel_tol` (m/s), sustained for `converge_hold` steps
        (so a transient fly-through does not trigger it). Set converge_pos_tol=None
        to disable early termination and run the full horizon. Histories are trimmed
        to the actual stop time, so animate() sees only real frames."""
        N = len(self.instances)
        time_grid = np.arange(t_span[0], t_span[1] + dt, dt)
        n_steps = len(time_grid)

        for inst in self.instances:
            inst.t = time_grid
            inst.X = np.zeros((n_steps, 6))
            inst.X[0] = inst.x0

        targets = np.array([inst.target for inst in self.instances])

        print(f"--- Starting Drone Show ---")
        print(f"Agents: {N} | Active Collision Constraints: {int(N*(N-1)/2)}")
        print(f"LQR nominal: K={np.round(_K,3)} (kp={_K[0]:.2f}, kd={_K[1]:.2f})")
        print(f"CBF filter: {'warm-started (build once)' if warm_start else 'rebuild each step'}")

        # Build the warm-started QP once (cost amortized over the whole run)
        cbf = None
        if warm_start:
            t_build = time.time()
            cbf = WarmStartCBF(N, D_s=D_s, D_sense=D_sense, epsilon=epsilon)
            print(f"  one-time QP build: {time.time()-t_build:.1f}s")

        start = time.time()
        last_k = n_steps - 1          # index of the final stored frame
        hold = 0                      # consecutive converged steps
        for k in range(1, n_steps):
            t_start = time_grid[k-1]
            t_end = time_grid[k]

            if k % 25 == 0:
                print(f"Simulating time {t_end:.2f}s / {t_span[1]:.2f}s ...")

            current_states = np.array([inst.X[k-1] for inst in self.instances])

            u_nom = np.zeros((N, 2))
            for i, inst in enumerate(self.instances):
                u_nom[i] = lqr_control(current_states[i], inst.target, a_max=a_max)

            if warm_start:
                u_safe = cbf(current_states, u_nom)
            else:
                u_safe = solve_cbf_qp(current_states, u_nom, D_s=D_s, D_sense=D_sense, epsilon=epsilon)

            for i, inst in enumerate(self.instances):
                def dyn(t, x, accel=u_safe[i], p=inst.params):
                    u_th = map_accel_to_thrusts(x, accel, p)
                    return f(x, p) + g_matrix(x, p) @ u_th

                sol = solve_ivp(dyn, [t_start, t_end], current_states[i], method="RK45")
                inst.X[k] = sol.y[:, -1]

            # Convergence check: all agents at target and nearly stopped
            if converge_pos_tol is not None:
                nxt = np.array([inst.X[k] for inst in self.instances])
                pos_err = np.linalg.norm(nxt[:, 0:2] - targets, axis=1)
                spd     = np.linalg.norm(nxt[:, 3:5], axis=1)
                if np.all(pos_err <= converge_pos_tol) and np.all(spd <= converge_vel_tol):
                    hold += 1
                    if hold >= converge_hold:
                        last_k = k
                        print(f"Converged at t = {t_end:.2f}s "
                              f"(max pos err {pos_err.max():.3f} m, max speed {spd.max():.3f} m/s)")
                        break
                else:
                    hold = 0
        else:
            last_k = n_steps - 1

        # Trim histories to the actual stop time so animate() sees only real frames
        if last_k < n_steps - 1:
            for inst in self.instances:
                inst.X = inst.X[:last_k + 1]
                inst.t = time_grid[:last_k + 1]

        end = time.time()
        print("Computation time: ", end-start)
        print(f"Simulation Complete! ({last_k+1} frames, t_end = {self.instances[0].t[-1]:.2f}s)")
        return self

    def animate(self, figsize=(10, 10), interval=40, trail_len=25, arm_scale=1.0,
                box_opacity=1.0, reveal_at_end=True, save_img=True):
        """box_opacity   : alpha of the faded target squares (0..1).
        reveal_at_end : if True, the background target squares stay hidden during
                        transit and appear only on the final frame, once the drones
                        have converged to their locations. The drones themselves are
                        visible throughout. Set False to show the squares the whole time."""
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect('equal')
        ax.set_xlabel('$x$ [m]')
        ax.set_ylabel('$y$ [m]')
        ax.set_title(f'Drone Show: {len(self.instances)}-Agent Image Assembly')
        
        ax.set_facecolor('#ffffff')
        fig.patch.set_facecolor('#ffffff')
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        ax.title.set_color('black')
        ax.tick_params(colors='black')
        ax.grid(True, alpha=0.15)
        
        all_x = np.concatenate([inst.X[:, 0] for inst in self.instances])
        all_y = np.concatenate([inst.X[:, 1] for inst in self.instances])
        pad = 1.0
        ax.set_xlim(all_x.min() - pad, all_x.max() + pad)
        ax.set_ylim(all_y.min() - pad, all_y.max() + pad)

        time_text = ax.text(0.02, 0.96, '', transform=ax.transAxes, fontsize=12, va='top', color='black')

        n_frames = len(self.instances[0].t)
        reveal_frame = n_frames - 30   # target squares appear on one of the last frames

        vehicle_artists = []
        # Original small drone dimensions, body is a SOLID fill of the agent's color.
        body_w, body_h = 0.12 * arm_scale, 0.03 * arm_scale

        for inst in self.instances:
            c = inst.color
            # Faded target square (the background marker of where this drone is headed).
            # Hidden during transit if reveal_at_end; shown only once converged.
            target_sq, = ax.plot(inst.target[0], inst.target[1], 's', color=c,
                                 markersize=12, alpha=box_opacity)
            target_sq.set_visible(not reveal_at_end)

            trail, = ax.plot([], [], '-', color=c, lw=1.5, alpha=0.6)

            # Bubble radius is D_s / 2 = 0.3 (collision-avoidance reference)
            bubble = plt.Circle((0, 0), 0.3, color=c, fill=False, linestyle='--', alpha=0.3)
            ax.add_patch(bubble)

            body = mpatches.FancyBboxPatch((-body_w, -body_h), 2*body_w, 2*body_h,
                                           boxstyle="round,pad=0.01", linewidth=0, facecolor=c)
            ax.add_patch(body)

            vehicle_artists.append(dict(trail=trail, body=body, bubble=bubble, target_sq=target_sq))

        def _update(frame):
            time_text.set_text(f't = {self.instances[0].t[frame]:.2f} s')
            show_targets = (not reveal_at_end) or (frame >= reveal_frame)
            for vi, (inst, arts) in enumerate(zip(self.instances, vehicle_artists)):
                X = inst.X
                px, py, theta = X[frame, 0], X[frame, 1], X[frame, 2]

                start = max(0, frame - trail_len)
                #arts['trail'].set_data(X[start:frame + 1, 0], X[start:frame + 1, 1])
                arts['bubble'].center = (px, py)
                arts['body'].set_transform(
                    mtransforms.Affine2D().rotate_around(0, 0, theta).translate(px, py) + ax.transData)
                arts['target_sq'].set_visible(show_targets)

            return [a for arts in vehicle_artists for a in arts.values()] + [time_text]

        anim = FuncAnimation(fig, _update, frames=n_frames, interval=40, blit=False)
        if save_img:
            anim.save("drone_show.gif", writer="pillow", fps=30)
        plt.tight_layout()
        return anim




def create_example_sprite(path="example_sprite.png"):
    """Write a small multi-color pixel-art heart (transparent background) to demo
    the pipeline. Replace `path` with any pixel sprite (PNG/GIF) of your own."""
    M, H = '#e63946', '#ff8fab'
    o = None
    grid = [
        [o, H, H, o, M, M, o],
        [H, H, H, M, M, M, M],
        [M, M, M, M, M, M, M],
        [o, M, M, M, M, M, o],
        [o, o, M, M, M, o, o],
        [o, o, o, M, o, o, o],
    ]
    Hh, Ww = len(grid), len(grid[0])
    arr = np.zeros((Hh, Ww, 4), dtype=np.uint8)
    for r in range(Hh):
        for c in range(Ww):
            hx = grid[r][c]
            if hx is None:
                continue
            arr[r, c] = [int(hx[1:3], 16), int(hx[3:5], 16), int(hx[5:7], 16), 255]
    Image.fromarray(arr, 'RGBA').save(path)
    return path

def load_sprite(path, target_width=None, alpha_thresh=128, bg_color=None, bg_tol=12
                ) -> Tuple[List[Tuple[int, int]], List[str], Tuple[int, int]]:
    """
    Read a pixel sprite and return (pixels_rc, colors, (W, H)).
    Each opaque, non-background pixel becomes one drone target.
      - target_width : optional NEAREST resize (keeps the look pixelated) to cap drone count.
      - alpha_thresh : pixels with alpha below this are treated as empty (transparent bg).
      - bg_color     : optional (r,g,b) background to drop for sprites without alpha.
    """
    img = Image.open(path).convert('RGBA')
    if target_width is not None:
        W0, H0 = img.size
        new_h = max(1, round(H0 * target_width / W0))
        img = img.resize((target_width, new_h), Image.NEAREST)
    arr = np.array(img)                      # H x W x 4
    H, W = arr.shape[:2]

    pixels_rc, colors = [], []
    for row in range(H):
        for col in range(W):
            r, g, b, a = (int(v) for v in arr[row, col])
            if a < alpha_thresh:
                continue
            if bg_color is not None and all(abs(c0 - c1) <= bg_tol
                                            for c0, c1 in zip((r, g, b), bg_color)):
                continue
            pixels_rc.append((row, col))
            colors.append('#%02x%02x%02x' % (r, g, b))
    return pixels_rc, colors, (W, H)

def pixels_to_targets(pixels_rc, W, H, spacing=1.0, base_y=5.0):
    """Map image (row, col) to world (x, y): centered in x, image-top at the top."""
    targets = []
    for (row, col) in pixels_rc:
        tx = (col - (W - 1) / 2.0) * spacing
        ty = base_y + (H - 1 - row) * spacing
        targets.append(np.array([tx, ty]))
    return targets

def make_start_positions(N, width, spacing=1.0, top_y=-0.5, seed=42):
    """Scrambled launch grid on the ground (>= D_s apart), shuffled so paths cross."""
    cols = max(width, 1)
    rows = int(np.ceil(N / cols))
    slots = [np.array([(c - (cols - 1) / 2.0) * spacing, top_y - r * spacing])
             for r in range(rows) for c in range(cols)]
    rng = np.random.default_rng(seed)
    rng.shuffle(slots)
    return slots[:N]





def run_drone_show_scenario(sprite_path=None, target_width=None, spacing=1.0, base_y=5.0,
                            t_end=120.0, max_drones=45, warm_start=True,
                            box_opacity=1.0, reveal_at_end=True):
    # No sprite supplied -> generate the demo heart so this runs out of the box.
    if sprite_path is None:
        sprite_path = create_example_sprite()

    pixels_rc, colors, (W, H) = load_sprite(sprite_path, target_width=target_width)
    N = len(pixels_rc)
    if N == 0:
        raise ValueError("No opaque pixels found in sprite — check alpha/background.")
    if N > max_drones:
        raise ValueError(
            f"Sprite has {N} lit pixels -> {N} drones ({N*(N-1)//2} pairwise "
            f"constraints). The centralized CBF-QP scales ~O(N^2); pass a smaller "
            f"`target_width` to downsample (currently capped at max_drones={max_drones}).")

    targets = pixels_to_targets(pixels_rc, W, H, spacing=spacing, base_y=base_y)
    starts = make_start_positions(N, W, spacing=spacing)

    print(f"Loaded '{sprite_path}': {W}x{H} sprite -> {N} drones")

    instances = []
    for i in range(N):
        x0 = np.array([starts[i][0], starts[i][1], 0.0, 0.0, 0.0, 0.0])
        instances.append(QuadrotorInstance(x0=x0, target=targets[i], color=colors[i]))

    sim = MultiQuadrotorSim(instances)
    # t_end now just CAPS the run; the sim stops early once all agents converge.
    sim.run_centralized(D_sense=5.0, t_span=(0.0, t_end), dt=0.04, warm_start=warm_start, a_max=None, epsilon=None)
    anim = sim.animate(arm_scale=1.0, trail_len=15,
                       box_opacity=box_opacity, reveal_at_end=reveal_at_end)
    plt.show()

if __name__ == "__main__":
    run_drone_show_scenario(sprite_path="figs/flappy_bird.png", max_drones=300)