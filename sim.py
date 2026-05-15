"""
sim.py
==================
Simulate N planar quadrotors in the same space with independent
dynamics, parameters, and controllers.

Key additions over the single-quadrotor version
------------------------------------------------
* QuadrotorInstance  – bundles (params, x0, controller, label, color)
                       for one vehicle.
* MultiQuadrotorSim  – holds a list of QuadrotorInstance objects,
                       runs them in parallel, stores results.
* run_multi_study()  – demo that creates 6 heterogeneous vehicles and
                       calls all plotting / animation helpers.

Everything from the original module (QuadrotorParams, f, g_matrix,
dynamics, simulate, make_feedback_lin_controller) is unchanged so
existing code that imports from this file keeps working.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Core physics (unchanged)
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


def dynamics(t: float, x: np.ndarray,
             u_func: Callable, params: QuadrotorParams) -> np.ndarray:
    u = np.asarray(u_func(t, x), dtype=float)
    return f(x, params) + g_matrix(x, params) @ u


def simulate(
    u_func: Callable[[float, np.ndarray], np.ndarray],
    x0: np.ndarray,
    t_span: Tuple[float, float],
    params: QuadrotorParams | None = None,
    dt: float = 0.01,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    if params is None:
        params = QuadrotorParams()
    t_eval = np.arange(t_span[0], t_span[1] + dt, dt)
    sol = solve_ivp(
        fun=lambda t, x: dynamics(t, x, u_func, params),
        t_span=t_span, y0=x0, method="RK45",
        t_eval=t_eval, rtol=rtol, atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"Solver failed: {sol.message}")
    return sol.t, sol.y.T


def make_feedback_lin_controller(
    params: QuadrotorParams,
    y_d: float = 1.0,
    th_d: float = 0.0,
    omega_y: float = 3.0,
    omega_th: float = 5.0,
) -> Callable:
    kp_y  = omega_y  ** 2
    kd_y  = 2 * omega_y
    kp_th = omega_th ** 2
    kd_th = 2 * omega_th
    m, I, r, g = params.m, params.I, params.r, params.g

    def controller(t: float, x: np.ndarray) -> np.ndarray:
        _, y, theta, _, yd, thd = x
        G = np.array([
            [np.cos(theta) / m,  np.cos(theta) / m],
            [r / I,             -r / I            ],
        ])
        F_vec = np.array([-g, 0.0])
        v = np.array([
            -kp_y  * (y     - y_d)  - kd_y  * yd,
            -kp_th * (theta - th_d) - kd_th * thd,
        ])
        try:
            u = np.linalg.solve(G, -F_vec + v)
        except np.linalg.LinAlgError:
            u = np.zeros(2)
        return u

    return controller


# ─────────────────────────────────────────────────────────────────────────────
# N-quadrotor infrastructure
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QuadrotorInstance:
    """
    Represents one quadrotor vehicle.

    Parameters
    ----------
    x0         : initial state  [x, y, θ, ẋ, ẏ, θ̇]
    controller : callable (t, x) → [u1, u2]
    params     : physical parameters (defaults to QuadrotorParams())
    label      : display name shown in plots / animation
    color      : any matplotlib colour spec; auto-assigned if None
    """
    x0:         np.ndarray
    controller: Callable
    params:     QuadrotorParams = field(default_factory=QuadrotorParams)
    label:      str             = ""
    color:      Optional[str]   = None

    # Simulation results – populated by MultiQuadrotorSim.run()
    t:  Optional[np.ndarray] = field(default=None, repr=False)
    X:  Optional[np.ndarray] = field(default=None, repr=False)   # (N_steps, 6)


class MultiQuadrotorSim:
    """
    Manages N QuadrotorInstance objects and provides helpers for
    simulation, plotting, and animation.

    Usage
    -----
    sim = MultiQuadrotorSim(instances)
    sim.run(t_span=(0, 6), dt=0.01)
    sim.plot_states()
    sim.plot_phase()
    sim.plot_controls()
    sim.animate()           # 2-D spatial animation
    plt.show()
    """

    _DEFAULT_PALETTE = plt.cm.tab10(np.linspace(0, 0.9, 10))

    def __init__(self, instances: List[QuadrotorInstance]):
        if not instances:
            raise ValueError("Need at least one QuadrotorInstance.")
        self.instances = instances
        # Assign colours for any vehicle that didn't specify one
        for i, inst in enumerate(self.instances):
            if inst.color is None:
                inst.color = self._DEFAULT_PALETTE[i % 10]
            if not inst.label:
                inst.label = f"Quad {i + 1}"

    # ── simulation ──────────────────────────────────────────────────────────

    def run(
        self,
        t_span: Tuple[float, float] = (0.0, 6.0),
        dt: float = 0.01,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        verbose: bool = True,
    ) -> "MultiQuadrotorSim":
        """
        Simulate every quadrotor over *t_span* independently.
        Results are stored on each QuadrotorInstance (.t, .X).
        Returns *self* for method chaining.
        """
        for inst in self.instances:
            if verbose:
                print(f"  Simulating {inst.label} …", end=" ", flush=True)
            t, X = simulate(inst.controller, inst.x0, t_span,
                            params=inst.params, dt=dt, rtol=rtol, atol=atol)
            inst.t = t
            inst.X = X
            if verbose:
                print("done.")
        return self

    # ── helpers ─────────────────────────────────────────────────────────────

    def _check_run(self):
        if any(inst.t is None for inst in self.instances):
            raise RuntimeError("Call .run() before plotting / animating.")

    def _controls(self, inst: QuadrotorInstance) -> np.ndarray:
        """Re-evaluate controller at every saved time-step → (N, 2)."""
        return np.array([inst.controller(inst.t[k], inst.X[k])
                         for k in range(len(inst.t))])

    # ── plotting ─────────────────────────────────────────────────────────────

    def plot_states(self, figsize=(13, 8)) -> plt.Figure:
        """
        Time-series plots of y, θ, ẏ, θ̇ for every quadrotor.
        Each vehicle's desired setpoint (if the controller exposes it) is
        shown as a dashed reference line when it can be inferred.
        """
        self._check_run()
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        ax_y, ax_th, ax_yd, ax_thd = (axes[0, 0], axes[0, 1],
                                       axes[1, 0], axes[1, 1])

        for inst in self.instances:
            kw = dict(color=inst.color, lw=1.8, label=inst.label)
            ax_y.plot(inst.t,  inst.X[:, 1], **kw)
            ax_th.plot(inst.t, inst.X[:, 2], **kw)
            ax_yd.plot(inst.t, inst.X[:, 4], **kw)
            ax_thd.plot(inst.t,inst.X[:, 5], **kw)

        specs = [
            (ax_y,   '$y$ [m]',                'Vertical position $y(t)$'),
            (ax_th,  r'$\theta$ [rad]',         r'Tilt angle $\theta(t)$'),
            (ax_yd,  r'$\dot{y}$ [m/s]',        r'Vertical velocity $\dot{y}(t)$'),
            (ax_thd, r'$\dot{\theta}$ [rad/s]', r'Angular rate $\dot{\theta}(t)$'),
        ]
        t_end = max(inst.t[-1] for inst in self.instances)
        for ax, ylabel, title in specs:
            ax.set_xlabel('Time [s]')
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontsize=11)
            ax.grid(True, alpha=0.35)
            ax.set_xlim(0, t_end)

        handles, labels = ax_y.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center',
                   ncol=min(len(self.instances), 4),
                   fontsize=8, framealpha=0.9,
                   bbox_to_anchor=(0.5, 0.0))
        fig.suptitle('State Time-Series — All Quadrotors', fontsize=13, y=1.01)
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        return fig

    def plot_phase(self, figsize=(11, 4.5)) -> plt.Figure:
        """Phase portraits  y–ẏ  and  θ–θ̇  for every quadrotor."""
        self._check_run()
        fig, (ax_y, ax_th) = plt.subplots(1, 2, figsize=figsize)

        for inst in self.instances:
            ax_y.plot( inst.X[:, 1], inst.X[:, 4],
                       color=inst.color, lw=1.6, label=inst.label)
            ax_y.plot( inst.x0[1],   inst.x0[4],
                       'o', color=inst.color, ms=6)

            ax_th.plot(inst.X[:, 2], inst.X[:, 5],
                       color=inst.color, lw=1.6, label=inst.label)
            ax_th.plot(inst.x0[2],   inst.x0[5],
                       'o', color=inst.color, ms=6)

        ax_y.set_xlabel('$y$ [m]');      ax_y.set_ylabel(r'$\dot{y}$ [m/s]')
        ax_y.set_title('$y$–$\\dot{y}$ Phase Portrait')
        ax_y.grid(True, alpha=0.35);     ax_y.legend(fontsize=8)

        ax_th.set_xlabel(r'$\theta$ [rad]')
        ax_th.set_ylabel(r'$\dot{\theta}$ [rad/s]')
        ax_th.set_title(r'$\theta$–$\dot{\theta}$ Phase Portrait')
        ax_th.grid(True, alpha=0.35);    ax_th.legend(fontsize=8)

        plt.tight_layout()
        return fig

    def plot_controls(self, figsize=(11, 4)) -> plt.Figure:
        """Rotor thrust time-series for every quadrotor."""
        self._check_run()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        t_end = max(inst.t[-1] for inst in self.instances)

        for inst in self.instances:
            U = self._controls(inst)
            ax1.plot(inst.t, U[:, 0], color=inst.color, lw=1.6, label=inst.label)
            ax2.plot(inst.t, U[:, 1], color=inst.color, lw=1.6, label=inst.label)

        for ax, title, ylabel in [
            (ax1, 'Rotor 1 thrust $u_1(t)$', '$u_1$ [N]'),
            (ax2, 'Rotor 2 thrust $u_2(t)$', '$u_2$ [N]'),
        ]:
            ax.set_xlabel('Time [s]')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.35)
            ax.set_xlim(0, t_end)
            ax.legend(fontsize=8)

        plt.tight_layout()
        return fig

    def plot_trajectories(self, figsize=(8, 6)) -> plt.Figure:
        """
        2-D x–y spatial trajectories for every quadrotor.
        Start positions are marked with circles, end positions with stars.
        """
        self._check_run()
        fig, ax = plt.subplots(figsize=figsize)

        for inst in self.instances:
            ax.plot(inst.X[:, 0], inst.X[:, 1],
                    color=inst.color, lw=1.8, label=inst.label)
            ax.plot(inst.X[0, 0],  inst.X[0, 1],
                    'o', color=inst.color, ms=7, zorder=5)
            ax.plot(inst.X[-1, 0], inst.X[-1, 1],
                    '*', color=inst.color, ms=10, zorder=5)

        ax.set_xlabel('$x$ [m]')
        ax.set_ylabel('$y$ [m]')
        ax.set_title('Spatial Trajectories  (○ start, ★ end)')
        ax.grid(True, alpha=0.35)
        ax.set_aspect('equal', 'datalim')
        ax.legend(fontsize=8)
        plt.tight_layout()
        return fig

    # ── 2-D animation ────────────────────────────────────────────────────────

    def animate(
        self,
        figsize: Tuple[float, float] = (10, 7),
        interval: int = 30,
        trail_len: int = 80,
        arm_scale: float = 1.0,
    ) -> FuncAnimation:
        """
        2-D animation showing all quadrotors flying simultaneously.

        Each vehicle is drawn as a coloured rectangle with two rotors.
        A short trajectory trail follows each vehicle.

        Parameters
        ----------
        figsize   : figure size
        interval  : milliseconds between frames
        trail_len : number of past positions to draw as a trail
        arm_scale : scale factor applied to the arm/body size for visibility
        """
        self._check_run()

        # ── resample all trajectories onto a common time grid ────────────
        t_min = max(inst.t[0]  for inst in self.instances)
        t_max = min(inst.t[-1] for inst in self.instances)
        n_frames = min(len(inst.t) for inst in self.instances)
        t_common = np.linspace(t_min, t_max, n_frames)

        # Interpolate each state onto the common grid
        X_all = []
        for inst in self.instances:
            X_interp = np.column_stack([
                np.interp(t_common, inst.t, inst.X[:, col])
                for col in range(6)
            ])
            X_all.append(X_interp)

        # ── figure setup ─────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=figsize)

        # Compute axis limits from all trajectories
        all_x = np.concatenate([X[:, 0] for X in X_all])
        all_y = np.concatenate([X[:, 1] for X in X_all])
        pad = 0.6
        ax.set_xlim(all_x.min() - pad, all_x.max() + pad)
        ax.set_ylim(all_y.min() - pad, all_y.max() + pad)
        ax.set_aspect('equal')
        ax.set_xlabel('$x$ [m]')
        ax.set_ylabel('$y$ [m]')
        ax.set_title('Multi-Quadrotor 2-D Animation')
        ax.grid(True, alpha=0.25)

        time_text = ax.text(0.02, 0.96, '', transform=ax.transAxes,
                            fontsize=10, va='top')

        # ── per-vehicle artists ───────────────────────────────────────────
        body_w  = 0.12 * arm_scale   # rectangle half-width
        body_h  = 0.03 * arm_scale   # rectangle half-height
        rotor_r = 0.05 * arm_scale   # rotor disc radius

        vehicle_artists = []   # list of dicts, one per quadrotor
        for inst in self.instances:
            c = inst.color

            # Trajectory trail
            trail, = ax.plot([], [], '-', color=c, lw=1.2, alpha=0.55)

            # Body rectangle (centred at origin, rotated later)
            body = mpatches.FancyBboxPatch(
                (-body_w, -body_h), 2 * body_w, 2 * body_h,
                boxstyle="round,pad=0.01",
                linewidth=1.2, edgecolor=c,
                facecolor=(*np.asarray(plt.cm.colors.to_rgb(c) if isinstance(c, str)
                            else c[:3]), 0.35),
            )
            ax.add_patch(body)

            # Two rotor discs
            r_left  = mpatches.Circle((0, 0), rotor_r, color=c,
                                       alpha=0.55, zorder=4)
            r_right = mpatches.Circle((0, 0), rotor_r, color=c,
                                       alpha=0.55, zorder=4)
            ax.add_patch(r_left)
            ax.add_patch(r_right)

            # Label text
            txt = ax.text(0, 0, inst.label, fontsize=7, ha='center',
                          va='bottom', color=c, fontweight='bold', zorder=6)

            vehicle_artists.append(dict(
                trail=trail, body=body,
                r_left=r_left, r_right=r_right, txt=txt,
            ))

        # Legend (static)
        legend_patches = [
            mpatches.Patch(color=inst.color, label=inst.label)
            for inst in self.instances
        ]
        ax.legend(handles=legend_patches, loc='upper right', fontsize=8)

        # ── update function ───────────────────────────────────────────────
        def _update(frame: int):
            time_text.set_text(f't = {t_common[frame]:.2f} s')

            for vi, (inst, arts) in enumerate(
                    zip(self.instances, vehicle_artists)):
                X  = X_all[vi]
                px, py, theta = X[frame, 0], X[frame, 1], X[frame, 2]

                # Trail
                start = max(0, frame - trail_len)
                arts['trail'].set_data(X[start:frame + 1, 0],
                                       X[start:frame + 1, 1])

                # Body – translate + rotate
                # FancyBboxPatch is defined centred at origin (-body_w, -body_h),
                # so we only need to rotate then translate to (px, py).
                arts['body'].set_transform(
                    mtransforms.Affine2D()
                    .rotate_around(0, 0, theta)
                    .translate(px, py)
                    + ax.transData
                )

                # Rotor positions in body frame
                r  = inst.params.r * arm_scale
                lx = px + (-r) * np.cos(theta) - 0 * np.sin(theta)
                ly = py + (-r) * np.sin(theta) + 0 * np.cos(theta)
                rx = px + ( r) * np.cos(theta) - 0 * np.sin(theta)
                ry = py + ( r) * np.sin(theta) + 0 * np.cos(theta)

                arts['r_left'].set_center((lx, ly))
                arts['r_right'].set_center((rx, ry))

                # Label offset above body
                arts['txt'].set_position((px, py + 0.12 * arm_scale))

            return ([a for arts in vehicle_artists
                     for a in (arts['trail'], arts['body'],
                                arts['r_left'], arts['r_right'],
                                arts['txt'])]
                    + [time_text])

        anim = FuncAnimation(fig, _update, frames=n_frames,
                             interval=interval, blit=False)
        plt.tight_layout()
        return anim


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

def run_multi_study():
    """
    Build a heterogeneous fleet of 6 quadrotors with different:
    - mass / inertia
    - initial positions
    - desired hover heights
    - closed-loop bandwidths
    and run the full suite of plots + the spatial animation.
    """

    # ── define vehicles ───────────────────────────────────────────────────
    vehicles: List[QuadrotorInstance] = []

    # 1 – lightweight, fast-bandwidth, hovers at y=1
    p1 = QuadrotorParams(m=0.8, I=0.008, r=0.12)
    vehicles.append(QuadrotorInstance(
        x0=np.array([0.0,  0.0,  0.0,  0.0,  0.0,  0.0]),
        controller=make_feedback_lin_controller(p1, y_d=1.0, th_d=0.0,
                                                omega_y=4.0, omega_th=6.0),
        params=p1, label='Quad A – light, fast',
    ))

    # 2 – heavier, slower, hovers at y=2
    p2 = QuadrotorParams(m=1.5, I=0.02, r=0.18)
    vehicles.append(QuadrotorInstance(
        x0=np.array([ 1.0,  2.5,  0.3,  0.0,  0.0,  0.0]),
        controller=make_feedback_lin_controller(p2, y_d=2.0, th_d=0.0,
                                                omega_y=2.0, omega_th=4.0),
        params=p2, label='Quad B – heavy, slow',
    ))

    # 3 – default params, starts below with velocities
    p3 = QuadrotorParams()
    vehicles.append(QuadrotorInstance(
        x0=np.array([-1.0, -0.5, -0.4,  0.0,  0.5, -0.3]),
        controller=make_feedback_lin_controller(p3, y_d=1.5, th_d=0.0,
                                                omega_y=3.0, omega_th=5.0),
        params=p3, label='Quad C – default',
    ))

    # 4 – high initial tilt, hovers at y=0.5
    p4 = QuadrotorParams(m=1.2, I=0.012, r=0.14)
    vehicles.append(QuadrotorInstance(
        x0=np.array([ 2.0,  3.0,  0.6,  0.2, -0.3,  0.2]),
        controller=make_feedback_lin_controller(p4, y_d=0.5, th_d=0.0,
                                                omega_y=3.5, omega_th=5.5),
        params=p4, label='Quad D – high tilt',
    ))

    # 5 – negative initial tilt, aggressive bandwidth
    p5 = QuadrotorParams(m=0.9, I=0.009, r=0.13)
    vehicles.append(QuadrotorInstance(
        x0=np.array([ 0.5,  0.5, -0.5, -0.1,  0.1,  0.4]),
        controller=make_feedback_lin_controller(p5, y_d=1.2, th_d=0.0,
                                                omega_y=5.0, omega_th=8.0),
        params=p5, label='Quad E – aggressive',
    ))

    # 6 – offset x, moderate tilt, hover at y=1.8
    p6 = QuadrotorParams(m=1.1, I=0.011, r=0.16)
    vehicles.append(QuadrotorInstance(
        x0=np.array([-2.0,  1.5,  0.2,  0.3, -0.2, -0.1]),
        controller=make_feedback_lin_controller(p6, y_d=1.8, th_d=0.0,
                                                omega_y=2.5, omega_th=4.5),
        params=p6, label='Quad F – moderate',
    ))

    # ── simulate ──────────────────────────────────────────────────────────
    print("Running multi-quadrotor simulation …")
    sim = MultiQuadrotorSim(vehicles)
    sim.run(t_span=(0.0, 6.0), dt=0.01)

    # ── plots ─────────────────────────────────────────────────────────────
    sim.plot_states()
    sim.plot_phase()
    sim.plot_controls()
    sim.plot_trajectories()

    # ── animation ─────────────────────────────────────────────────────────
    anim = sim.animate(arm_scale=1.2, trail_len=100)
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Minimal API example (adding quadrotors programmatically)
# ─────────────────────────────────────────────────────────────────────────────
def example_programmatic_n(n: int = 4, t_end: float = 5.0):
    """
    Spawn *n* quadrotors arranged in a horizontal line, each targeting a
    different hover altitude.  Demonstrates the simplest programmatic usage.

        sim = MultiQuadrotorSim([...])
        sim.run(...)
        sim.plot_states()
    """
    rng = np.random.default_rng(42)
    instances = []
    for i in range(n):
        params = QuadrotorParams(
            m=rng.uniform(0.8, 1.4),
            I=rng.uniform(0.007, 0.015),
            r=rng.uniform(0.11, 0.18),
        )
        x0 = np.array([
            float(i) * 1.2 - (n - 1) * 0.6,   # spread in x
            rng.uniform(-0.5, 0.5),           # random y start
            rng.uniform(-0.3, 0.3),           # random tilt
            0.0, 0.0, 0.0,
        ])
        y_d = rng.uniform(0.5, 2.5)
        ctrl = make_feedback_lin_controller(params, y_d=y_d,
                                            omega_y=rng.uniform(2, 5),
                                            omega_th=rng.uniform(4, 8))
        instances.append(QuadrotorInstance(
            x0=x0, controller=ctrl, params=params,
            label=f'Q{i+1} (y_d={y_d:.1f}m)'
        ))

    sim = MultiQuadrotorSim(instances)
    sim.run(t_span=(0.0, t_end))
    sim.plot_states()
    sim.plot_trajectories()
    anim = sim.animate()
    plt.show()


if __name__ == "__main__":
    # run_multi_study()
    example_programmatic_n(n=20)