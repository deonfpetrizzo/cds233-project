import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass

@dataclass
class QuadrotorParams:
    m: float = 1.0    # mass [kg]
    I: float = 0.01   # moment of inertia [kg·m²]
    r: float = 0.15   # half-span [m]
    g: float = 9.81   # gravity [m/s²]

# 1. High-Level Nominal Controller (Double Integrator)
def get_nominal_acceleration(state, target_pos, kp=3.0, kd=2.5):
    """ PD controller to drive the drone to a target (x, y) coordinate. """
    x, y, _, vx, vy, _ = state
    tx, ty = target_pos
    
    ax_nom = kp * (tx - x) - kd * vx
    ay_nom = kp * (ty - y) - kd * vy
    return np.array([ax_nom, ay_nom])


# 2. THE CBF-QP (Section 8 Math implemented in CVXPY)
def solve_cbf_qp(states, u_nom, D_s=0.6, alpha1=2.0, gamma=2.0):
    """
    Takes in current states and nominal accelerations.
    Returns safe accelerations that prevent collisions.
    """
    N = states.shape[0]
    u = cp.Variable(2 * N) # Stacked safe accelerations:
    
    # Objective: Minimize difference between safe u and nominal u
    u_nom_flat = u_nom.flatten()
    objective = cp.Minimize(cp.sum_squares(u - u_nom_flat))
    
    constraints = []
    
    # Pairwise collision avoidance constraints (The N(N-1)/2 loop)
    for i in range(N):
        for j in range(i + 1, N):
            p_i, v_i = states[i, 0:2], states[i, 3:5]
            p_j, v_j = states[j, 0:2], states[j, 3:5]
            
            dp = p_i - p_j
            dv = v_i - v_j
            
            #A coefficients for u_i and u_j
            A_i = 2 * dp
            A_j = -2 * dp
            
            # RHS of the inequality
            term1 = -2 * np.linalg.norm(dv)**2
            term2 = -(gamma + alpha1) * (2 * np.dot(dp, dv))
            term3 = -alpha1 * gamma * (np.linalg.norm(dp)**2 - D_s**2)
            RHS = term1 + term2 + term3
            
            # Add to CVXPY constraints
            # u[2*i:2*i+2] gets the 2D acceleration for drone i
            constraints.append(A_i @ u[2*i : 2*i+2] + A_j @ u[2*j : 2*j+2] >= RHS)
            
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.OSQP)
        if prob.status == 'optimal':
            return u.value.reshape((N, 2))
    except Exception as e:
        print("QP Failed, defaulting to nominal")
    
    return u_nom # Fallback if QP fails

# 3. Low-Level Controller (Map Accel -> Thrusts)
def map_accel_to_thrusts(state, accel_safe, params):
    """ Converts safe 2D acceleration into real rotor thrusts (u1, u2) """
    x, y, theta, vx, vy, theta_dot = state
    ax, ay = accel_safe
    
    # 1. Calculate total thrust needed
    T_total = params.m * np.sqrt(ax**2 + (ay + params.g)**2)
    
    # 2. Calculate desired tilt angle to achieve x-acceleration
    theta_d = np.arctan2(-ax, ay + params.g)
    
    # 3. Attitude PD controller to track theta_d
    kp_th, kd_th = 20.0, 5.0
    torque = params.I * (kp_th * (theta_d - theta) - kd_th * theta_dot)
    
    # 4. Map Total Thrust and Torque to individual rotors
    u1 = (T_total + torque / params.r) / 2.0
    u2 = (T_total - torque / params.r) / 2.0
    return np.array([u1, u2])

# 4. Centralized Simulation Loop
def simulate_swarm(initial_states, targets, t_end=6.0, dt=0.02):
    N = initial_states.shape[0]
    params = QuadrotorParams()
    
    time = np.arange(0, t_end, dt)
    history = np.zeros((len(time), N, 6))
    history[0] = initial_states
    
    print("Simulating Swarm with Centralized CBF-QP...")
    for k in range(1, len(time)):
        current_states = history[k-1]
        
        # 1. Get Nominal Control
        u_nom = np.zeros((N, 2))
        for i in range(N):
            u_nom[i] = get_nominal_acceleration(current_states[i], targets[i])
            
        # 2. Filter through CBF-QP
        u_safe = solve_cbf_qp(current_states, u_nom, D_s=0.6)
        
        # 3. Apply physics
        next_states = np.zeros((N, 6))
        for i in range(N):
            thrusts = map_accel_to_thrusts(current_states[i], u_safe[i], params)
            u1, u2 = thrusts
            
            # Physics dynamics
            th = current_states[i, 2]
            x_ddot = -(u1 + u2) * np.sin(th) / params.m
            y_ddot = (u1 + u2) * np.cos(th) / params.m - params.g
            th_ddot = params.r * (u1 - u2) / params.I
            
            # Update state (position += vel*dt, vel += accel*dt)
            next_states[i, 0:3] = current_states[i, 0:3] + current_states[i, 3:6] * dt
            next_states[i, 3:6] = current_states[i, 3:6] + np.array([x_ddot, y_ddot, th_ddot]) * dt
            
        history[k] = next_states
        
    print("Simulation Complete!")
    return time, history

# 5. Animation

# (Head-on Collision Test)
def run_and_animate():
    # Scenario: Drone 0 is at x=-2, wants to go to x=2
    #           Drone 1 is at x=2, wants to go to x=-2
    # They are at the exact same altitude (y=1). Without CBF, they WILL crash at x=0.
    
    # State: [x, y, theta, vx, vy, theta_dot]
    initial_states = np.array([
        [-2.5, 1.0, 0.0, 0.0, 0.0, 0.0],
        [ 2.5, 1.0, 0.0, 0.0, 0.0, 0.0]
    ])
    
    targets = np.array([
        [ 2.5, 1.0],
        [-2.5, 1.0]
    ])
    
    t, history = simulate_swarm(initial_states, targets, t_end=5.0, dt=0.02)
    
    # Basic Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-3, 3)
    ax.set_ylim(0, 3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title("Multi-Agent CBF: Head-on Collision Avoidance")
    
    # Plot target points
    ax.plot(2.5, 1.0, 'b*', markersize=12, label="Target 1")
    ax.plot(-2.5, 1.0, 'r*', markersize=12, label="Target 2")
    
    drone1_dot, = ax.plot([], [], 'bo', markersize=10, label="Drone 1")
    drone2_dot, = ax.plot([], [], 'ro', markersize=10, label="Drone 2")
    
    # Draw a circle representing the Safe Distance (D_s = 0.6) around Drone 1
    safe_radius = plt.Circle((0, 0), 0.3, color='b', fill=False, linestyle='--')
    ax.add_patch(safe_radius)
    
    ax.legend(loc='upper right')

    def update(frame):
        # Update drone positions
        x1, y1 = history[frame, 0, 0:2]
        x2, y2 = history[frame, 1, 0:2]
        
        drone1_dot.set_data([x1], [y1])
        drone2_dot.set_data([x2], [y2])
        safe_radius.center = (x1, y1)
        
        return drone1_dot, drone2_dot, safe_radius

    anim = FuncAnimation(fig, update, frames=len(t), interval=20, blit=True)
    plt.show()

if __name__ == "__main__":
    run_and_animate()