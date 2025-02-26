import numpy as np

import matplotlib.pyplot as plt


def rotation_matrix(roll, pitch, yaw):
    # Rotation matrix around X-axis (roll)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])

    # Rotation matrix around Y-axis (pitch)
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])

    # Rotation matrix around Z-axis (yaw)
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])

    # Combine rotations: R = Rz * Ry * Rx
    R = Rz @ Ry @ Rx

    return R


import numpy as np


def rotation_matrix(roll, pitch, yaw):
    """Generate a 3D rotation matrix given roll, pitch, and yaw (in radians)."""
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])

    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])

    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])

    return R_z @ R_y @ R_x  # Combined rotation matrix (Z-Y-X intrinsic rotations)


def weight_force(roll, pitch, yaw, mass, g=9.81):
    g_vector = np.array([0, 0, -mass * g])  # Gravity force in world frame (downward)
    R = rotation_matrix(roll, pitch, yaw)  # Compute rotation matrix
    F_rotated = R @ g_vector  # Transform gravity force to rotated frame
    return F_rotated


def buoyant_force(roll, pitch, yaw, volume, d, rho=1000, g=9.81):
    """
    Compute the buoyant force and torque on an object submerged in a fluid.

    :param roll: Rotation around x-axis (radians)
    :param pitch: Rotation around y-axis (radians)
    :param yaw: Rotation around z-axis (radians)
    :param volume: Submerged volume (m³)
    :param d: 3D vector of distance from center of mass to center of buoyancy (array-like)
    :param rho: Fluid density (kg/m³), default is water (1000 kg/m³)
    :param g: Gravitational acceleration (m/s²), default is 9.81 m/s²
    :return: 1D NumPy array [Fx, Fy, Fz, Mx, My, Mz]
    """
    d = np.array(d)  # Ensure d is a NumPy array
    gravity_vector = np.array([0, 0, -1])  # Gravity acts downward in world frame
    buoyant_force_magnitude = -1 *rho * g * volume  # Magnitude of buoyant force
    buoyant_force_world = buoyant_force_magnitude * gravity_vector  # In world frame

    # Rotate buoyant force to the object's frame
    R = rotation_matrix(roll, pitch, yaw)
    buoyant_force_body = R @ buoyant_force_world

    # Compute torque using cross product τ = d × F_buoyancy
    torque = np.cross(d, buoyant_force_body)

    # Combine force and torque into a single output array
    return np.concatenate((buoyant_force_body, torque))


def pos_vel_case_1(V_i, S_i, m, C, T, dt):
    if T == 0 and V_i == 0:
        S = S_i
        V = 0
        return S, V
    elif T == 0 and V_i > 0:
        S = S_i - (m / C) * np.log(m / abs(-C * V_i * dt - m))
        V = - (-1 / V_i - C * dt / m) ** (-1)
        return S, V
    elif T == 0 and V_i < 0:
        S = - S_i - (m / C) * np.log(m / abs(C * V_i * dt - m))
        V = - (1 / V_i - C * dt / m) ** (-1)
        return -S, -V
    else:
        raise ValueError("Invalid input")


def pos_vel_case_2(V_i, S_i, m, C, T, dt):
    #print(f"V_i = {V_i}, S_i = {S_i}, T = {T}  case 2")


    if T > 0 and V_i >= 0:
        a = (T / C) ** (1 / 2)

        # check if at terminal velocity to avoid /0 error
        if V_i == a:
            V = V_i
            S = S_i + dt * V
        else:
            A = (a + V_i) / (a - V_i)
            B = 2 * a * C / m
            D = abs(A * np.exp(B * dt) + 1)
            E = 2 * np.log(D) / B - dt
            F = 2 *np.log(abs(A+1)) / B
            S = S_i + a * E - a * F
            num3 = A * np.exp(B * dt) - 1
            den3 = A * np.exp(B * dt) + 1
            V = a * num3 / den3
        return S, V
    elif T < 0 and V_i <= 0:
        #print("calling case 2 from case 2")
        S, V = pos_vel_case_2(-V_i, -S_i, m, C, -T, dt)
        return -S, -V
    else:
        #print(f"Vi: {V_i}, S: {S_i}, V: {V_i}, T: {T}, dt: {dt}")
        raise ValueError("Invalid input")


def pos_vel_case_3(V_i, S_i, m, C, T, dt):
    a = (C / m) ** (1 / 2)
    A = np.arctan(V_i / a)
    B = (C * a * dt / m) + A
    D = m * np.log(abs(1 / np.cos(B))) / C
    E = m * np.log(abs(1 / np.cos(A))) / C

    #print(f"V_i = {V_i}, S_i = {S_i}, T = {T}")

    if T > 0 > V_i:
        S = S_i + D - E
        V = a * np.tan(A + a * C * dt / m)
        if V > 0:
            dt_0 = np.arctan(V/a)
        return S, V

        # this whole block was causing major issues
        # time at which velocity becomes 0
        # t_to_v0 = -m / (a * C) * np.arctan(V_i / a)
        # if t_to_v0 > dt:
        #     print("t_to_v0 > dt")
        #     S1, V1 = pos_vel_case_3(V_i, S_i, m, C, T, t_to_v0)
        #     print("calling case 2 from case 3")
        #     S2, V2 = pos_vel_case_2(V1, S1, m, C, T, dt - t_to_v0)
        #     return S1 + S2, V1 + V2
        # else:
        #     print("t_to_v0 < dt")
        #     S = S_i + D - E
        #     V = a * np.tan(A + a * C * dt / m)
        #    return S, V
    elif T < 0 and V_i > 0:
        S, V = pos_vel_case_3(-V_i, -S_i, m, C, -T, dt)
        return -S, -V
    else:
        raise ValueError("Invalid input")


def pos_vel(V_i, S_i, m, C, T, dt):
    if C < 0 or m <= 0 or dt <= 0:
        #print(f"C = {C}, m = {m}, dt = {dt}")
        raise ValueError("Invalid input")
    elif T == 0:
        return pos_vel_case_1(V_i, S_i, m, C, T, dt)
    elif V_i == 0:
        return pos_vel_case_2(V_i, S_i, m, C, T, dt)
    elif (V_i > 0 and T > 0) or (V_i < 0 and T < 0):
        return pos_vel_case_2(V_i, S_i, m, C, T, dt)
    elif (V_i < 0 < T) or (V_i > 0 > T):
        return pos_vel_case_3(V_i, S_i, m, C, T, dt)
    else:
        raise ValueError("Invalid input")


def plot_pos_vel(V_i, S_i, m, C, T, dt, t):

    time_steps = np.arange(0, t, dt)  # Generate time steps
    positions = []
    velocities = []

    S, V = S_i, V_i  # Initialize position and velocity

    #positions.append(S)
    #velocities.append(V)

    for _ in time_steps:
        S, V = pos_vel(V, S, m, C, T, dt)  # Update position and velocity
        positions.append(S)
        velocities.append(V)

    # Plot results
    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Position", color="tab:blue")
    ax1.plot(time_steps, positions, label="Position", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()  # Create a second y-axis for velocity
    ax2.set_ylabel("Velocity", color="tab:red")
    ax2.plot(time_steps, velocities, label="Velocity", color="tab:red", linestyle="dashed")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    fig.tight_layout()  # Adjust layout to fit both plots
    plt.title("Position and Velocity Over Time")
    plt.show(block=True)


#plot_pos_vel(1, 0, 1, 1, -1, 0.01, 5)
