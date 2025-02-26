import numpy as np

import matplotlib.pyplot as plt

# Constants (keep all uppercase for consistency)
RHO = 1000
GRAVITY = 9.81

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


def wrap_to_pi(angle):
    """
    Wrap an angle from (-∞, ∞) to (-π, π).

    :param angle: NumPy array or list of angle in radians
    :return: NumPy array with angle wrapped to the range (-π, π)
    """
    converted_angle = (angle + np.pi) % (2 * np.pi) - np.pi
    # converted_angle = 2*np.atan(np.tan(angle/2)) # secondary option for calculating angle

    if converted_angle == -1* np.pi:
        return np.pi
    return converted_angle

def wrap_array_to_pi(angles):
    result = np.zeros_like(angles)
    for i  in range(len(angles)):
        result[i] = wrap_to_pi(angles[i])

def weight_force(roll, pitch, yaw, mass, g=GRAVITY):
    g_vector = np.array([0, 0, -mass * g])  # Gravity force in world frame (downward)
    R = rotation_matrix(roll, pitch, yaw)  # Compute rotation matrix
    F_rotated = R @ g_vector  # Transform gravity force to rotated frame
    return F_rotated


def buoyant_force(roll, pitch, yaw, volume, d, rho=RHO, g=GRAVITY):
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
    b_vector = np.array([0, 0, rho * g * volume])  # Gravity acts downward in world frame

    # Rotate buoyant force to the object's frame
    R = rotation_matrix(roll, pitch, yaw)
    buoyant_force_body = R @ b_vector

    # Compute torque using cross product τ = d × F_buoyancy
    # I do not know why d needs to be negative here
    torque = np.cross(-d, buoyant_force_body)

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

def pwm_force_scalar(pwm):
    """
    Forward function:
      pwm in [1100..1900] -> force (float)

    Uses piecewise cubic polynomials in terms of x = pwm/1000,
    following your original formula exactly.
    """

    pwm = pwm / 1000.0
    x = pwm
    if not (1100 <= pwm <= 1900):
        raise ValueError(f"PWM {pwm} out of valid range [1100..1900]")


    # --- Piece 1 ---
    if 1100 <= pwm < 1460:
        # a1, b1, c1, d1
        return -1 * (
            -1.24422882971549e-8 * x**3
            + 4.02057100632393e-5   * x**2
            - 0.0348619861030835   * x
            + 3.90671429105423
        )
    # --- Piece 2 ---
    elif 1460 <= pwm <= 1540:
        # force = 0
        return 0.0
    # --- Piece 3 ---
    elif 1540 < pwm <= 1900:
        # a3, b3, c3, d3
        return (
            -1.64293565374284e-8 * x**3
            + 9.45962838560648e-5 * x**2
            - 0.170812079190679   * x
            + 98.7232373648272
        )
    else:
        raise ValueError(f"PWM {pwm} out of valid range [1100..1900]")


def _solve_cubic(a, b, c, d):
    """
    Solve the cubic equation a*x^3 + b*x^2 + c*x + d = 0 for all roots.
    Return only the real roots, ignoring complex ones.
    """
    coeffs = np.array([a, b, c, d], dtype=float)
    roots = np.roots(coeffs)
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-12:
            real_roots.append(r.real)
    return real_roots

class PWMModel:
    def pwm_force_scalar(self, x_microseconds):
        """Original function (unchanged)."""
        x = x_microseconds / 1000.0
        if 1100 <= x < 1460:
            return (-1.24422882971549e-8)*x**3 \
                   + (4.02057100632393e-5)*x**2 \
                   - 0.0348619861030835*x        \
                   + 3.90671429105423
        elif 1460 <= x <= 1540:
            return 0
        elif 1540 < x <= 1900:
            return (-1.64293565374284e-8)*x**3 \
                   + (9.45962838560648e-5)*x**2  \
                   - 0.170812079190679*x        \
                   + 98.7232373648272
        else:
            raise ValueError(f"PWM {x_microseconds} out of valid range [1100000..1900000].")

    def force_to_pwm_scalar(self, force_val):
        """
        Inverse of pwm_force_scalar:
        Returns a PWM value in microseconds that yields `force_val`.
        Raises ValueError if there is no valid solution.
        """
        # --- Coefficients for the two cubic segments (SAME as original) ---
        # Segment A: x in [1100,1460)
        a1 = -1.24422882971549e-8
        b1 =  4.02057100632393e-5
        c1 = -0.0348619861030835
        d1 =  3.90671429105423

        # Segment C: x in (1540,1900]
        a2 = -1.64293565374284e-8
        b2 =  9.45962838560648e-5
        c2 = -0.170812079190679
        d2 =  98.7232373648272

        # Approximate min/max force from each segment
        f_min_1 = -2.36  # ~ p1(1100)
        f_max_1 =  0.0   # ~ p1(1460)
        f_min_2 =  0.12  # ~ p2(1540) from polynomial alone
        f_max_2 =  3.02  # ~ p2(1900)

        # 1) Check if force_val is out of possible range
        if force_val < f_min_1 or force_val > f_max_2:
            raise ValueError(f"force={force_val} is out of the valid range ~[{f_min_1}, {f_max_2}].")

        # 2) If in [-2.35,0), invert first cubic on [1100..1460)
        if f_min_1 <= force_val < 0:
            # Solve p1(x) = force_val  =>  a1*x^3 + b1*x^2 + c1*x + (d1 - force_val) = 0
            coeffs = [a1, b1, c1, (d1 - force_val)]
            roots = np.roots(coeffs)  # returns array of up to 3 roots (complex or real)
            # We need the real root in [1100..1460).
            for r in roots:
                if np.isreal(r):
                    r_real = r.real
                    if 1100 <= r_real < 1460:
                        return r_real * 1000  # convert back to microseconds

            # If no real root in range, no valid solution
            raise ValueError(f"No valid root in [1100,1460) for force={force_val}.")

        # 3) If force_val == 0, any x in [1460..1540] is valid. Choose 1500 by convention:
        if abs(force_val) < 1e-12:
            return 1500.0 * 1000  # or 1460*1000, 1540*1000, etc.

        # 4) If 0 < force_val < 0.12, no solution
        if 0 < force_val < f_min_2:
            return 1500.0 * 1000

        # 5) If in [0.12, 3.02], invert second cubic on (1540..1900]
        #    Solve p2(x) = force_val => a2*x^3 + b2*x^2 + c2*x + (d2 - force_val) = 0
        coeffs = [a2, b2, c2, (d2 - force_val)]
        roots = np.roots(coeffs)
        for r in roots:
            if np.isreal(r):
                r_real = r.real
                # The code piece is for x>1540..1900], so we check (1540,1900]
                # Strictly speaking, the condition is 1540 < x <= 1900.
                # We'll allow a tiny numerical tolerance around 1540.
                if 1540 < r_real <= 1900:
                    return r_real * 1000

        raise ValueError(f"No valid root in (1540,1900] for force={force_val}.")

def estimate_t(W, F):
    W_pinv = np.linalg.pinv(W)
    return W_pinv @ F
