import numpy as np
import math_stuff

rev_pulse = 1100 * 1000
stop_pulse = 1500 * 1000
fwd_pulse_raw = 1900 * 1000  # Don't use this one, it's output can't be replicated in reverse
rev_adj = 0.97  # Thrusters are more powerful in fwd direction
fwd_pulse = int(fwd_pulse_raw * rev_adj)
frequency = 10
pwm_file = "pwm_file.csv"

zero_set = np.zeros(8, dtype=int)
stop_set = np.full(8, stop_pulse)
fwd_set = np.concatenate((np.full(4, stop_pulse), np.full(4, fwd_pulse)))
crab_set = np.concatenate((np.full(4, stop_pulse), [fwd_pulse, rev_pulse, rev_pulse, fwd_pulse]))
down_set = np.concatenate((np.full(4, rev_pulse), np.full(4, stop_pulse)))
barrell = np.concatenate(([rev_pulse, fwd_pulse, rev_pulse, fwd_pulse], np.full(4, stop_pulse)))
summer = np.concatenate(([rev_pulse, rev_pulse, fwd_pulse, fwd_pulse], np.full(4, stop_pulse)))
spin = np.concatenate((np.full(4, stop_pulse), [fwd_pulse, rev_pulse, fwd_pulse, rev_pulse]))
torpedo = np.concatenate(([rev_pulse, fwd_pulse, rev_pulse, fwd_pulse], np.full(4, fwd_pulse)))


class Plant:
    def __init__(self):
        self.mass = 5.51
        self.mass_moment_of_inertia = np.array([0, 0, 0])
        self.six_axis_mass = np.full(6, 0)
        self.six_axis_mass[0:3] = np.full(3, self.mass)
        self.six_axis_mass[3:] = self.mass_moment_of_inertia

        self.six_axis_mass[0:3] = self.mass
        self.volume_inches = 449.157
        self.volume = self.volume_inches * pow(0.0254, 3)
        self.rho_water = 1000
        self.combined_drag_coefs = np.array([0.041, 0.05, 0.125, 0.005, 0.005, 0.005]) * self.rho_water

        self.current_pwms = stop_set
        self.current_position = np.array([0, 0, 0, 0, 0, 0])
        self.current_velocity = np.array([0, 0, 0, 0, 0, 0])
        self.current_acceleration = np.array([0, 0, 0, 0, 0, 0])

        self.mass_center_inches = np.array([0.466, 0, 1.561])
        self.mass_center = self.mass_center_inches * 0.0254

        self.volume_center = np.array([0, 0, 0.1])

        # Thruster positions
        self.thruster_positions = np.array([
            [0.2535, -0.2035, 0.042],
            [0.2535, 0.2035, 0.042],
            [-0.2545, -0.2035, 0.042],
            [-0.2545, 0.2035, 0.042],
            [0.1670, -0.1375, -0.049],
            [0.1670, 0.1375, -0.049],
            [-0.1975, -0.1165, -0.049],
            [-0.1975, 0.1165, -0.049]
        ])

        # Thruster directions
        sin45 = np.sin(np.pi / 4)
        self.thruster_directions = np.array([
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [-sin45, -sin45, 0],
            [-sin45, sin45, 0],
            [-sin45, sin45, 0],
            [-sin45, -sin45, 0]
        ])

        # Thruster torques (cross product of position and direction)
        self.thruster_torques = np.cross(self.thruster_positions, self.thruster_directions)

        # Compute wrench matrix (6x8)
        self.wrench_matrix_transposed = np.hstack((self.thruster_directions, self.thruster_torques)).T

        # Transpose to get wrench matrix (6x8)
        self.wrench_matrix = self.wrench_matrix_transposed.T

        # state_log
        self.state_Log = [{'time': 0,
                           'position': self.current_position,
                           'velocity': self.current_velocity,
                           'acceleration': self.current_acceleration,
                           'pwm': self.current_pwms}]


    def pwm_force_scalar(self, x):
        x = x / 1000
        if 1100 <= x < 1460:
            force = (-1.24422882971549e-8) * x ** 3 + (
                4.02057100632393e-5) * x ** 2 - 0.0348619861030835 * x + 3.90671429105423
        elif 1460 <= x <= 1540:
            force = 0
        elif 1540 < x <= 1900:
            force = (-1.64293565374284e-8) * x ** 3 + (
                9.45962838560648e-5) * x ** 2 - 0.170812079190679 * x + 98.7232373648272
        else:
            raise ValueError(f"PWM value {x} out of valid range (1100-1900)")
        return force

    def pwm_force(self, pwm_set):
        thruster_forces = np.array([self.pwm_force_scalar(pwm) for pwm in pwm_set])
        force = np.dot(thruster_forces, self.wrench_matrix)
        return force

    def set_pwm(self, pwm_set):
        self.current_pwms = pwm_set
        thrusters = np.array([self.pwm_force_scalar(pwm) for pwm in pwm_set])
        thruster_forces = np.dot(self.wrench_matrix, thrusters)
        weight = self.weight_force()
        boyancy = self.boyant_force()

    def weight_force(self):
        mass = self.mass
        g = 9.81
        weight_magnitude = - mass * g

        orientation = self.current_position[3:]
        result = np.zeros(6)

        # reference weight (z direction)
        result[0:3] = weight_magnitude * np.array([0, 0, 1])

        R = math_stuff.rotation_matrix(orientation[0], orientation[1], orientation[2])

        result[0:3] = R @ result[0:3]
        return result

    def boyant_force(self):
        volume = self.volume
        g = 9.81
        rho = self.rho_water
        magnitude = volume * rho * g

        orientation = self.current_position[3:]
        result = np.zeros(6)

        # reference weight (z direction)
        result[0:3] = magnitude * np.array([0, 0, 1])

        R = math_stuff.rotation_matrix(orientation[0], orientation[1], orientation[2])

        result[0:3] = R @ result[0:3]

        result[3:] = np.cross(self.volume_center, result[0:3])

        return result

    def drag_force(self):
        """explanation on notion:
        https: // www.notion.so / crsucd /
        Rotational - drag - analysis - 1478
        a3eca2f0801d86f2e0c8fb675c0d
        these values are estimates and should be improved experimentally"""

        drag_coefs = self.combined_drag_coefs
        drag_force = np.zeros(6)

        for i in range(6):
                drag_force[i] = self.current_velocity[i] * abs(self.current_velocity[i]) * drag_coefs[i]

        return drag_force

    def total_force(self):
        weight = self.weight_force()
        boyancy = self.boyant_force()
        drag = self.drag_force()
        thrust = self.pwm_force(self.current_pwms)

        all_forces = weight + boyancy + drag + thrust
        return all_forces











plant = Plant()
print(plant.weight_force())
print(plant.boyant_force())
plant.current_position = np.array([0, 0, 0, np.pi/2, 0, 0])
print(plant.weight_force())
print(plant.boyant_force())
print("fucker!\n")
print(plant.total_force())

