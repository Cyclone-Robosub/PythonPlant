import numpy as np
import matplotlib.pyplot as plt
import math_stuff
from math_stuff import buoyant_force, weight_force

rev_pulse = 1100 * 1000
stop_pulse = 1500 * 1000
fwd_pulse_raw = 1900 * 1000  # Don't use this one, it's output can't be replicated in reverse
rev_adj = 0.97  # Thrusters are more powerful in fwd direction
fwd_pulse = int(fwd_pulse_raw * rev_adj)
frequency = 100
pwm_file = "pwm_file.csv"

zero_set = np.zeros(8, dtype=int)
stop_set = np.full(8, stop_pulse)
rev_set = np.concatenate((np.full(4, stop_pulse), np.full(4, fwd_pulse)))
fwd_set = np.concatenate((np.full(4, stop_pulse), np.full(4, rev_pulse)))
crab_set = np.concatenate((np.full(4, stop_pulse), [fwd_pulse, rev_pulse, rev_pulse, fwd_pulse]))
down_set = np.concatenate((np.full(4, rev_pulse), np.full(4, stop_pulse)))
barrell_set = np.concatenate(([rev_pulse, fwd_pulse, rev_pulse, fwd_pulse], np.full(4, stop_pulse)))
summer_set = np.concatenate(([rev_pulse, rev_pulse, fwd_pulse, fwd_pulse], np.full(4, stop_pulse)))
spin_set = np.concatenate((np.full(4, stop_pulse), [fwd_pulse, rev_pulse, fwd_pulse, rev_pulse]))
torpedo_set = np.concatenate(([rev_pulse, fwd_pulse, rev_pulse, fwd_pulse], np.full(4, fwd_pulse)))


class Plant:
    def __init__(self):

        # this pertains to iterative approximations done in simulation, Hz
        self.default_frequency = 100

        self.mass = 5.51
        self.height = 0.3 # z axis height
        self.mass_moment_of_inertia = np.array([1, 1, 1])
        self.six_axis_mass = np.full(6, 0)
        self.six_axis_mass[0:3] = np.full(3, self.mass)
        self.six_axis_mass[3:] = self.mass_moment_of_inertia

        self.six_axis_mass[0:3] = self.mass
        self.volume_inches = 450
        self.volume = self.volume_inches * pow(0.0254, 3)
        self.rho_water = 1000
        self.combined_drag_coefs = np.array([0.041, 0.05, 0.125, 0.05, 0.1, 0.05])
        self.combined_drag_coefs = [self.combined_drag_coefs[i] * 10 for i in range(len(self.combined_drag_coefs))]

        self.mass_center_inches = np.array([0, 0, 0])
        self.mass_center = self.mass_center_inches * 0.0254
        self.volume_center = np.array([0, 0, 0.1])

        #artifical max set so that we can do controls effectivly. Only relevant for controls
        self.min_force = np.array([-2, -2, -2, -2, -2, -2])
        self.max_force = np.array([2, 2, 2, 2, 2, 2])
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

        self.next_pwm = stop_set
        # state_log
        self.state_log = [{'time': 0,
                           'position': np.zeros(6),
                           'velocity': np.zeros(6),
                           'acceleration': np.zeros(6),
                           'totalForces': np.zeros(6),
                           'weightForces': np.zeros(6),
                           'buoyantForces': np.zeros(6),
                           'thrusterForces': np.zeros(6),
                           'pwm': stop_set}]

    def time(self):
        return self.state_log[-1]['time']
    def roll(self):
        return self.state_log[-1]['position'][3]
    def pitch(self):
        return self.state_log[-1]['position'][4]
    def yaw(self):
        return self.state_log[-1]['position'][5]
    def position(self):
        return self.state_log[-1]['position']
    def velocity(self):
        return self.state_log[-1]['velocity']
    def acceleration(self):
        return self.state_log[-1]['acceleration']
    def total_forces(self):
        return self.state_log[-1]['totalForces']
    def weight_forces(self):
        return self.state_log[-1]['weightForces']
    def buoyant_forces(self):
        return self.state_log[-1]['buoyantForces']
    def pwms(self):
        return self.state_log[-1]['pwm']

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
        self.next_pwm = pwm_set
    def weight_force(self):
        result = np.zeros(6)
        result[0:3] = math_stuff.weight_force(self.roll(), self.pitch(), self.yaw(), self.mass)
        return result
    def buoyant_force(self):
        result = math_stuff.buoyant_force(self.roll(), self.pitch(), self.yaw(), self.volume, self.volume_center)
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
                drag_force[i] =  - self.velocity()[i] * abs(self.velocity()[i]) * drag_coefs[i]
        return drag_force

    def total_force(self):
        weight = self.weight_force()
        buoyancy = self.buoyant_force()
        thrust = self.pwm_force(self.pwms())
        all_forces = weight + buoyancy + thrust
        return all_forces

    def simple_step(self):

        dt = 1/self.default_frequency
        position = np.zeros(6)
        velocity = np.zeros(6)
        acceleration = np.zeros(6)
        total_force = np.zeros(6)
        buoyant_force = np.zeros(6)
        weight_force = np.zeros(6)
        thrust = np.zeros(6)

        for i in range(6):
            Si = self.position()[i]
            Vi = self.velocity()[i]
            m = self.six_axis_mass[i]
            W = self.weight_force()[i]
            B = self.buoyant_force()[i]
            T = self.total_force()[i]
            thrust = self.pwm_force(self.pwms())[i]
            D = self.drag_force()[i]
            T += D
            A = T / m
            V = Vi + A * dt
            S = Si + V * dt
            position[i] = S
            velocity[i] = V
            acceleration[i] = A
            total_force[i] = T
            weight_force[i] = W
            buoyant_force[i] = B

        self.state_log.append({'time': self.time() + dt,
                               'position': position,
                               'velocity': velocity,
                               'acceleration': acceleration,
                               'totalForces': total_force,
                               'thrusterForces': thrust,
                               'weightForces': weight_force,
                               'buoyantForces': buoyant_force,
                               'pwm': self.next_pwm})

    def step(self):

        dt = 1/self.default_frequency
        position = np.zeros(6)
        velocity = np.zeros(6)
        acceleration = np.zeros(6)
        total_force = np.zeros(6)
        buoyant_force = np.zeros(6)
        weight_force = np.zeros(6)

        for i in range(6):
            Si = self.position()[i]
            Vi = self.velocity()[i]
            m = self.six_axis_mass[i]
            C = self.combined_drag_coefs[i]
            W = self.weight_force()[i]
            B = self.buoyant_force()[i]
            T = self.total_force()[i]
            A = T / m
            S, V = math_stuff.pos_vel(Vi, Si, m, C, T, dt)
            position[i] = S
            velocity[i] = V
            acceleration[i] = A
            total_force[i] = T
            weight_force[i] = W
            buoyant_force[i] = B

        self.state_log.append({'time': self.time() + dt,
                               'position': position,
                               'velocity': velocity,
                               'acceleration': acceleration,
                               'totalForces': total_force,
                               'weightForces': weight_force,
                               'buoyantForces': buoyant_force,
                               'pwm': self.next_pwm})
    def run_pwm(self, pwm_set, time):
        """This should work similarly to simulate_pwm, but does not need to produce graphs.
          Instead, it should simply append the state list of dictionaries for every period"""
        
        self.set_pwm(pwm_set)
        for i in range(time * self.default_frequency):
            self.simple_step()

    def print_dictionary(self):
        lengthOfDictionary = len(self.state_log)
        for i in range(lengthOfDictionary):
            print(f"Time: {self.state_log[i]['time']}\n")
            print(f"Position: {self.state_log[i]['position']}\n")
            print(f"Velocity: {self.state_log[i]['velocity']}\n")
            print(f"Acceleration: {self.state_log[i]['acceleration']}\n")
            print(f"Total Forces: {self.state_log[i]['totalForces']}\n")
            print(f"Weight Forces: {self.state_log[i]['weightForces']}\n")
            print(f"Buoyant Forces: {self.state_log[i]['buoyantForces']}\n")
            print(f"Pwm: {self.state_log[i]['pwm']}\n")
    def graph_acceleration(self):
        time = self.state_log[-1]['time']

        dt = 1/self.default_frequency
        time_steps = np.arange(0, time, dt)  # Generate time steps
        #These two functions create the craziest graphs ever. 
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("rad/s", color="tab:blue")
        ax1.plot(time_steps, [self.state_log[i]['acceleration'][0] for i in range(len(time_steps))], label="Position", color="tab:blue")
        ax1.plot(time_steps, [self.state_log[i]['acceleration'][1] for i in range(len(time_steps))], label="Position", color="tab:red")
        ax1.plot(time_steps, [self.state_log[i]['acceleration'][2] for i in range(len(time_steps))], label="Position", color="tab:green")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        plt.title("Linear Acceleration")
        plt.show(block=True)

        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("rad/s", color="tab:blue")
        ax1.plot(time_steps, [self.state_log[i]['acceleration'][3] for i in range(len(time_steps))], label="Position", color="tab:blue")
        ax1.plot(time_steps, [self.state_log[i]['acceleration'][4] for i in range(len(time_steps))], label="Position", color="tab:red")
        ax1.plot(time_steps, [self.state_log[i]['acceleration'][5] for i in range(len(time_steps))], label="Position", color="tab:green")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        fig.tight_layout()  # Adjust layout to fit both plots
        plt.title("Rotational Acceleration")
        plt.show(block=True)
    def graph_velocity(self):
        time = self.state_log[-1]['time']
        dt = 1/self.default_frequency
        time_steps = np.arange(0, time, dt)  # Generate time steps
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("m/s", color="tab:blue")
        ax1.plot(time_steps, [self.state_log[i]['velocity'][0] for i in range(len(time_steps))], label="X-Axis", color="tab:blue")
        ax1.plot(time_steps, [self.state_log[i]['velocity'][1] for i in range(len(time_steps))], label="Y-Axis", color="tab:red")
        ax1.plot(time_steps, [self.state_log[i]['velocity'][2] for i in range(len(time_steps))], label="Z-Axis", color="tab:green")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        plt.title("Linear Velocity")
        plt.legend(loc="upper left")

        plt.show(block=True)

        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("rad/s", color="tab:blue")
        ax1.plot(time_steps, [self.state_log[i]['velocity'][3] for i in range(len(time_steps))], label="X-Axis", color="tab:blue")
        ax1.plot(time_steps, [self.state_log[i]['velocity'][4] for i in range(len(time_steps))], label="Y-Axis", color="tab:red")
        ax1.plot(time_steps, [self.state_log[i]['velocity'][5] for i in range(len(time_steps))], label="Z-Axis", color="tab:green")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        plt.title("Rotational Velocity")
        plt.legend(loc="upper left")
        plt.show(block=True)
    def graph_position(self):
        time = self.state_log[-1]['time']

        dt = 1/self.default_frequency
        time_steps = np.arange(0, time, dt)  # Generate time steps

        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("m", color="tab:blue")
        ax1.plot(time_steps, [self.state_log[i]['position'][0] for i in range(len(time_steps))], label="X-Axis", color="tab:blue")
        ax1.plot(time_steps, [self.state_log[i]['position'][1] for i in range(len(time_steps))], label="Y-Axis", color="tab:red")
        ax1.plot(time_steps, [self.state_log[i]['position'][2] for i in range(len(time_steps))], label="Z-Axis", color="tab:green")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        plt.title("Linear Position")
        plt.legend(loc="upper left")
        plt.show(block=True)

        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Rads", color="tab:blue")
        # adjusted_rotation = math_stuff.wrap_to_half_pi(self.state_log[-1]['position'][3:])
        ax1.plot(time_steps, [self.state_log[i]['position'][3] for i in range(len(time_steps))], label="X-Axis", color="tab:blue")
        ax1.plot(time_steps, [self.state_log[i]['position'][4] for i in range(len(time_steps))], label="Y-Axis", color="tab:red")
        ax1.plot(time_steps, [self.state_log[i]['position'][5] for i in range(len(time_steps))], label="Z-Axis", color="tab:green")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        plt.title("Rotational Position")
        plt.legend(loc="upper left")

        plt.show(block=True)
    def graph_total_forces(self):
        time = self.state_log[-1]['time']

        dt = 1/self.default_frequency
        time_steps = np.arange(0, time, dt)  # Generate time steps
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("m", color="tab:blue")
        ax1.plot(time_steps, [self.state_log[i]['totalForces'][0] for i in range(len(time_steps))], label="X-Axis", color="tab:blue")
        ax1.plot(time_steps, [self.state_log[i]['totalForces'][1] for i in range(len(time_steps))], label="Y-Axis", color="tab:red")
        ax1.plot(time_steps, [self.state_log[i]['totalForces'][2] for i in range(len(time_steps))], label="Z-Axis", color="tab:green")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        #fig.tight_layout()  # Adjust layout to fit both plots
        plt.title("Linear Total Forces")
        plt.legend(loc="upper left")

        plt.show(block=True)

        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Rads", color="tab:blue")
        ax1.plot(time_steps, [self.state_log[i]['totalForces'][3] for i in range(len(time_steps))], label="X-Axis", color="tab:blue")
        ax1.plot(time_steps, [self.state_log[i]['totalForces'][4] for i in range(len(time_steps))], label="Y-Axis", color="tab:red")
        ax1.plot(time_steps, [self.state_log[i]['totalForces'][5] for i in range(len(time_steps))], label="Z-Axis", color="tab:green")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        plt.title("Rotational Total Forces")
        plt.legend(loc="upper left")
        plt.show(block=True)
    def graph_weight_forces(self):
        time = self.state_log[-1]['time']

        dt = 1/self.default_frequency
        time_steps = np.arange(0, time, dt)  # Generate time steps
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("m", color="tab:blue")
        ax1.plot(time_steps, [self.state_log[i]['weightForces'][0] for i in range(len(time_steps))], label="X-Axis", color="tab:blue")
        ax1.plot(time_steps, [self.state_log[i]['weightForces'][1] for i in range(len(time_steps))], label="Y-Axis", color="tab:red")
        ax1.plot(time_steps, [self.state_log[i]['weightForces'][2] for i in range(len(time_steps))], label="Z-Axis", color="tab:green")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        plt.title("Linear Weight Forces")
        plt.legend(loc="upper left")

        plt.show(block=True)

        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Rads", color="tab:blue")
        ax1.plot(time_steps, [self.state_log[i]['weightForces'][3] for i in range(len(time_steps))], label="X-Axis", color="tab:blue")
        ax1.plot(time_steps, [self.state_log[i]['weightForces'][4] for i in range(len(time_steps))], label="Y-Axis", color="tab:red")
        ax1.plot(time_steps, [self.state_log[i]['weightForces'][5] for i in range(len(time_steps))], label="Z-Axis", color="tab:green")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        plt.title("Rotational Weight Forces")
        plt.legend(loc="upper left")

        plt.show(block=True)
    def graph_buoyant_forces(self):
        time = self.state_log[-1]['time']

        dt = 1/self.default_frequency
        time_steps = np.arange(0, time, dt)  # Generate time steps

        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("m", color="tab:blue")
        ax1.plot(time_steps, [self.state_log[i]['buoyantForces'][0] for i in range(len(time_steps))], label="X-Axis", color="tab:blue")
        ax1.plot(time_steps, [self.state_log[i]['buoyantForces'][1] for i in range(len(time_steps))], label="Y-Axis", color="tab:red")
        ax1.plot(time_steps, [self.state_log[i]['buoyantForces'][2] for i in range(len(time_steps))], label="Z-Axis", color="tab:green")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        plt.title("Linear Buoyant Forces")
        plt.legend(loc="upper left")
        plt.show(block=True)

        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Rads", color="tab:blue")
        ax1.plot(time_steps, [self.state_log[i]['buoyantForces'][3] for i in range(len(time_steps))], label="X-Axis", color="tab:blue")
        ax1.plot(time_steps, [self.state_log[i]['buoyantForces'][4] for i in range(len(time_steps))], label="Y-Axis", color="tab:red")
        ax1.plot(time_steps, [self.state_log[i]['buoyantForces'][5] for i in range(len(time_steps))], label="Z-Axis", color="tab:green")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        plt.title("Rotational Buoyant Forces")
        plt.legend(loc="upper left")

        plt.show(block=True)
    def graph_pwm_signals(self):
        time = self.state_log[-1]['time']
        dt = 1/self.default_frequency
        time_steps = np.arange(0, time, dt)  # Generate time steps
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("m", color="tab:blue")
        ax1.plot(time_steps, [self.state_log[i]['pwm'][0] for i in range(len(time_steps))], label="Position", color="tab:blue")
        ax1.plot(time_steps, [self.state_log[i]['pwm'][1] for i in range(len(time_steps))], label="Position", color="tab:red")
        ax1.plot(time_steps, [self.state_log[i]['pwm'][2] for i in range(len(time_steps))], label="Position", color="tab:green")
        ax1.plot(time_steps, [self.state_log[i]['pwm'][3] for i in range(len(time_steps))], label="Position", color="tab:blue")
        ax1.plot(time_steps, [self.state_log[i]['pwm'][4] for i in range(len(time_steps))], label="Position", color="tab:red")
        ax1.plot(time_steps, [self.state_log[i]['pwm'][5] for i in range(len(time_steps))], label="Position", color="tab:green")
        ax1.plot(time_steps, [self.state_log[i]['pwm'][5] for i in range(len(time_steps))], label="Position", color="tab:blue")
        ax1.plot(time_steps, [self.state_log[i]['pwm'][5] for i in range(len(time_steps))], label="Position", color="tab:red")

        ax1.tick_params(axis="y", labelcolor="tab:blue")
        fig.tight_layout()  # Adjust layout to fit both plots
        plt.title("PWM Signals")
        plt.show(block=True)

def example1():
    plant = Plant()
    plant.run_pwm(fwd_set,10)
    plant.run_pwm(rev_set,10)
    plant.graph_position()
    plant.graph_velocity()
    plant.graph_buoyant_forces()
    plant.graph_total_forces()
