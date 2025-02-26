import numpy as np
from Plant import Plant
from math_stuff import *

class PID:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.plant = Plant()
        self.pwm_model = PWMModel()

        self.error = [[0,0,0,0,0,0]]
        self.integral_error = [0,0,0,0,0,0]
        self.derivative_error = [0,0,0,0,0,0]


    def fwd_step(self, v, t):
        return np.array([t*v, 0, 0, 0, 0, 0])

    def thrust_compute(self, force):
        thrusters = estimate_t(self.plant.wrench_matrix_transposed, force)
        pwms = [self.pwm_model.force_to_pwm_scalar(thruster) for thruster in thrusters]
        return pwms

    def compute_pos_error(self, ref, v):
        t = self.plant.time()
        desired = ref(v, t)
        actual = self.plant.position()
        error = desired - actual
        return error

    def correct_error(self):
        forces = np.zeros_like(self.error[-1])
        for i in range(6):
            forces[i] += self.error[-1][i] * self.Kp
            forces[i] += self.integral_error[i] * self.Ki
            forces[i] += self.derivative_error[i] * self.Kd
        for i in range(6):
            if forces[i] > self.plant.max_force[i]:
                forces[i] = self.plant.max_force[i]
            if i < self.plant.min_force[i]:
                forces[i] = self.plant.min_force[i]
        pwms = self.thrust_compute(forces)
        return pwms


    def linear_travel(self, v, t):
        return [i * t for i in v]

    def linear_run(self, ref, v, t):

        # start by reseting plant. This will change once initial conditions work
        self.plant = Plant()
        f = self.plant.default_frequency
        dt = 1 / f
        current_time = 0
        for i in range(t* f):

            error = self.compute_pos_error(ref, v)
            self.derivative_error = (error - self.error[-1]) / dt
            self.error.append(error)
            self.integral_error += error
            try:
                correction = self.correct_error()
                self.plant.set_pwm(correction)
            except:
                print("oh shit!")
                pass

            print(f"i: {i}current_time: {current_time}\n"
                  f"error: {error}\n"
                  f"correction: {correction}\n")
            self.plant.simple_step()
            current_time += dt


pid = PID(Kp = 100, Ki = 10, Kd = 10)
pid.linear_run(pid.linear_travel, [0.1,0,0,0,0,0], 10)

pid.plant.graph_position()
pid.plant.graph_velocity()
pid.plant.graph_total_forces()
pid.plant.graph_pwm_signals()
