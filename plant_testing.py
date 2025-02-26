import numpy as np
import math_stuff as ms
import Plant as pl
import unittest

from math_stuff import buoyant_force


class TestPlant(unittest.TestCase):
    def test_buoyant_force(self):
        plant = pl.Plant()
        plant.state_log[-1] = {'time': 0,
                           'position': np.array([0, 0, 0, 0, 0.1, 0]),
                           'velocity': np.zeros(6),
                           'acceleration': np.zeros(6),
                           'totalForces': np.zeros(6),
                           'weightForces': np.zeros(6),
                           'buoyantForces': np.zeros(6),
                           'pwm': pl.stop_set}
        buoyant_force = plant.buoyant_force()
        self.assertTrue(buoyant_force[4] < 0)
        self.assertTrue(buoyant_force[2] > 0)

    def test_run_pwm(self):
        plant = pl.Plant()
        plant.run_pwm(pl.fwd_set, 10)
        for log in plant.state_log:
            for i in range(len(log['position'])):
                self.assertTrue(log['velocity'][i] * log['totalForces'][i] >= 0,
                                msg=f"\ni: {i} FAILED\nLOG: {log}\n")


