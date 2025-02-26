import math
import numpy as np
import math_stuff as ms
import unittest
import math_stuff

# Assuming your function is defined as:
# from your_module import rotation_matrix

def is_orthogonal(matrix, tol=1e-6):
    """Check if a matrix is orthogonal (R * R.T = I)."""
    I = np.eye(matrix.shape[0])
    return np.allclose(matrix @ matrix.T, I, atol=tol)

class TestRotationMatrix(unittest.TestCase):
    def test_identity_rotation(self):
        """Test that a (0, 0, 0) rotation results in an identity matrix."""
        R = ms.rotation_matrix(0, 0, 0)
        np.testing.assert_array_almost_equal(R, np.eye(3), decimal=6)

    def test_orthogonality(self):
        """Test that the rotation matrix is orthogonal."""
        angles = [(0, 0, 0), (np.pi/4, 0, 0), (0, np.pi/4, 0), (0, 0, np.pi/4),
                  (np.pi/2, np.pi/2, np.pi/2), (-np.pi/3, np.pi/3, -np.pi/3)]
        for roll, pitch, yaw in angles:
            R = ms.rotation_matrix(roll, pitch, yaw)
            self.assertTrue(is_orthogonal(R), f"Matrix is not orthogonal for angles {(roll, pitch, yaw)}")

    def test_determinant(self):
        """Test that the determinant of the rotation matrix is 1 (property of SO(3) matrices)."""
        angles = [(0, 0, 0), (np.pi/4, 0, 0), (0, np.pi/4, 0), (0, 0, np.pi/4),
                  (np.pi/2, np.pi/2, np.pi/2), (-np.pi/3, np.pi/3, -np.pi/3)]
        for roll, pitch, yaw in angles:
            R = ms.rotation_matrix(roll, pitch, yaw)
            det = np.linalg.det(R)
            self.assertAlmostEqual(det, 1.0, places=6, msg=f"Determinant is not 1 for angles {(roll, pitch, yaw)}")

class TestWeightForce(unittest.TestCase):
    def test_no_rotation(self):
        """Test weight force when roll, pitch, and yaw are zero."""
        mass = 10  # kg
        expected_force = np.array([0, 0, -mass * 9.81])  # Only along -Z
        computed_force = ms.weight_force(0, 0, 0, mass)
        np.testing.assert_array_almost_equal(computed_force, expected_force, decimal=6)

    def test_orthogonality_preservation(self):
        """Test that the weight force maintains its magnitude after transformation."""
        mass = 5  # kg
        angles = [(np.pi/6, np.pi/4, np.pi/3), (-np.pi/3, np.pi/6, -np.pi/4), (np.pi/2, 0, np.pi)]
        expected_magnitude = mass * 9.81

        for roll, pitch, yaw in angles:
            computed_force = ms.weight_force(roll, pitch, yaw, mass)
            magnitude = np.linalg.norm(computed_force)
            self.assertAlmostEqual(magnitude, expected_magnitude, places=6,
                                   msg=f"Incorrect magnitude for angles {(roll, pitch, yaw)}")

    def test_known_rotation_cases(self):
        """Test against manually computed force transformations."""
        mass = 2  # kg
        g = 9.81

        test_cases = [
            # (roll, pitch, yaw) -> Expected force components (Fx, Fy, Fz)
            (0, 0, 0, np.array([0, 0, -mass * g])),
            (np.pi/2, 0, 0, np.array([0, mass * g, 0])),  # 90° Roll -> Gravity along Y
            (0, np.pi/2, 0, np.array([- mass * g, 0, 0])),  # 90° Pitch -> Gravity along X
            (0, 0, np.pi/2, np.array([0, 0, -mass * g])),  # 90° Yaw -> No effect on gravity
        ]

        for roll, pitch, yaw, expected_force in test_cases:
            computed_force = ms.weight_force(roll, pitch, yaw, mass)
            np.testing.assert_array_almost_equal(computed_force, expected_force, decimal=6,
                                                 err_msg=f"Failed for angles {(roll, pitch, yaw)}")


class TestBuoyantForce(unittest.TestCase):

    def test_no_rotation(self):
        """Test that buoyant force acts only in the Z direction when no rotation is applied."""
        volume = 1.0  # m³
        d = np.array([0.1, -0.05, -0.2])  # Arbitrary displacement vector
        expected_force = np.array([0, 0, 1000 * 9.81 * volume])  # Buoyant force should act in +Z
        result = ms.buoyant_force(0, 0, 0, volume, d)

        np.testing.assert_array_almost_equal(result[:3], expected_force, decimal=6,
                                             err_msg="Incorrect force in no rotation case")

    def test_torque_calculation(self):
        """Test that torque is correctly computed using the cross product."""
        volume = 0.5  # m³
        d = np.array([0.2, -0.1, -0.3])  # Displacement vector
        expected_force = np.array([0, 0, 1000 * 9.81 * volume])  # Buoyancy acts in +Z direction

        # Expected torque τ = d × F_buoyancy
        expected_torque = np.cross(d,  expected_force)

        result = ms.buoyant_force(0, 0, 0, volume, d)

        np.testing.assert_array_almost_equal(result[3:], expected_torque, decimal=6,
                                             err_msg="Incorrect torque calculation")

    def test_known_rotation_cases(self):
        """Test against known rotation transformations."""
        volume = 1.0  # m³
        d = np.array([0, 0, 0])  # Displacement vector

        test_cases = [
            # (roll, pitch, yaw) -> Expected force and torque
            (0, 0, 0, np.array([0, 0, 1000 * 9.81 * volume])),
            (np.pi / 2, 0, 0, np.array([0, -1000 * 9.81 * volume, 0])),  # 90° Roll -> Buoyancy along Y
            (0, np.pi / 2, 0, np.array([1000 * 9.81 * volume, 0, 0])),  # 90° Pitch -> Buoyancy along X
            (0, 0, np.pi / 2, np.array([0, 0, 1000 * 9.81 * volume])),  # 90° Yaw -> No effect on buoyancy
        ]

        for roll, pitch, yaw, expected_force in test_cases:
            result = ms.buoyant_force(roll, pitch, yaw, volume, d)
            np.testing.assert_array_almost_equal(result[:3], expected_force, decimal=6,
                                                 err_msg=f"Failed force test for angles {(roll, pitch, yaw)}")

if __name__ == '__main__':
    unittest.main()
