"""
Comprehensive Unit Tests for Linear Algebra & Sequential Pipeline
Tests verify mathematical invariants, vector/matrix norms, QR decomposition,
and projection matrix algebraic properties.
"""

import math
import unittest
from src.linear_algebra import (
    Vector,
    Matrix,
    solve_linear_system,
    qr_decomposition,
    solve_qr_least_squares,
    orthogonal_projection_matrix,
)
from src.pipeline import HousePriceLeastSquaresPipeline


class TestVectorOperations(unittest.TestCase):
    def setUp(self):
        self.u = Vector([1.0, 2.0, -3.0])
        self.v = Vector([4.0, -1.0, 2.0])

    def test_addition_and_subtraction(self):
        w = self.u + self.v
        self.assertEqual(w.elements, [5.0, 1.0, -1.0])
        diff = self.u - self.v
        self.assertEqual(diff.elements, [-3.0, 3.0, -5.0])

    def test_dot_product(self):
        # 1*4 + 2*(-1) + (-3)*2 = 4 - 2 - 6 = -4
        self.assertAlmostEqual(self.u.dot(self.v), -4.0)

    def test_norms(self):
        # L1: |1| + |2| + |-3| = 6
        self.assertAlmostEqual(self.u.norm_l1(), 6.0)
        # L2: sqrt(1 + 4 + 9) = sqrt(14)
        self.assertAlmostEqual(self.u.norm_l2(), math.sqrt(14.0))
        # Linf: max(1, 2, 3) = 3
        self.assertAlmostEqual(self.u.norm_inf(), 3.0)


class TestMatrixOperations(unittest.TestCase):
    def test_transpose_and_multiplication(self):
        A = Matrix([[1, 2, 3], [4, 5, 6]])
        At = A.transpose()
        self.assertEqual(At.shape(), (3, 2))
        self.assertEqual(At.data, [[1, 4], [2, 5], [3, 6]])

        AtA = At @ A
        self.assertEqual(AtA.shape(), (3, 3))
        # Verification of symmetric property (A^T A)^T = A^T A
        self.assertAlmostEqual((AtA.transpose() - AtA).norm_frobenius(), 0.0)

    def test_matrix_inversion(self):
        M = Matrix([[4, 7], [2, 6]])
        M_inv = M.inverse()
        # M * M_inv == I
        prod = M @ M_inv
        I = Matrix.identity(2)
        diff = (prod - I).norm_frobenius()
        self.assertLess(diff, 1e-10)

    def test_solve_linear_system(self):
        A = Matrix([[3, 2, -1], [2, -2, 4], [-1, 0.5, -1]])
        b = Vector([1, -2, 0])
        x = solve_linear_system(A, b)
        # Verify Ax == b
        Ax = A @ x
        for ax_i, b_i in zip(Ax.elements, b.elements):
            self.assertAlmostEqual(ax_i, b_i, places=6)


class TestOrthogonalProjections(unittest.TestCase):
    def setUp(self):
        # Overdetermined system A (4 x 2)
        self.A = Matrix([[1, 1], [1, 2], [1, 3], [1, 4]])
        self.y = Vector([1.5, 2.8, 3.9, 5.1])

    def test_projection_matrix_properties(self):
        P = orthogonal_projection_matrix(self.A)
        # 1. Symmetry: P^T = P
        sym_diff = (P.transpose() - P).norm_frobenius()
        self.assertLess(sym_diff, 1e-12)

        # 2. Idempotency: P^2 = P
        P_sq = P @ P
        idem_diff = (P_sq - P).norm_frobenius()
        self.assertLess(idem_diff, 1e-12)

    def test_least_squares_orthogonality(self):
        P = orthogonal_projection_matrix(self.A)
        y_hat = P @ self.y
        e = self.y - y_hat

        # Residual must be orthogonal to Col(A): A^T e = 0
        At_e = self.A.transpose() @ e
        self.assertLess(At_e.norm_l2(), 1e-10)

        # Fitted values y_hat and residual e must be orthogonal: y_hat . e = 0
        self.assertAlmostEqual(y_hat.dot(e), 0.0, places=10)

        # Pythagorean theorem: ||y||^2 = ||y_hat||^2 + ||e||^2
        norm_y_sq = self.y.norm_l2() ** 2
        norm_sum_sq = (y_hat.norm_l2() ** 2) + (e.norm_l2() ** 2)
        self.assertAlmostEqual(norm_y_sq, norm_sum_sq, places=10)

    def test_qr_vs_normal_equations(self):
        beta_qr = solve_qr_least_squares(self.A, self.y)
        AtA_inv = (self.A.transpose() @ self.A).inverse()
        beta_normal = AtA_inv @ (self.A.transpose() @ self.y)

        diff = (beta_qr - beta_normal).norm_l2()
        self.assertLess(diff, 1e-10)


class TestSequentialPipeline(unittest.TestCase):
    def test_pipeline_execution(self):
        pipeline = HousePriceLeastSquaresPipeline("data/houses.csv")
        evaluations = pipeline.execute_full_pipeline()
        self.assertEqual(len(evaluations), 3)

        # Verify that all 8 stages were recorded in order
        self.assertEqual(len(pipeline.stage_history), 8)
        for i, stage in enumerate(pipeline.stage_history, 1):
            self.assertEqual(stage.stage_number, i)

        # Verify best model has highest R^2 and lowest RMSE
        best = evaluations[0]
        self.assertGreater(best.r_squared, 0.99)
        self.assertLess(best.rmse, 10.0)
        self.assertLess(best.orthogonality_norm, 1e-4)


if __name__ == "__main__":
    unittest.main()
