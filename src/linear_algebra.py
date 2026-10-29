"""
Linear Algebra & Vector Calculus Module
Author: Linear Algebra Mini-Project Team
Description: Pure-Python linear algebra engine implementing foundational concepts:
             Vectors, Matrices, Linear Systems, QR Decomposition,
             Orthogonal Projections, and Vector/Matrix Norms.
"""

import math
from typing import List, Tuple, Union

Number = Union[int, float]


class Vector:
    """Represents an n-dimensional vector in R^n with fundamental vector calculus operations."""

    def __init__(self, elements: List[Number]):
        self.elements = [float(x) for x in elements]
        self.dim = len(self.elements)

    def __len__(self) -> int:
        return self.dim

    def __getitem__(self, idx: int) -> float:
        return self.elements[idx]

    def __setitem__(self, idx: int, value: Number):
        self.elements[idx] = float(value)

    def __repr__(self) -> str:
        formatted = ", ".join(f"{x:.4f}" for x in self.elements[:6])
        if self.dim > 6:
            formatted += f", ... ({self.dim} elements total)"
        return f"Vector([{formatted}])"

    def to_list(self) -> List[float]:
        return list(self.elements)

    def __add__(self, other: 'Vector') -> 'Vector':
        if self.dim != other.dim:
            raise ValueError(f"Dimension mismatch for addition: {self.dim} != {other.dim}")
        return Vector([a + b for a, b in zip(self.elements, other.elements)])

    def __sub__(self, other: 'Vector') -> 'Vector':
        if self.dim != other.dim:
            raise ValueError(f"Dimension mismatch for subtraction: {self.dim} != {other.dim}")
        return Vector([a - b for a, b in zip(self.elements, other.elements)])

    def __mul__(self, scalar: Number) -> 'Vector':
        return Vector([x * float(scalar) for x in self.elements])

    def __rmul__(self, scalar: Number) -> 'Vector':
        return self.__mul__(scalar)

    def dot(self, other: 'Vector') -> float:
        """Inner product (dot product) <u, v> = u^T v."""
        if self.dim != other.dim:
            raise ValueError(f"Dimension mismatch for dot product: {self.dim} != {other.dim}")
        return sum(a * b for a, b in zip(self.elements, other.elements))

    def norm_l1(self) -> float:
        """L1 norm (Manhattan norm): ||v||_1 = sum(|v_i|)."""
        return sum(abs(x) for x in self.elements)

    def norm_l2(self) -> float:
        """L2 norm (Euclidean norm): ||v||_2 = sqrt(sum(v_i^2))."""
        return math.sqrt(sum(x * x for x in self.elements))

    def norm_inf(self) -> float:
        """L_infinity norm (Chebyshev / Maximum norm): ||v||_inf = max(|v_i|)."""
        return max(abs(x) for x in self.elements) if self.elements else 0.0

    def mean(self) -> float:
        return sum(self.elements) / self.dim if self.dim > 0 else 0.0

    def variance(self) -> float:
        m = self.mean()
        return sum((x - m) ** 2 for x in self.elements) / (self.dim - 1) if self.dim > 1 else 0.0

    def std(self) -> float:
        return math.sqrt(self.variance())

    def normalize(self) -> 'Vector':
        """Returns unit vector v / ||v||_2."""
        l2 = self.norm_l2()
        if l2 == 0:
            raise ValueError("Cannot normalize a zero vector.")
        return Vector([x / l2 for x in self.elements])


class Matrix:
    """Represents an m x n matrix in R^{m x n} with standard linear algebraic operations."""

    def __init__(self, data: List[List[Number]]):
        if not data or not data[0]:
            raise ValueError("Matrix cannot be empty.")
        self.rows = len(data)
        self.cols = len(data[0])
        # Validate rectangular shape
        for r in data:
            if len(r) != self.cols:
                raise ValueError("All rows in Matrix must have equal length.")
        self.data = [[float(val) for val in row] for row in data]

    @classmethod
    def zeros(cls, rows: int, cols: int) -> 'Matrix':
        return cls([[0.0] * cols for _ in range(rows)])

    @classmethod
    def identity(cls, n: int) -> 'Matrix':
        data = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        return cls(data)

    @classmethod
    def from_columns(cls, cols: List[Vector]) -> 'Matrix':
        """Constructs matrix A where columns are the given Vectors."""
        if not cols:
            raise ValueError("Column list cannot be empty.")
        m = cols[0].dim
        for c in cols:
            if c.dim != m:
                raise ValueError("All column vectors must have identical dimension.")
        data = [[cols[j][i] for j in range(len(cols))] for i in range(m)]
        return cls(data)

    def shape(self) -> Tuple[int, int]:
        return (self.rows, self.cols)

    def __getitem__(self, idx: int) -> List[float]:
        return self.data[idx]

    def __repr__(self) -> str:
        rows_str = []
        for i in range(min(5, self.rows)):
            row_preview = ", ".join(f"{x:.4f}" for x in self.data[i][:5])
            if self.cols > 5:
                row_preview += ", ..."
            rows_str.append(f"  [{row_preview}]")
        if self.rows > 5:
            rows_str.append(f"  ... ({self.rows}x{self.cols} total)")
        return "Matrix([\n" + ",\n".join(rows_str) + "\n])"

    def get_column(self, col_idx: int) -> Vector:
        """Extracts column as a Vector."""
        return Vector([self.data[i][col_idx] for i in range(self.rows)])

    def get_row(self, row_idx: int) -> Vector:
        """Extracts row as a Vector."""
        return Vector(self.data[row_idx])

    def transpose(self) -> 'Matrix':
        """Computes matrix transpose A^T."""
        transposed_data = [[self.data[i][j] for i in range(self.rows)] for j in range(self.cols)]
        return Matrix(transposed_data)

    def __add__(self, other: 'Matrix') -> 'Matrix':
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError(f"Shape mismatch: {self.shape()} vs {other.shape()}")
        data = [
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ]
        return Matrix(data)

    def __sub__(self, other: 'Matrix') -> 'Matrix':
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError(f"Shape mismatch: {self.shape()} vs {other.shape()}")
        data = [
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ]
        return Matrix(data)

    def __mul__(self, scalar: Number) -> 'Matrix':
        return Matrix([[x * float(scalar) for x in row] for row in self.data])

    def __rmul__(self, scalar: Number) -> 'Matrix':
        return self.__mul__(scalar)

    def matmul(self, other: Union['Matrix', Vector]) -> Union['Matrix', Vector]:
        """Matrix multiplication A @ B or A @ v."""
        if isinstance(other, Vector):
            if self.cols != other.dim:
                raise ValueError(f"Cannot multiply {self.shape()} by vector of dim {other.dim}")
            result = [
                sum(self.data[i][k] * other[k] for k in range(self.cols))
                for i in range(self.rows)
            ]
            return Vector(result)
        elif isinstance(other, Matrix):
            if self.cols != other.rows:
                raise ValueError(f"Cannot multiply shapes {self.shape()} and {other.shape()}")
            result = [
                [
                    sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                    for j in range(other.cols)
                ]
                for i in range(self.rows)
            ]
            return Matrix(result)
        else:
            raise TypeError("Unsupported operand for matmul.")

    def __matmul__(self, other: Union['Matrix', Vector]) -> Union['Matrix', Vector]:
        return self.matmul(other)

    def norm_frobenius(self) -> float:
        """Frobenius norm ||A||_F = sqrt(sum(A_ij^2))."""
        return math.sqrt(sum(val * val for row in self.data for val in row))

    def norm_1(self) -> float:
        """Maximum absolute column sum norm."""
        return max(sum(abs(self.data[i][j]) for i in range(self.rows)) for j in range(self.cols))

    def norm_inf(self) -> float:
        """Maximum absolute row sum norm."""
        return max(sum(abs(val) for val in row) for row in self.data)

    def inverse(self) -> 'Matrix':
        """Computes inverse of square matrix A^{-1} via Gauss-Jordan elimination with partial pivoting."""
        if self.rows != self.cols:
            raise ValueError(f"Only square matrices have an inverse. Matrix shape: {self.shape()}")
        n = self.rows
        # Augmented matrix [A | I]
        aug = [self.data[i] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

        for col in range(n):
            # Partial pivoting for numerical stability
            max_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
            if abs(aug[max_row][col]) < 1e-12:
                raise ValueError("Matrix is singular or near-singular (non-invertible).")
            # Swap rows
            aug[col], aug[max_row] = aug[max_row], aug[col]

            # Scale pivot row to 1
            pivot = aug[col][col]
            for j in range(2 * n):
                aug[col][j] /= pivot

            # Eliminate in other rows
            for r in range(n):
                if r != col:
                    factor = aug[r][col]
                    for j in range(col, 2 * n):
                        aug[r][j] -= factor * aug[col][j]

        inv_data = [[aug[i][j + n] for j in range(n)] for i in range(n)]
        return Matrix(inv_data)


def solve_linear_system(A: Matrix, b: Vector) -> Vector:
    """
    Solves square linear system A x = b using Gaussian elimination with back-substitution.
    """
    if A.rows != A.cols:
        raise ValueError("Matrix must be square to solve A x = b directly.")
    if A.rows != b.dim:
        raise ValueError(f"Dimension mismatch between A ({A.rows}) and b ({b.dim}).")

    n = A.rows
    # Augmented matrix [A | b]
    aug = [A.data[i] + [b.elements[i]] for i in range(n)]

    # Forward elimination with partial pivoting
    for i in range(n):
        pivot_row = max(range(i, n), key=lambda r: abs(aug[r][i]))
        if abs(aug[pivot_row][i]) < 1e-12:
            raise ValueError("Matrix is singular; system does not have a unique solution.")
        aug[i], aug[pivot_row] = aug[pivot_row], aug[i]

        for k in range(i + 1, n):
            factor = aug[k][i] / aug[i][i]
            for j in range(i, n + 1):
                aug[k][j] -= factor * aug[i][j]

    # Back substitution
    x = [0.0] * n
    for i in reversed(range(n)):
        total = aug[i][n]
        for j in range(i + 1, n):
            total -= aug[i][j] * x[j]
        x[i] = total / aug[i][i]

    return Vector(x)


def qr_decomposition(A: Matrix) -> Tuple[Matrix, Matrix]:
    """
    Computes thin QR decomposition A = Q R using Modified Gram-Schmidt orthogonalization.
    Q has orthonormal columns (Q^T Q = I_n), R is upper-triangular n x n.
    """
    m, n = A.shape()
    v_cols = [A.get_column(j).elements[:] for j in range(n)]
    q_cols = [[0.0] * m for _ in range(n)]
    R_data = [[0.0] * n for _ in range(n)]

    for j in range(n):
        v = Vector(v_cols[j])
        for i in range(j):
            q_i = Vector(q_cols[i])
            r_ij = q_i.dot(v)
            R_data[i][j] = r_ij
            v = v - (q_i * r_ij)
        r_jj = v.norm_l2()
        if r_jj < 1e-12:
            raise ValueError(f"Columns of design matrix are linearly dependent at column {j}.")
        R_data[j][j] = r_jj
        q_j = v * (1.0 / r_jj)
        q_cols[j] = q_j.elements[:]

    # Assemble Q from column vectors
    Q_data = [[q_cols[j][i] for j in range(n)] for i in range(m)]
    return Matrix(Q_data), Matrix(R_data)


def solve_qr_least_squares(A: Matrix, y: Vector) -> Vector:
    """
    Solves least-squares problem min ||A beta - y||_2 via QR decomposition.
    A = Q R  ==>  R beta = Q^T y.
    Solved by back substitution since R is upper triangular.
    """
    Q, R = qr_decomposition(A)
    # Q^T @ y
    qty = Q.transpose().matmul(y)
    n = R.cols
    # Back substitution
    beta = [0.0] * n
    for i in reversed(range(n)):
        s = sum(R.data[i][j] * beta[j] for j in range(i + 1, n))
        beta[i] = (qty[i] - s) / R.data[i][i]
    return Vector(beta)


def orthogonal_projection_matrix(A: Matrix) -> Matrix:
    """
    Computes orthogonal projection matrix P onto the column space Col(A):
    P = A (A^T A)^{-1} A^T.
    Properties: P^2 = P (idempotent), P^T = P (symmetric).
    """
    At = A.transpose()
    AtA = At.matmul(A)
    AtA_inv = AtA.inverse()
    return A.matmul(AtA_inv).matmul(At)


def condition_number_estimate(A: Matrix) -> float:
    """
    Estimates 1-norm condition number kappa(A) = ||A||_1 * ||A^{-1}||_1 for square A.
    """
    if A.rows != A.cols:
        # For rectangular A, condition number of normal equations matrix A^T A
        AtA = A.transpose().matmul(A)
        return AtA.norm_1() * AtA.inverse().norm_1()
    return A.norm_1() * A.inverse().norm_1()
