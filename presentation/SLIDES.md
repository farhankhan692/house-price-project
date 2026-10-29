# Viva Presentation Slide Deck
## House Price Least-Squares Model Comparison System
**Linear Algebra & Vector Calculus Mini-Project (Topic 49)**

---

### Slide 1: Title & Overview
- **Project Title:** House Price Least-Squares Model Comparison System
- **Topic:** #49 (Modules 1, 2, 3)
- **Subject:** Linear Algebra & Vector Calculus
- **Aim:** Formulate, solve, verify, and mathematically compare multiple least-squares models for property price estimation.
- **Key Concepts:** Matrices, Vectors, Linear Systems, Least-Squares, Orthogonal Projection, Norms ($L_1, L_2, L_\infty$).

---

### Slide 2: Problem Statement & Geometric Intuition
- **The Challenge:** Real estate price prediction with $m = 50$ properties and $n = 2 \dots 6$ features.
- **Overdetermined Linear System:** $A\beta = y$, where $m > n$.
- **Why no exact solution exists:** Target vector $y$ lies in $\mathbb{R}^m$, but generally outside the column space $\text{Col}(A)$.
- **Least-Squares Objective:** Find $\hat{y} \in \text{Col}(A)$ that minimizes the Euclidean distance $\|y - \hat{y}\|_2$.

---

### Slide 3: Sequential Mathematical Architecture
```
Stage 1: House Data (CSV Records)
   ↓
Stage 2: Matrix Formation (Data Matrix D ∈ R^{50x5}, y ∈ R^50)
   ↓
Stage 3: Feature Vectors (Column extraction x_j, statistics)
   ↓
Stage 4: Design Matrices (A1, A2, A3 with intercept 1)
   ↓
Stage 5: Least-Squares Solvers (Normal Equations & QR Factorization)
   ↓
Stage 6: Predictions (Orthogonal Projection ŷ = Aβ = Py)
   ↓
Stage 7: Residual Analysis (Orthogonality check Aᵀe = 0)
   ↓
Stage 8: Model Comparison (Error Norms L1, L2, L_inf, Adj R²)
```

---

### Slide 4: Mathematical Derivation & Normal Equations
1. **Residual Vector:** $e = y - A\beta$.
2. **Orthogonality Requirement:** Residual must be orthogonal to every column $a_j$ of $A$:
   $$\langle a_j, e \rangle = 0 \iff A^T e = \mathbf{0}$$
3. **Normal Equations:**
   $$A^T (y - A\beta) = \mathbf{0} \implies (A^T A)\beta = A^T y$$
4. **Parameter Solution:**
   $$\beta^* = (A^T A)^{-1} A^T y$$

---

### Slide 5: The Orthogonal Projection Matrix $P$
- **Projection Operator:** $P = A(A^T A)^{-1} A^T \in \mathbb{R}^{m \times m}$
- **Core Algebraic Invariants:**
  1. **Idempotency:** $P^2 = P$ (repeated projection has no further effect).
  2. **Symmetry:** $P^T = P$ (self-adjoint operator).
  3. **Complementary Subspace:** $(I - P)$ projects onto $\text{Col}(A)^\perp = \text{Nul}(A^T)$.
  4. **Pythagorean Decomposition:** $\|y\|_2^2 = \|\hat{y}\|_2^2 + \|e\|_2^2$.

---

### Slide 6: Candidate Models
- **Model 1: Simple Linear Regression**
  $$A_1 = [\mathbf{1} \mid x_{\text{area}}] \in \mathbb{R}^{50 \times 2}$$
- **Model 2: Multivariable Linear Regression**
  $$A_2 = [\mathbf{1} \mid x_{\text{area}} \mid x_{\text{bed}} \mid x_{\text{bath}} \mid x_{\text{age}} \mid x_{\text{dist}}] \in \mathbb{R}^{50 \times 6}$$
- **Model 3: Polynomial & Interaction Model**
  $$A_3 = \left[\mathbf{1} \mid x_{\text{area}} \mid \frac{x_{\text{area}}^2}{1000} \mid x_{\text{bed}} \mid x_{\text{age}} \mid \frac{x_{\text{area}} \cdot x_{\text{bed}}}{1000}\right] \in \mathbb{R}^{50 \times 6}$$

---

### Slide 7: Experimental Results & Model Comparison Table
| Metric | Model 1 (Simple) | Model 2 (Multivariable) | Model 3 (Poly & Interaction) |
|---|:---:|:---:|:---:|
| **Parameters ($n$)** | 2 | 6 | 6 |
| **Adjusted $R^2$** | 0.9981 | 0.9982 | **0.9994** |
| **$L_1$ Norm $\|e\|_1$** | \$273.99k | \$229.95k | **\$151.07k** |
| **$L_2$ Norm $\|e\|_2$** | \$48.19k | \$44.35k | **\$25.84k** |
| **RMSE ($k$)** | \$6.82k | \$6.27k | **\$3.65k** (42% reduction) |
| **$L_\infty$ Norm $\|e\|_\infty$** | \$18.05k | \$20.00k | **\$7.37k** (Worst-case bound) |
| **Rank** | 3rd | 2nd | **1st (Best)** |

---

### Slide 8: Mathematical Verification & Numerical Stability
- **Orthogonality Check:** $\|A^T e\|_2 < 2.13 \times 10^{-5} \approx 0$ (Verified).
- **Idempotence Check:** $\|P^2 - P\|_F \approx 10^{-13}$ (Verified).
- **Projection Identity:** $\|A\beta - Py\|_2 < 10^{-9}$ (Verified).
- **QR Factorization Comparison:** Solutions from Normal Equations and Modified Gram-Schmidt QR match to 12 decimal digits.

---

### Slide 9: Software Architecture & Deployment
- **Pure-Python Engine:** Custom `Vector` and `Matrix` classes with zero dependencies.
- **Automated Testing:** 10/10 passing unit tests verifying mathematical theorems.
- **Deployment Ready:**
  - **Streamlit Community Cloud:** 1-click cloud dashboard (`app.py`).
  - **GitHub Pages:** Standalone zero-server visualizer (`docs/index.html`).
  - **CI/CD:** GitHub Actions workflow (`.github/workflows/ci.yml`).

---

### Slide 10: Conclusion & Takeaways
1. Least squares is not just a statistical tool—it is the geometric projection of a vector onto a finite-dimensional subspace.
2. Incorporating non-linear curvature and feature interaction reduced prediction error by **42%**.
3. Multidimensional norm evaluation ($L_1, L_2, L_\infty$) provides a comprehensive profile of average, Euclidean, and worst-case model behavior.
