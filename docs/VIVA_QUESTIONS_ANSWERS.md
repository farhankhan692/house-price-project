# Viva Voce & Presentation Defense Guide
**House Price Least-Squares Model Comparison System (Topic 49)**  
**Course:** Linear Algebra & Vector Calculus (Modules 1, 2, 3)

---

## Section A: Core Linear Algebra & Theoretical Foundations

### Q1: What is the fundamental aim of this mini-project?
**Answer:**
To formulate, solve, verify, and compare multiple mathematical least-squares models for predicting house prices using an 8-stage sequential computational pipeline based on linear algebra (Matrices, Vectors, Linear Systems, Orthogonal Projections, QR Decomposition, and Vector Norms).

### Q2: Why is the house price estimation problem formulated as an overdetermined linear system?
**Answer:**
In real-world housing data, we have $m = 50$ properties (equations) but only $n = 2$ to $6$ feature parameters (unknowns). Because $m > n$, there are more equations than unknowns. Since the target price vector $y \in \mathbb{R}^m$ almost never lies exactly in the column space $\text{Col}(A) \subset \mathbb{R}^m$, the system $A\beta = y$ has no exact solution. Hence, we must seek a least-squares approximation.

### Q3: How is least squares fundamentally interpreted as an orthogonal projection?
**Answer:**
Let $\text{Col}(A)$ be the subspace spanned by the feature column vectors of the design matrix $A$. The vector of actual prices $y$ lies outside this subspace. By the **Orthogonal Projection Theorem**, the unique point $\hat{y} \in \text{Col}(A)$ that minimizes the Euclidean distance $\|y - \hat{y}\|_2$ is the orthogonal projection of $y$ onto $\text{Col}(A)$. The residual vector $e = y - \hat{y}$ is strictly orthogonal to $\text{Col}(A)$, meaning $e \in \text{Col}(A)^\perp = \text{Nul}(A^T)$.

### Q4: Derive the Normal Equations $(A^T A)\beta = A^T y$ from first principles.
**Answer:**
1. Let the residual vector be $e = y - A\beta$.
2. To minimize $\|e\|_2^2$, $e$ must be orthogonal to every column vector $a_j$ of the design matrix $A$:
   $$\langle a_j, e \rangle = a_j^T e = 0, \quad \forall j \in \{1, 2, \dots, n\}$$
3. Stacking these $n$ dot products into matrix notation:
   $$A^T e = \mathbf{0}$$
4. Substitute $e = y - A\beta$:
   $$A^T (y - A\beta) = \mathbf{0} \implies A^T y - A^T A \beta = \mathbf{0}$$
   $$(A^T A)\beta = A^T y$$
5. If the columns of $A$ are linearly independent, $(A^T A)$ is invertible, yielding:
   $$\beta = (A^T A)^{-1} A^T y$$

### Q5: What is the Orthogonal Projection Matrix $P$, and what are its key algebraic properties?
**Answer:**
The projection matrix is defined as $P = A(A^T A)^{-1} A^T$. Its four essential properties are:
1. **Idempotence ($P^2 = P$)**: Projecting a vector that already lies in $\text{Col}(A)$ leaves it unchanged:
   $$P^2 = [A(A^TA)^{-1}A^T][A(A^TA)^{-1}A^T] = A(A^TA)^{-1}(A^TA)(A^TA)^{-1}A^T = A(A^TA)^{-1}A^T = P$$
2. **Symmetry ($P^T = P$)**:
   $$P^T = [A(A^TA)^{-1}A^T]^T = (A^T)^T [(A^TA)^{-1}]^T A^T = A [(A^TA)^T]^{-1} A^T = A(A^TA)^{-1}A^T = P$$
3. **Complementary Subspace Projection ($I - P$)**: The matrix $(I - P)$ projects onto the orthogonal complement $\text{Col}(A)^\perp = \text{Nul}(A^T)$, so $e = (I - P)y$.
4. **Pythagorean Identity**: $\|y\|_2^2 = \|\hat{y}\|_2^2 + \|e\|_2^2$.

### Q6: Why did you also implement QR Decomposition in addition to the Normal Equations?
**Answer:**
The Normal Equations require computing $A^T A$. The condition number of $A^T A$ is the square of the condition number of $A$:
$$\kappa(A^T A) = (\kappa(A))^2$$
If $A$ is moderately ill-conditioned (e.g., $\kappa(A) \approx 10^4$), $A^T A$ will have $\kappa(A^T A) \approx 10^8$, which leads to catastrophic numerical roundoff errors. 
By using **Modified Gram-Schmidt QR decomposition**, $A = QR$ where $Q^T Q = I$ and $R$ is upper triangular. Then:
$$A\beta \approx y \implies QR\beta \approx y \implies R\beta = Q^T y$$
This solves for $\beta$ with condition number $\kappa(R) = \kappa(A)$, doubling numerical stability!

---

## Section B: Sequential Computational Pipeline & Model Comparison

### Q7: Explain the 8 sequential stages in your computational pipeline.
**Answer:**
1. **House Data**: Ingest raw dataset records $(x_i, y_i)$.
2. **Matrix Formation**: Convert records into data matrix $D \in \mathbb{R}^{m \times p}$ and target vector $y \in \mathbb{R}^m$.
3. **Feature Vectors**: Extract individual column vectors $x_j$ and compute variance, standard deviation, and norms.
4. **Design Matrices**: Construct matrices $A_1, A_2, A_3$ appending the intercept basis vector $\mathbf{1}$.
5. **Least-Squares Solutions**: Compute $\beta_k$ and projection matrices $P_k$ using Gauss-Jordan and QR solvers.
6. **Predictions**: Form orthogonal projection $\hat{y}_k = A_k \beta_k = P_k y$.
7. **Residual Analysis**: Compute $e_k = y - \hat{y}_k$ and verify the invariant $A_k^T e_k \approx \mathbf{0}$.
8. **Model Comparison**: Compute error vector norms ($\|e\|_1, \|e\|_2, \|e\|_\infty$), RMSE, and Adjusted $R^2$.

### Q8: What are the three models compared in this project?
**Answer:**
1. **Model 1 (Simple Linear)**: Baseline bivariate model using Living Area only ($n=2$).
2. **Model 2 (Multivariable Linear)**: Multiple features including Living Area, Bedrooms, Bathrooms, Age, and Transit Distance ($n=6$).
3. **Model 3 (Polynomial & Interaction)**: Non-linear regression incorporating living area curvature ($x_{\text{area}}^2/1000$) and area-bedroom interaction ($(x_{\text{area}} \cdot x_{\text{bed}})/1000$) ($n=6$).

### Q9: Why compare multiple vector error norms ($L_1, L_2, L_\infty$)?
**Answer:**
- **$L_2$ Norm (Euclidean)**: Measures root mean squared error. It is mathematically tied to the projection theorem because Euclidean distance is minimized by orthogonal projection. However, it penalizes outliers quadratically.
- **$L_1$ Norm (Manhattan)**: Measures total absolute deviation $\sum |e_i|$. It is linear and more robust to outliers, providing a direct metric of average dollar error per property.
- **$L_\infty$ Norm (Chebyshev)**: Measures the maximum worst-case error $\max |e_i|$ across all properties. In financial appraisals and mortgage lending, the worst-case single prediction failure is critical for risk evaluation.

### Q10: Which model performed best and why?
**Answer:**
**Model 3 (Polynomial & Interaction)** performed best across every single metric:
- **Adjusted $R^2$**: 0.9994 (vs 0.9982 for Model 2 and 0.9981 for Model 1).
- **RMSE ($L_2$ normalized)**: \$3.65k (vs \$6.27k for Model 2 and \$6.82k for Model 1) — a **42% error reduction**.
- **Worst-case error ($L_\infty$)**: \$7.37k (vs \$20.00k for Model 2).

**Mathematical Justification:** Real estate price functions are non-linear; large square footage exhibits diminishing marginal returns, while the interaction term ($x_{\text{area}} \cdot x_{\text{bed}}$) captures the synergy between living space and room utility.

---

## Section C: Verification, Implementation & Numerical Invariants

### Q11: How did your software empirically verify that the solution is a true orthogonal projection?
**Answer:**
We verified three numerical invariants in `src/pipeline.py` and `tests/test_linear_algebra.py`:
1. **Orthogonality**: $\|A^T e\|_2 = 2.13 \times 10^{-5} \approx 0$.
2. **Projection Idempotency**: $\|P^2 - P\|_F = 3.07 \times 10^{-13} \approx 0$.
3. **Equivalence**: $\|A\beta - Py\|_2 = 1.16 \times 10^{-9} \approx 0$.

### Q12: Why did you write a custom linear algebra engine instead of relying entirely on external libraries?
**Answer:**
Writing `Vector` and `Matrix` classes from first principles in pure Python demonstrates complete mastery of the underlying linear algebra:
- Explicit implementation of Gaussian elimination with partial pivoting.
- Explicit QR decomposition using Modified Gram-Schmidt orthogonalization.
- Zero-dependency portability (runs out-of-the-box on any standard Python interpreter, GitHub CI pipeline, or cloud server).

### Q13: What deployment targets are supported?
**Answer:**
1. **Streamlit Community Cloud**: Interactive dashboard with real-time sliders and 3D subspace visualizations (`app.py`).
2. **GitHub Pages**: Static, zero-server web dashboard in `docs/index.html`.
3. **CLI & CI/CD**: Automated GitHub Actions workflow testing matrix theorems on every git push.
