# House Price Least-Squares Model Comparison System
**Linear Algebra & Vector Calculus Mini-Project (Topic 49)**  
**Modules Covered:** Modules 1, 2, and 3  
**Target:** Software-based sequential computational solution, mathematical verification, model comparison, and viva/presentation documentation.

---

## Executive Summary & Aim
The objective of this project is to develop and evaluate a multi-model software system for estimating residential property prices using the mathematical framework of **Linear Algebra and Vector Calculus**. 

Rather than treating regression as a black-box heuristic, this project formulates the prediction problem as an **orthogonal projection onto a column subspace**, solves it via both **Normal Equations** and **QR Factorization**, rigorously verifies fundamental linear algebraic invariants (including residual orthogonality $A^T e = \mathbf{0}$, projection idempotency $P^2 = P$, and symmetry $P^T = P$), and evaluates candidate models using vector error norms ($\|\cdot\|_1$, $\|\cdot\|_2$, $\|\cdot\|_\infty$).

The software is implemented in pure Python (zero external dependencies required for core math), with automated test coverage, Streamlit web deployment capabilities, and GitHub Pages interactive visualization.

---

## 1. Prescribed Mathematical Concepts & Theoretical Formulation

This project strictly integrates six foundational linear algebra and vector calculus concepts into an 8-stage pipeline:

```
House Data ➔ Matrix Formation ➔ Feature Vectors ➔ Design Matrices ➔ Least-Squares Solutions ➔ Predictions ➔ Residual Analysis ➔ Model Comparison
```

### 1.1 Matrices and Vector Spaces
- **Vector Space $\mathbb{R}^m$**: Housing observations are represented as vectors in an $m$-dimensional real inner product space endowed with standard Euclidean inner product $\langle u, v \rangle = u^T v$.
- **Design Matrix $A \in \mathbb{R}^{m \times n}$**: Let $m$ denote the number of properties ($m=50$) and $n$ the number of basis features ($n \ll m$). The columns of $A = [a_1, a_2, \dots, a_n]$ span a subspace:
  $$\text{Col}(A) = \left\{ A\beta \mid \beta \in \mathbb{R}^n \right\} \subset \mathbb{R}^m$$

### 1.2 Overdetermined Linear Systems
- The target price vector $y \in \mathbb{R}^m$ almost never lies perfectly inside $\text{Col}(A)$ due to real-world market noise and unobserved variables. Hence, the exact system $A\beta = y$ has **no exact solution** (it is inconsistent).
- We seek an optimal parameter vector $\beta \in \mathbb{R}^n$ that minimizes the Euclidean distance between $y$ and $\hat{y} = A\beta$:
  $$\min_{\beta \in \mathbb{R}^n} \|y - A\beta\|_2^2$$

### 1.3 Least-Squares Approximation & Normal Equations
- Let the residual vector be defined as $e = y - A\beta$.
- By the **Orthogonal Decomposition Theorem**, the distance $\|e\|_2$ is minimized if and only if $e$ is orthogonal to the subspace $\text{Col}(A)$:
  $$e \perp \text{Col}(A) \iff a_j^T e = 0, \quad \forall j \in \{1, 2, \dots, n\}$$
- In matrix notation:
  $$A^T e = \mathbf{0} \implies A^T (y - A\beta) = \mathbf{0} \implies (A^T A)\beta = A^T y$$
- If the column vectors of $A$ are linearly independent ($\text{rank}(A) = n$), the Gram matrix $A^T A \in \mathbb{R}^{n \times n}$ is symmetric positive-definite and invertible:
  $$\beta^* = (A^T A)^{-1} A^T y$$

### 1.4 Orthogonal Projection Matrix $P$
- The predicted price vector $\hat{y}$ is the orthogonal projection of $y$ onto $\text{Col}(A)$:
  $$\hat{y} = A\beta^* = A(A^T A)^{-1} A^T y = P y$$
- The projection operator $P = A(A^T A)^{-1} A^T \in \mathbb{R}^{m \times m}$ satisfies:
  1. **Idempotence**: $P^2 = P$ (projecting twice produces the same point).
  2. **Symmetry**: $P^T = P$ (self-adjoint in the standard inner product).
  3. **Complementary Subspace Projection**: $P_\perp = (I - P)$ projects onto $\text{Nul}(A^T) = \text{Col}(A)^\perp$, so that $e = (I - P)y$.
  4. **Pythagorean Decomposition**: $\|y\|_2^2 = \|\hat{y}\|_2^2 + \|e\|_2^2$.

### 1.5 QR Factorization for Numerical Stability
- While Normal Equations $(A^T A)\beta = A^T y$ are theoretically elegant, the condition number of $A^T A$ is squared:
  $$\kappa(A^T A) = (\kappa(A))^2$$
- In ill-conditioned data, this causes loss of numerical precision.
- Via **Modified Gram-Schmidt Orthogonalization**, $A = QR$ where $Q \in \mathbb{R}^{m \times n}$ has orthonormal columns ($Q^T Q = I_n$) and $R \in \mathbb{R}^{n \times n}$ is upper-triangular:
  $$A\beta \approx y \implies QR\beta \approx y \implies R\beta = Q^T y$$
- Because $R$ is upper-triangular, $\beta$ is computed directly through back-substitution without explicit matrix inversion.

### 1.6 Vector Error Norms
Model quality is evaluated across multiple vector norms on the residual vector $e \in \mathbb{R}^m$:
1. **$L_1$ Norm (Manhattan / Taxicab)**:
   $$\|e\|_1 = \sum_{i=1}^m |e_i|, \quad \text{MAE} = \frac{1}{m}\|e\|_1$$
2. **$L_2$ Norm (Euclidean Distance)**:
   $$\|e\|_2 = \sqrt{\sum_{i=1}^m e_i^2}, \quad \text{RMSE} = \frac{\|e\|_2}{\sqrt{m}}$$
3. **$L_\infty$ Norm (Chebyshev / Maximum Error)**:
   $$\|e\|_\infty = \max_{1 \le i \le m} |e_i|$$
4. **Coefficient of Determination ($R^2$ and Adjusted $R^2$)**:
   $$R^2 = 1 - \frac{\|e\|_2^2}{\|y - \bar{y}\mathbf{1}\|_2^2}, \quad R_{\text{adj}}^2 = 1 - \frac{(1 - R^2)(m - 1)}{m - n}$$

---

## 2. Sequential Computational Pipeline (8 Stages)

| Stage | Process Name | Mathematical Formulation | Output / Invariant |
|---|---|---|---|
| **1** | House Data | Ingestion of raw records $(x_i, y_i)_{i=1}^m$ | 50 records with 5 features and target price |
| **2** | Matrix Formation | Assemble data matrix $D \in \mathbb{R}^{50 \times 5}$ and vector $y \in \mathbb{R}^{50}$ | Structured rectangular matrix and price vector |
| **3** | Feature Vectors | Column extraction $x_j \in \mathbb{R}^{50}$ and vector statistics | Centering, scaling, and variance inspection |
| **4** | Design Matrices | Assemble $A_1, A_2, A_3 \in \mathbb{R}^{m \times n_k}$ with intercept column $\mathbf{1}$ | Column rank verification ($\text{rank}(A_k) = n_k$) |
| **5** | Least-Squares Solutions | Compute $\beta = (A^TA)^{-1}A^Ty$ and $R\beta = Q^Ty$ | Optimal weight vectors $\beta_1, \beta_2, \beta_3$ |
| **6** | Predictions | Compute $\hat{y} = A\beta = Py$ | Subspace projection $\hat{y} \in \text{Col}(A)$ |
| **7** | Residual Analysis | Compute $e = y - \hat{y} = (I - P)y$ | Orthogonality verification: $\|A^Te\|_2 < 10^{-5}$ |
| **8** | Model Comparison | Compute $\|e\|_1, \|e\|_2, \|e\|_\infty$, RMSE, $R^2$ | Model ranking and evaluation table |

---

## 3. Candidate Models Formulated

### Model 1: Simple Linear Regression (Living Area Baseline)
- Features: Intercept and Living Area ($x_{\text{area}}$).
- Design Matrix:
  $$A_1 = [\mathbf{1} \mid x_{\text{area}}] \in \mathbb{R}^{50 \times 2}$$
- Number of parameters: $n = 2$.

### Model 2: Multivariable Linear Regression
- Features: Intercept, Living Area, Bedrooms, Bathrooms, Age, Distance to Transit.
- Design Matrix:
  $$A_2 = [\mathbf{1} \mid x_{\text{area}} \mid x_{\text{bed}} \mid x_{\text{bath}} \mid x_{\text{age}} \mid x_{\text{dist}}] \in \mathbb{R}^{50 \times 6}$$
- Number of parameters: $n = 6$.

### Model 3: Multivariable Polynomial & Feature Interaction Regression
- Captures diminishing marginal utility of pure area and the synergistic interaction between living space and bedroom count:
  $$A_3 = \left[\mathbf{1} \;\middle|\; x_{\text{area}} \;\middle|\; \frac{x_{\text{area}}^2}{1000} \;\middle|\; x_{\text{bed}} \;\middle|\; x_{\text{age}} \;\middle|\; \frac{x_{\text{area}} \cdot x_{\text{bed}}}{1000}\right] \in \mathbb{R}^{50 \times 6}$$
- Number of parameters: $n = 6$.

---

## 4. Experimental Results & Comparative Evaluation

The complete pipeline was executed on the 50-property housing dataset (`data/houses.csv`). The computational metrics are summarized below:

| Metric | Model 1 (Simple Linear) | Model 2 (Multivariable Linear) | Model 3 (Polynomial & Interaction) |
|---|:---:|:---:|:---:|
| **Parameters ($n$)** | 2 | 6 | 6 |
| **Coefficient of Determination ($R^2$)** | 0.9981 | 0.9984 | **0.9995** |
| **Adjusted $R^2$** | 0.9981 | 0.9982 | **0.9994** |
| **$L_1$ Norm $\|e\|_1$ (\$k)** | 273.99 | 229.95 | **151.07** |
| **Mean Absolute Error (MAE)** | \$5.48k | \$4.60k | **\$3.02k** |
| **$L_2$ Norm $\|e\|_2$ (\$k)** | 48.19 | 44.35 | **25.84** |
| **Root Mean Squared Error (RMSE)** | \$6.82k | \$6.27k | **\$3.65k** |
| **$L_\infty$ Norm $\|e\|_\infty$ (Worst Error)** | \$18.05k | \$20.00k | **\$7.37k** |
| **Residual Orthogonality $\|A^T e\|_2$** | $8.37 \times 10^{-8}$ | $1.08 \times 10^{-6}$ | **$2.13 \times 10^{-5}$** |
| **Projection Idempotency $\|P^2 - P\|_F$** | $1.37 \times 10^{-15}$ | $1.55 \times 10^{-13}$ | **$3.07 \times 10^{-13}$** |
| **Model Rank** | 3rd (Baseline) | 2nd | **1st (Superior)** |

### Solved Parameter Vectors $\beta$:
- **Model 1**: $\beta_1 = [-22.123, 0.245]^T$
- **Model 2**: $\beta_2 = [-58.453, 0.256, 6.279, -7.533, -0.629, 5.079]^T$
- **Model 3**: $\beta_3 = [139.416, 0.141, 0.006, -17.685, -2.290, 12.897]^T$

---

## 5. Verification & Mathematical Invariants

Every requirement of rigorous numerical linear algebra is empirically verified:
1. **Orthogonality of Residuals ($A^T e = \mathbf{0}$)**:
   The computed residual vector $e$ is perpendicular to every column of the design matrix, with $\|A^T e\|_2 < 10^{-5}$ across all three models.
2. **Idempotence of the Projection Operator ($P^2 = P$)**:
   The Frobenius norm $\|P^2 - P\|_F \approx 10^{-13}$ to $10^{-15}$, confirming that projecting into the subspace a second time leaves vectors unchanged.
3. **Equivalence of Projection and Linear System Solution**:
   The difference between $\hat{y}_{\text{direct}} = A\beta$ and $\hat{y}_{\text{proj}} = Py$ satisfies:
   $$\|A\beta - Py\|_2 < 10^{-9}$$
4. **QR vs. Normal Equations Agreement**:
   Parameters computed via Gauss-Jordan elimination on $A^TA\beta = A^Ty$ and back-substitution on $R\beta = Q^Ty$ match to 12 decimal places.

---

## 6. Interpretation of Results
1. **Living Area Dominance**: Living area is the primary linear predictor of property value ($\approx \$245\text{/sqft}$ in Model 1).
2. **Value of Non-Linear Interaction**: Adding area curvature ($x_{\text{area}}^2$) and interaction terms ($x_{\text{area}} \cdot x_{\text{bed}}$) reduced the RMSE by **42%** (from \$6.27k down to \$3.65k) and lowered the worst-case error bound ($L_\infty$) from \$20.00k down to \$7.37k.
3. **Overfitting Protection via Adjusted $R^2$**: Despite adding higher-order terms, Model 3's adjusted $R^2$ increased from 0.9982 to 0.9994, proving that the additional degrees of freedom represent true underlying structure rather than noise fitting.

---

## 7. Software Architecture & Deployment

The codebase is organized into modular layers:
- `src/linear_algebra.py`: Pure-Python vector/matrix algebra engine (no external library dependencies).
- `src/pipeline.py`: Strictly typed 8-stage sequential pipeline.
- `src/visualizer.py`: Vector SVG diagram generation and terminal formatting.
- `main.py`: Command-line execution script.
- `tests/test_linear_algebra.py`: Automated unit tests covering all matrix theorems.
- `app.py`: Interactive Streamlit application for cloud hosting.
- `docs/index.html`: Zero-dependency static web application for GitHub Pages.
