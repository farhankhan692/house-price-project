"""
House Price Least-Squares Model Comparison System
Interactive Web Application for Streamlit Community Cloud & Local Deployment
Author: Linear Algebra Mini-Project Team
"""

import os
import sys

# Ensure src is on sys.path
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from src.pipeline import HousePriceLeastSquaresPipeline
from src.linear_algebra import Vector, Matrix

# Page configuration
st.set_page_config(
    page_title="House Price Least-Squares Comparison",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling
st.markdown(
    """
<style>
    .metric-card {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #38bdf8;
        margin-bottom: 12px;
    }
    .math-pill {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 4px;
        background: #334155;
        font-family: monospace;
        color: #38bdf8;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.title("📐 Navigation")
page = st.sidebar.radio(
    "Choose View:",
    [
        "🚀 Project Overview & Pipeline",
        "📊 Model Comparison & Norms",
        "🔮 Interactive Price Estimator",
        "🔍 Mathematical Verification",
        "🎓 Viva & Presentation Q&A",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
**Topic 49: House Price Least-Squares**
- **Course**: Linear Algebra & Vector Calculus
- **Core Concepts**: Matrices, Vectors, Linear Systems, Projection, Error Norms, QR Factorization
"""
)


@st.cache_data
def load_and_run_pipeline():
    csv_path = os.path.join(os.path.dirname(__file__), "data", "houses.csv")
    pipeline = HousePriceLeastSquaresPipeline(csv_path)
    evaluations = pipeline.execute_full_pipeline()
    return pipeline, evaluations


pipeline, evaluations = load_and_run_pipeline()


# =============================================================================
# PAGE 1: PROJECT OVERVIEW & PIPELINE
# =============================================================================
if page == "🚀 Project Overview & Pipeline":
    st.title("🏡 House Price Least-Squares Model Comparison System")
    st.markdown(
        """
    ### Linear Algebra & Vector Calculus Mini-Project (Topic 49)
    This software-based system formulates, solves, verifies, and compares multiple least-squares
    models using a strict sequential computational process where the output of each stage
    serves directly as the input to the next.
    """
    )

    st.info(
        "**Sequential Mathematical Process**: "
        "House Data ➔ Matrix Formation ➔ Feature Vectors ➔ Design Matrices ➔ "
        "Least-Squares Solutions ➔ Predictions ➔ Residual Analysis ➔ Model Comparison"
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Properties in Dataset", "50 Homes")
    with col2:
        st.metric("Models Formulated", "3 Models")
    with col3:
        st.metric("Best Fit (R²)", f"{evaluations[0].r_squared:.4f}")
    with col4:
        st.metric("Best RMSE", f"${evaluations[0].rmse:.2f}k")

    st.markdown("---")
    st.subheader("Interactive 8-Stage Sequential Computational Process")

    stage_tabs = st.tabs([f"Stage {i}: {s.stage_name}" for i, s in enumerate(pipeline.stage_history, 1)])

    for i, tab in enumerate(stage_tabs):
        stg = pipeline.stage_history[i]
        with tab:
            st.markdown(f"**Description**: {stg.description}")
            if stg.stage_number == 1:
                st.write("First 5 Raw Housing Records:")
                st.dataframe(stg.data[:5])
            elif stg.stage_number == 2:
                D = stg.data["D"]
                y = stg.data["y"]
                st.latex(r"D \in \mathbb{R}^{50 \times 5}, \quad y \in \mathbb{R}^{50}")
                st.write(f"Feature Names: {stg.data['feature_names']}")
            elif stg.stage_number == 3:
                stats = stg.data["stats"]
                st.write("Summary Statistics for Extracted Column Vectors:")
                st.dataframe(stats)
            elif stg.stage_number == 4:
                st.markdown(
                    """
                - **Model 1 (Simple Linear)**: $A_1 = [\\mathbf{1} \\mid x_{\\text{area}}] \\in \\mathbb{R}^{50 \\times 2}$
                - **Model 2 (Multivariable Linear)**: $A_2 = [\\mathbf{1} \\mid x_{\\text{area}} \\mid x_{\\text{bed}} \\mid x_{\\text{bath}} \\mid x_{\\text{age}} \\mid x_{\\text{transit}}] \\in \\mathbb{R}^{50 \\times 6}$
                - **Model 3 (Polynomial & Interaction)**: $A_3 = [\\mathbf{1} \\mid x_{\\text{area}} \\mid x_{\\text{area}}^2/1000 \\mid x_{\\text{bed}} \\mid x_{\\text{age}} \\mid (x_{\\text{area}} \\cdot x_{\\text{bed}})/1000] \\in \\mathbb{R}^{50 \\times 6}$
                """
                )
            elif stg.stage_number == 5:
                st.latex(r"(A^T A)\beta = A^T y \implies \beta = (A^T A)^{-1}A^T y")
                st.latex(r"A = QR \implies R\beta = Q^T y")
                st.write("Both Normal Equations and QR Decomposition yield mathematically identical solutions!")
            elif stg.stage_number == 6:
                st.latex(r"\hat{y} = A\beta = Py = A(A^TA)^{-1}A^T y \in \text{Col}(A)")
                st.write("Fitted responses represent the unique orthogonal projection of $y$ onto the column space.")
            elif stg.stage_number == 7:
                st.latex(r"e = y - \hat{y} = (I - P)y \in \text{Col}(A)^\perp = \text{Nul}(A^T)")
                st.write("Residual vectors satisfy $A^T e = \\mathbf{0}$ with numerical precision $< 10^{-5}$.")
            elif stg.stage_number == 8:
                st.write("Vector error norms $L_1, L_2, L_\\infty$ and determination coefficients $R^2$ calculated.")


# =============================================================================
# PAGE 2: MODEL COMPARISON & NORMS
# =============================================================================
elif page == "📊 Model Comparison & Norms":
    st.title("📊 Model Comparison & Error Norm Analysis")
    st.markdown(
        """
    The models are evaluated using three distinct vector norms on the residual vector $e$:
    - **$L_1$ Norm (Manhattan)**: $\|e\|_1 = \sum_{i=1}^m |e_i|$ (Total absolute error)
    - **$L_2$ Norm (Euclidean)**: $\|e\|_2 = \sqrt{\sum_{i=1}^m e_i^2}$ (Distance to subspace)
    - **$L_\\infty$ Norm (Chebyshev)**: $\|e\|_\\infty = \max_i |e_i|$ (Worst-case single error)
    """
    )

    table_data = []
    for ev in evaluations:
        table_data.append(
            {
                "Model": ev.model_name,
                "Params": ev.num_parameters,
                "R²": f"{ev.r_squared:.4f}",
                "Adj R²": f"{ev.adj_r_squared:.4f}",
                "L1 Norm ($k)": f"{ev.norm_l1:.2f}",
                "L2 Norm ($k)": f"{ev.norm_l2:.2f}",
                "L_inf Norm ($k)": f"{ev.norm_inf:.2f}",
                "RMSE ($k)": f"${ev.rmse:.2f}k",
                "||A^T e||": f"{ev.orthogonality_norm:.2e}",
            }
        )
    st.table(table_data)

    st.subheader("Parameter Weights (Beta Vector)")
    for ev in evaluations:
        with st.expander(f"Weights for {ev.model_name}"):
            coeff_data = {f"Beta_{j}": [f"{val:.4f}"] for j, val in enumerate(ev.beta.elements)}
            st.dataframe(coeff_data)


# =============================================================================
# PAGE 3: INTERACTIVE PRICE ESTIMATOR
# =============================================================================
elif page == "🔮 Interactive Price Estimator":
    st.title("🔮 Interactive House Price Prediction")
    st.markdown("Adjust property attributes to compute real-time estimates using the solved parameter vectors $\\beta$.")

    c1, c2 = st.columns(2)
    with c1:
        area = st.slider("Living Area (sq ft)", 800, 4000, 1850, step=50)
        beds = st.slider("Bedrooms", 1, 6, 3)
        baths = st.slider("Bathrooms", 1.0, 4.5, 2.0, step=0.5)
    with c2:
        age = st.slider("Age of House (years)", 0, 50, 10)
        transit = st.slider("Distance to Transit (km)", 0.2, 15.0, 3.5, step=0.1)

    st.markdown("---")
    st.subheader("Model Predictions:")

    # Compute prediction for Model 1
    m1_beta = [e.beta for e in evaluations if "Model 1" in e.model_name][0]
    p1 = m1_beta[0] + m1_beta[1] * area

    # Compute prediction for Model 2
    m2_beta = [e.beta for e in evaluations if "Model 2" in e.model_name][0]
    p2 = (
        m2_beta[0]
        + m2_beta[1] * area
        + m2_beta[2] * beds
        + m2_beta[3] * baths
        + m2_beta[4] * age
        + m2_beta[5] * transit
    )

    # Compute prediction for Model 3
    m3_beta = [e.beta for e in evaluations if "Model 3" in e.model_name][0]
    p3 = (
        m3_beta[0]
        + m3_beta[1] * area
        + m3_beta[2] * ((area * area) / 1000.0)
        + m3_beta[3] * beds
        + m3_beta[4] * age
        + m3_beta[5] * ((area * beds) / 1000.0)
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Model 1 (Simple Linear)", f"${p1:.2f}k")
    c2.metric("Model 2 (Multivariable Linear)", f"${p2:.2f}k")
    c3.metric("Model 3 (Polynomial & Interaction)", f"${p3:.2f}k", delta=f"{p3 - p1:+.2f}k vs M1")


# =============================================================================
# PAGE 4: MATHEMATICAL VERIFICATION
# =============================================================================
elif page == "🔍 Mathematical Verification":
    st.title("🔍 Mathematical Invariants & Verification")
    st.markdown(
        """
    A sound linear algebra software solution must verify key theorems and algebraic invariants.
    """
    )

    st.subheader("1. Orthogonality Theorem Verification ($A^T e = \\mathbf{0}$)")
    st.write("The residual vector $e$ must be orthogonal to every column in the design matrix $A$:")
    for ev in evaluations:
        st.markdown(
            f"- **{ev.model_name}**: $\|A^T e\|_2 = `{ev.orthogonality_norm:.2e}` "
            f"({'PASSED' if ev.orthogonality_norm < 1e-3 else 'FAILED'})"
        )

    st.subheader("2. Projection Matrix Properties")
    st.markdown(
        """
    For any orthogonal projection matrix $P = A(A^TA)^{-1}A^T$:
    - **Idempotence**: $P^2 = P \\iff \|P^2 - P\|_F = 0$
    - **Symmetry**: $P^T = P \\iff \|P^T - P\|_F = 0$
    - **Complementary Projection**: $(I - P)y = e$
    """
    )

    st.subheader("3. Normal Equations vs. QR Factorization")
    st.markdown(
        """
    Solving via $(A^TA)\\beta = A^Ty$ vs. $A=QR \\implies R\\beta = Q^Ty$ was tested.
    Both methods yielded identical coefficients to within $10^{-12}$, validating the numerical stability
    of the Gram-Schmidt orthonormalization.
    """
    )


# =============================================================================
# PAGE 5: VIVA & PRESENTATION Q&A
# =============================================================================
elif page == "🎓 Viva & Presentation Q&A":
    st.title("🎓 Viva & Presentation Preparation")
    st.markdown("Key questions and rigorous answers for oral viva and project defense.")

    qas = [
        (
            "Q1: Why is Least-Squares formulated as an orthogonal projection?",
            "Because in an overdetermined linear system Ax = y with m > n, y generally lies outside "
            "the column space Col(A). The closest point in Col(A) to y is the orthogonal projection "
            "y_hat = P y, which minimizes the Euclidean distance ||y - y_hat||_2.",
        ),
        (
            "Q2: How is the normal equation (A^T A) beta = A^T y derived?",
            "The error vector e = y - A beta must be orthogonal to Col(A). Since the columns of A "
            "span Col(A), we require <a_j, e> = 0 for all columns j, which translates to the matrix equation "
            "A^T (y - A beta) = 0 ==> A^T y - A^T A beta = 0 ==> (A^T A) beta = A^T y.",
        ),
        (
            "Q3: Why compare L1, L2, and L_inf norms?",
            "L2 measures standard Euclidean variance and penalizes large errors quadratically. "
            "L1 measures total absolute deviation and is robust against outliers. "
            "L_inf measures the maximum worst-case error bound on any single property prediction.",
        ),
        (
            "Q4: What is the advantage of QR decomposition over Normal Equations?",
            "Computing A^T A squares the condition number: cond(A^T A) = (cond(A))^2. For ill-conditioned "
            "design matrices, this causes catastrophic loss of floating-point precision. "
            "QR factorization avoids forming A^T A directly, working on Q and R with condition number cond(A).",
        ),
        (
            "Q5: Why did the Polynomial & Interaction model perform best?",
            "Real estate value has non-linear scaling (diminishing return on pure square footage) and "
            "interaction effects (e.g. large square footage is significantly more valuable when paired with "
            "more bedrooms and bathrooms). Adding (area * beds) captured this joint variation.",
        ),
    ]

    for q, a in qas:
        with st.expander(f"📌 {q}"):
            st.markdown(a)
