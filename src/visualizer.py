"""
Visualization and Terminal Reporting Module
Generates ASCII comparison tables and standalone SVG graphics
for least-squares projections, residual distributions, and model performance.
Requires zero external dependencies.
"""

from typing import List
from src.pipeline import ModelEvaluation


def format_cli_comparison_table(evaluations: List[ModelEvaluation]) -> str:
    """Formats a clean markdown/ASCII comparison table for the terminal."""
    header = (
        "+-----------------------------------+--------+---------+----------+----------+---------+---------+-----------+\n"
        "| Model Name                        | Params | R^2     | Adj R^2  | L1 Norm  | L2 Norm | L_inf   | RMSE ($k) |\n"
        "+-----------------------------------+--------+---------+----------+----------+---------+---------+-----------+"
    )
    rows = []
    for ev in evaluations:
        row = (
            f"| {ev.model_name:<33} | {ev.num_parameters:^6} | {ev.r_squared:^7.4f} | "
            f"{ev.adj_r_squared:^8.4f} | {ev.norm_l1:^8.2f} | {ev.norm_l2:^7.2f} | "
            f"{ev.norm_inf:^7.2f} | {ev.rmse:^9.2f} |"
        )
        rows.append(row)
    footer = (
        "+-----------------------------------+--------+---------+----------+----------+---------+---------+-----------+"
    )
    return "\n".join([header] + rows + [footer])


def generate_svg_projection_diagram(filepath: str):
    """
    Generates an educational SVG illustrating the geometry of Least Squares:
    Subspace Col(A), Actual Price vector y, Projection y_hat = P y, and Residual e = (I - P) y.
    """
    svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 500" width="100%" height="100%" style="background:#0f172a; font-family:system-ui, -apple-system, sans-serif;">
  <defs>
    <linearGradient id="planeGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#3b82f6" stop-opacity="0.25"/>
      <stop offset="100%" stop-color="#1d4ed8" stop-opacity="0.08"/>
    </linearGradient>
    <marker id="arrow-blue" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#38bdf8"/>
    </marker>
    <marker id="arrow-emerald" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#34d399"/>
    </marker>
    <marker id="arrow-rose" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#f43f5e"/>
    </marker>
  </defs>

  <!-- Title -->
  <text x="400" y="40" text-anchor="middle" fill="#f8fafc" font-size="20" font-weight="700">Orthogonal Projection in Least-Squares Approximation</text>
  <text x="400" y="65" text-anchor="middle" fill="#94a3b8" font-size="13">Fundamental Subspace Theorem: y = ŷ + e where ŷ ∈ Col(A) and e ∈ Col(A)⊥</text>

  <!-- Column Space Plane (Col(A)) -->
  <polygon points="120,400 480,440 720,280 360,240" fill="url(#planeGrad)" stroke="#60a5fa" stroke-width="2" stroke-dasharray="4,4"/>
  <text x="640" y="270" fill="#93c5fd" font-size="15" font-weight="600">Column Space Col(A)</text>

  <!-- Origin -->
  <circle cx="280" cy="380" r="5" fill="#f8fafc"/>
  <text x="260" y="395" fill="#cbd5e1" font-size="14">Origin 0</text>

  <!-- Basis Vectors in Col(A) -->
  <line x1="280" y1="380" x2="420" y2="400" stroke="#64748b" stroke-width="2" marker-end="url(#arrow-blue)"/>
  <text x="430" y="415" fill="#94a3b8" font-size="12">a₁ (Feature 1)</text>

  <line x1="280" y1="380" x2="460" y2="300" stroke="#64748b" stroke-width="2" marker-end="url(#arrow-blue)"/>
  <text x="470" y="300" fill="#94a3b8" font-size="12">a₂ (Feature 2)</text>

  <!-- Target Vector y -->
  <line x1="280" y1="380" x2="520" y2="130" stroke="#38bdf8" stroke-width="3.5" marker-end="url(#arrow-blue)"/>
  <text x="420" y="230" fill="#38bdf8" font-size="16" font-weight="700">y (Actual Prices)</text>

  <!-- Projected Vector y_hat -->
  <line x1="280" y1="380" x2="520" y2="340" stroke="#34d399" stroke-width="3.5" marker-end="url(#arrow-emerald)"/>
  <text x="400" y="380" fill="#34d399" font-size="16" font-weight="700">ŷ = Aβ = Py (Prediction)</text>

  <!-- Residual Vector e -->
  <line x1="520" y1="340" x2="520" y2="135" stroke="#f43f5e" stroke-width="3" stroke-dasharray="6,4" marker-end="url(#arrow-rose)"/>
  <text x="535" y="235" fill="#f43f5e" font-size="15" font-weight="700">e = y - ŷ = (I - P)y</text>

  <!-- Right Angle Symbol -->
  <polyline points="520,315 500,315 500,340" fill="none" stroke="#f43f5e" stroke-width="1.8"/>
  <text x="485" y="332" fill="#f43f5e" font-size="10" font-weight="bold">90°</text>

  <!-- Math Summary Box -->
  <rect x="50" y="100" width="220" height="150" rx="8" fill="#1e293b" stroke="#334155" stroke-width="1.5"/>
  <text x="65" y="125" fill="#38bdf8" font-size="13" font-weight="bold">Key Invariants Verified:</text>
  <text x="65" y="150" fill="#e2e8f0" font-size="12">1. Aᵀ e = 0  (Orthogonality)</text>
  <text x="65" y="175" fill="#e2e8f0" font-size="12">2. P² = P  (Idempotency)</text>
  <text x="65" y="200" fill="#e2e8f0" font-size="12">3. Pᵀ = P  (Symmetry)</text>
  <text x="65" y="225" fill="#e2e8f0" font-size="12">4. ||y||² = ||ŷ||² + ||e||²</text>
</svg>"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(svg_content)


def generate_svg_comparison_chart(evaluations: List[ModelEvaluation], filepath: str):
    """
    Generates a clean vector SVG bar chart comparing RMSE, L1, and R^2 across models.
    """
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 450" width="100%" height="100%" style="background:#0f172a; font-family:system-ui, -apple-system, sans-serif;">
  <text x="400" y="38" text-anchor="middle" fill="#f8fafc" font-size="20" font-weight="700">House Price Model Performance Comparison</text>
  <text x="400" y="62" text-anchor="middle" fill="#94a3b8" font-size="13">Evaluation of Error Norms (L1, L2 / RMSE) and Adjusted R²</text>

  <!-- Legend -->
  <rect x="180" y="85" width="16" height="12" fill="#38bdf8" rx="2"/>
  <text x="205" y="96" fill="#cbd5e1" font-size="12">RMSE ($k) [lower is better]</text>
  <rect x="420" y="85" width="16" height="12" fill="#34d399" rx="2"/>
  <text x="445" y="96" fill="#cbd5e1" font-size="12">Adj R² (%) [higher is better]</text>

  <!-- Chart Grid Area -->
  <line x1="80" y1="360" x2="740" y2="360" stroke="#334155" stroke-width="1.5"/>
"""
    # Plot bars
    bar_width = 70
    group_gap = 210
    start_x = 130

    max_rmse = max(ev.rmse for ev in evaluations)

    for i, ev in enumerate(evaluations):
        gx = start_x + i * group_gap
        # Bar 1: RMSE (height scaled to max 180px)
        rmse_h = (ev.rmse / max_rmse) * 180
        rmse_y = 360 - rmse_h

        # Bar 2: Adj R2 (height scaled from 0.98 to 1.0 -> 0 to 180px)
        r2_pct = ev.adj_r_squared * 100
        # scale from 98% to 100%
        r2_scaled = max(10, (ev.adj_r_squared - 0.98) / (1.0 - 0.98) * 180)
        r2_y = 360 - r2_scaled

        short_name = ev.model_name.split("(")[0].strip()

        svg += f"""
  <!-- Group {i+1}: {ev.model_name} -->
  <text x="{gx + bar_width}" y="385" text-anchor="middle" fill="#f1f5f9" font-size="13" font-weight="600">{short_name}</text>
  <text x="{gx + bar_width}" y="405" text-anchor="middle" fill="#94a3b8" font-size="11">({ev.num_parameters} params)</text>

  <!-- RMSE Bar -->
  <rect x="{gx}" y="{rmse_y:.1f}" width="{bar_width*0.8:.1f}" height="{rmse_h:.1f}" fill="#38bdf8" rx="4"/>
  <text x="{gx + bar_width*0.4:.1f}" y="{rmse_y - 8:.1f}" text-anchor="middle" fill="#38bdf8" font-size="12" font-weight="bold">${ev.rmse:.2f}k</text>

  <!-- R2 Bar -->
  <rect x="{gx + bar_width:.1f}" y="{r2_y:.1f}" width="{bar_width*0.8:.1f}" height="{r2_scaled:.1f}" fill="#34d399" rx="4"/>
  <text x="{gx + bar_width*1.4:.1f}" y="{r2_y - 8:.1f}" text-anchor="middle" fill="#34d399" font-size="12" font-weight="bold">{r2_pct:.2f}%</text>
"""
    svg += "</svg>"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(svg)
