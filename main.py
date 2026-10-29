#!/usr/bin/env python3
"""
House Price Least-Squares Model Comparison System
Main CLI Execution Script

Sequential Mathematical Process:
House data -> Matrix formation -> Feature vectors -> Design matrices ->
Least-squares solutions -> Predictions -> Residual analysis -> Model comparison
"""

import os
import sys
import argparse
from src.pipeline import HousePriceLeastSquaresPipeline
from src.visualizer import (
    format_cli_comparison_table,
    generate_svg_projection_diagram,
    generate_svg_comparison_chart,
)


def main():
    parser = argparse.ArgumentParser(
        description="Linear Algebra Mini-Project: House Price Least-Squares Comparison System"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/houses.csv",
        help="Path to CSV dataset containing property features and prices",
    )
    parser.add_argument(
        "--export-svg",
        action="store_true",
        default=True,
        help="Export SVG visualization diagrams to docs/assets/",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("  LINEAR ALGEBRA & VECTOR CALCULUS MINI-PROJECT")
    print("  Topic 49: House Price Least-Squares Model Comparison System")
    print("=" * 80)
    print(f"Dataset path: {args.data}\n")

    if not os.path.exists(args.data):
        print(f"Error: Dataset file '{args.data}' not found!")
        sys.exit(1)

    pipeline = HousePriceLeastSquaresPipeline(args.data)

    print("Executing 8-Stage Sequential Mathematical Process:")
    print("-" * 80)

    # Stage 1
    print("[Stage 1/8] House Data...")
    records = pipeline.stage_1_house_data()
    print(f"            Loaded {len(records)} property records.")

    # Stage 2
    print("[Stage 2/8] Matrix Formation...")
    D, y, feature_names = pipeline.stage_2_matrix_formation(records)
    print(f"            Feature Matrix D shape: {D.shape()}, Target Vector y dim: {y.dim}")

    # Stage 3
    print("[Stage 3/8] Feature Vectors...")
    stage_3_data = pipeline.stage_3_feature_vectors(D, y, feature_names)
    for feat, stats in stage_3_data["stats"].items():
        print(f"            Vector '{feat}': Mean={stats['mean']:.2f}, Std={stats['std']:.2f}, L2-Norm={stats['norm_l2']:.2f}")

    # Stage 4
    print("[Stage 4/8] Design Matrices...")
    design_dict = pipeline.stage_4_design_matrices(stage_3_data)
    for name, info in design_dict.items():
        print(f"            {name}: Matrix shape {info['matrix'].shape()} with {len(info['labels'])} basis vectors")

    # Stage 5
    print("[Stage 5/8] Least-Squares Solutions...")
    solutions = pipeline.stage_5_least_squares_solutions(design_dict, y)
    for name, sol in solutions.items():
        beta_str = ", ".join(f"{b:.3f}" for b in sol["beta_normal"].elements)
        print(f"            {name}:")
        print(f"              Normal Eq beta: [{beta_str}]")
        print(f"              QR Sol beta:    [{', '.join(f'{b:.3f}' for b in sol['beta_qr'].elements)}]")
        print(f"              A^T A Condition Number: {sol['condition_number']:.2e}")

    # Stage 6
    print("[Stage 6/8] Predictions (Orthogonal Projections)...")
    predictions = pipeline.stage_6_predictions(solutions, y)
    for name, pred in predictions.items():
        print(f"            {name}: Projection check ||A beta - P y||_2 = {pred['equiv_error']:.2e}")

    # Stage 7
    print("[Stage 7/8] Residual Analysis & Orthogonality Verification...")
    residuals = pipeline.stage_7_residual_analysis(predictions, y)
    for name, res in residuals.items():
        print(f"            {name}:")
        print(f"              Orthogonality ||A^T e||_2:     {res['orthogonality_norm']:.2e} (Invariant: 0)")
        print(f"              Projection Diff ||(I-P)y - e||: {res['proj_diff']:.2e}")
        print(f"              Idempotency ||P^2 - P||_F:      {res['idempotence_error']:.2e}")

    # Stage 8
    print("[Stage 8/8] Model Comparison & Error Norm Evaluation...")
    evaluations = pipeline.stage_8_model_comparison(residuals, y)

    print("\n" + "=" * 80)
    print("FINAL MODEL COMPARISON RESULTS:")
    print("=" * 80)
    print(format_cli_comparison_table(evaluations))
    print("\nRanking Summary:")
    for rank, ev in enumerate(evaluations, 1):
        print(f"  Rank {rank}: {ev.model_name} -> Adj R^2 = {ev.adj_r_squared:.4f}, RMSE = ${ev.rmse:.2f}k, L_inf = ${ev.norm_inf:.2f}k")

    # Export SVGs
    if args.export_svg:
        os.makedirs("docs/assets", exist_ok=True)
        proj_svg = "docs/assets/orthogonal_projection.svg"
        comp_svg = "docs/assets/model_comparison.svg"
        generate_svg_projection_diagram(proj_svg)
        generate_svg_comparison_chart(evaluations, comp_svg)
        print(f"\nGenerated vector SVG visualizations:")
        print(f"  - {proj_svg}")
        print(f"  - {comp_svg}")

    print("\nExecution completed successfully!")


if __name__ == "__main__":
    main()
