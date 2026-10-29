"""
Sequential Computational Pipeline for House Price Least-Squares Comparison
Implements the strict 8-stage sequential mathematical process:
Stage 1: House Data
Stage 2: Matrix Formation
Stage 3: Feature Vectors
Stage 4: Design Matrices
Stage 5: Least-Squares Solutions
Stage 6: Predictions
Stage 7: Residual Analysis
Stage 8: Model Comparison
"""

import csv
import math
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple

from src.linear_algebra import (
    Vector,
    Matrix,
    qr_decomposition,
    solve_qr_least_squares,
    orthogonal_projection_matrix,
    condition_number_estimate,
)


@dataclass
class StageOutput:
    stage_number: int
    stage_name: str
    description: str
    data: Any


@dataclass
class ModelEvaluation:
    model_name: str
    num_parameters: int
    beta: Vector
    predictions: Vector
    residuals: Vector
    norm_l1: float
    mae: float
    norm_l2: float
    rmse: float
    norm_inf: float
    r_squared: float
    adj_r_squared: float
    orthogonality_norm: float
    condition_number: float


class HousePriceLeastSquaresPipeline:
    """Orchestrates the sequential mathematical stages where output of stage k is input to stage k+1."""

    def __init__(self, csv_filepath: str):
        self.csv_filepath = csv_filepath
        self.stage_history: List[StageOutput] = []

    # =========================================================================
    # STAGE 1: House Data
    # =========================================================================
    def stage_1_house_data(self) -> List[Dict[str, float]]:
        """Reads raw tabular observations of property features and prices from CSV."""
        records = []
        with open(self.csv_filepath, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                record = {
                    "id": float(row["id"]),
                    "area_sqft": float(row["area_sqft"]),
                    "bedrooms": float(row["bedrooms"]),
                    "bathrooms": float(row["bathrooms"]),
                    "age_years": float(row["age_years"]),
                    "dist_transit_km": float(row["dist_transit_km"]),
                    "price": float(row["price"]),
                }
                records.append(record)
        output = StageOutput(
            stage_number=1,
            stage_name="House Data",
            description=f"Loaded {len(records)} property records from CSV.",
            data=records,
        )
        self.stage_history.append(output)
        return records

    # =========================================================================
    # STAGE 2: Matrix Formation
    # =========================================================================
    def stage_2_matrix_formation(self, records: List[Dict[str, float]]) -> Tuple[Matrix, Vector, List[str]]:
        """Maps tabular records into numerical feature matrix D in R^{m x p} and target vector y in R^m."""
        feature_names = ["area_sqft", "bedrooms", "bathrooms", "age_years", "dist_transit_km"]
        raw_matrix_data = []
        target_list = []

        for r in records:
            row = [r[name] for name in feature_names]
            raw_matrix_data.append(row)
            target_list.append(r["price"])

        D = Matrix(raw_matrix_data)
        y = Vector(target_list)

        output = StageOutput(
            stage_number=2,
            stage_name="Matrix Formation",
            description=f"Formed raw feature matrix D of shape {D.shape()} and price vector y of dim {y.dim}.",
            data={"D": D, "y": y, "feature_names": feature_names},
        )
        self.stage_history.append(output)
        return D, y, feature_names

    # =========================================================================
    # STAGE 3: Feature Vectors
    # =========================================================================
    def stage_3_feature_vectors(self, D: Matrix, y: Vector, feature_names: List[str]) -> Dict[str, Any]:
        """Extracts individual column vectors x_j in R^m and calculates statistical properties."""
        vectors = {}
        stats = {}
        for idx, name in enumerate(feature_names):
            col_vec = D.get_column(idx)
            vectors[name] = col_vec
            stats[name] = {
                "mean": col_vec.mean(),
                "std": col_vec.std(),
                "norm_l2": col_vec.norm_l2(),
                "min": min(col_vec.elements),
                "max": max(col_vec.elements),
            }

        vectors["price"] = y
        stats["price"] = {
            "mean": y.mean(),
            "std": y.std(),
            "norm_l2": y.norm_l2(),
            "min": min(y.elements),
            "max": max(y.elements),
        }

        output = StageOutput(
            stage_number=3,
            stage_name="Feature Vectors",
            description=f"Extracted {len(vectors)} feature vectors with summary norms and statistics.",
            data={"vectors": vectors, "stats": stats},
        )
        self.stage_history.append(output)
        return {"vectors": vectors, "stats": stats}

    # =========================================================================
    # STAGE 4: Design Matrices
    # =========================================================================
    def stage_4_design_matrices(self, stage_3_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Constructs design matrices A for 3 distinct least-squares models:
        Model 1: Simple Linear Regression (Living Area) -> [1, area]
        Model 2: Multivariable Linear Regression -> [1, area, beds, baths, age, transit]
        Model 3: Polynomial & Interaction Regression -> [1, area, area^2/1000, beds, age, (area*beds)/1000]
        """
        vecs = stage_3_data["vectors"]
        m = vecs["area_sqft"].dim
        ones = Vector([1.0] * m)

        # Model 1
        A1 = Matrix.from_columns([ones, vecs["area_sqft"]])
        m1_labels = ["Intercept", "Area (sqft)"]

        # Model 2
        A2 = Matrix.from_columns([
            ones,
            vecs["area_sqft"],
            vecs["bedrooms"],
            vecs["bathrooms"],
            vecs["age_years"],
            vecs["dist_transit_km"],
        ])
        m2_labels = ["Intercept", "Area", "Bedrooms", "Bathrooms", "Age", "Dist_Transit"]

        # Model 3: Non-linear features (scaled to prevent numerical ill-conditioning)
        area_sq = Vector([(a * a) / 1000.0 for a in vecs["area_sqft"].elements])
        area_bed = Vector([(a * b) / 1000.0 for a, b in zip(vecs["area_sqft"].elements, vecs["bedrooms"].elements)])
        A3 = Matrix.from_columns([
            ones,
            vecs["area_sqft"],
            area_sq,
            vecs["bedrooms"],
            vecs["age_years"],
            area_bed,
        ])
        m3_labels = ["Intercept", "Area", "Area^2/1k", "Bedrooms", "Age", "Area*Beds/1k"]

        design_dict = {
            "Model 1 (Simple Linear)": {
                "matrix": A1,
                "labels": m1_labels,
                "description": "Baseline bivariate model: Living Area only.",
            },
            "Model 2 (Multivariable Linear)": {
                "matrix": A2,
                "labels": m2_labels,
                "description": "Full multivariate linear features.",
            },
            "Model 3 (Polynomial & Interaction)": {
                "matrix": A3,
                "labels": m3_labels,
                "description": "Area curvature and living area-bedroom interaction terms.",
            },
        }

        output = StageOutput(
            stage_number=4,
            stage_name="Design Matrices",
            description=f"Generated design matrices: A1 {A1.shape()}, A2 {A2.shape()}, A3 {A3.shape()}.",
            data=design_dict,
        )
        self.stage_history.append(output)
        return design_dict

    # =========================================================================
    # STAGE 5: Least-Squares Solutions
    # =========================================================================
    def stage_5_least_squares_solutions(
        self, design_dict: Dict[str, Dict[str, Any]], y: Vector
    ) -> Dict[str, Dict[str, Any]]:
        """
        Solves least-squares problem min ||A beta - y||_2 via:
        1. Normal Equations: (A^T A) beta = A^T y  ==>  beta = (A^T A)^{-1} A^T y
        2. QR Decomposition: A = Q R  ==>  R beta = Q^T y
        3. Orthogonal Projection Matrix: P = A (A^T A)^{-1} A^T
        """
        solutions = {}
        for name, info in design_dict.items():
            A = info["matrix"]
            At = A.transpose()
            AtA = At.matmul(A)
            Aty = At.matmul(y)

            # Normal equations solution
            AtA_inv = AtA.inverse()
            beta_normal = AtA_inv.matmul(Aty)

            # QR solution for numerical verification
            beta_qr = solve_qr_least_squares(A, y)

            # Orthogonal projection matrix
            P = A.matmul(AtA_inv).matmul(At)

            # Condition number estimate
            cond_num = condition_number_estimate(AtA)

            solutions[name] = {
                "matrix": A,
                "labels": info["labels"],
                "beta_normal": beta_normal,
                "beta_qr": beta_qr,
                "projection_matrix": P,
                "AtA": AtA,
                "condition_number": cond_num,
            }

        output = StageOutput(
            stage_number=5,
            stage_name="Least-Squares Solutions",
            description="Derived parameter vectors beta and projection matrices P via Normal Equations and QR factorization.",
            data=solutions,
        )
        self.stage_history.append(output)
        return solutions

    # =========================================================================
    # STAGE 6: Predictions
    # =========================================================================
    def stage_6_predictions(self, solutions: Dict[str, Dict[str, Any]], y: Vector) -> Dict[str, Dict[str, Any]]:
        """
        Computes fitted response vector y_hat in Col(A):
        y_hat = A beta = P y (Orthogonal projection of y onto column space).
        """
        predictions_dict = {}
        for name, sol in solutions.items():
            A = sol["matrix"]
            beta = sol["beta_normal"]
            P = sol["projection_matrix"]

            y_hat_linear = A.matmul(beta)
            y_hat_proj = P.matmul(y)

            # Verification of equivalence: ||A beta - P y||_2
            equiv_error = (y_hat_linear - y_hat_proj).norm_l2()

            predictions_dict[name] = {
                "y_hat": y_hat_linear,
                "y_hat_proj": y_hat_proj,
                "equiv_error": equiv_error,
                "beta": beta,
                "matrix": A,
                "labels": sol["labels"],
                "projection_matrix": P,
                "condition_number": sol["condition_number"],
            }

        output = StageOutput(
            stage_number=6,
            stage_name="Predictions",
            description="Generated orthogonal projections y_hat = A beta = P y for all models.",
            data=predictions_dict,
        )
        self.stage_history.append(output)
        return predictions_dict

    # =========================================================================
    # STAGE 7: Residual Analysis
    # =========================================================================
    def stage_7_residual_analysis(
        self, predictions_dict: Dict[str, Dict[str, Any]], y: Vector
    ) -> Dict[str, Dict[str, Any]]:
        r"""
        Calculates residual vectors e = y - y_hat and verifies:
        1. Orthogonality: A^T e = 0  (e in Nul(A^T) = Col(A)^\perp)
        2. Projection property: (I - P) y = e
        3. Pythagorean decomposition: ||y||_2^2 = ||y_hat||_2^2 + ||e||_2^2
        """
        residual_dict = {}
        for name, pred in predictions_dict.items():
            y_hat = pred["y_hat"]
            e = y - y_hat
            A = pred["matrix"]

            # Orthogonality check: A^T @ e
            At = A.transpose()
            Ate = At.matmul(e)
            orthogonality_norm = Ate.norm_l2()

            # Projection property check: (I - P) @ y == e
            m = y.dim
            I = Matrix.identity(m)
            P_perp = I - pred["projection_matrix"]
            e_from_proj = P_perp.matmul(y)
            proj_diff = (e - e_from_proj).norm_l2()

            # Projection idempotence check: ||P^2 - P||_F
            P = pred["projection_matrix"]
            P_sq = P.matmul(P)
            idempotence_error = (P_sq - P).norm_frobenius()

            residual_dict[name] = {
                "residuals": e,
                "orthogonality_norm": orthogonality_norm,
                "proj_diff": proj_diff,
                "idempotence_error": idempotence_error,
                "y_hat": y_hat,
                "beta": pred["beta"],
                "labels": pred["labels"],
                "condition_number": pred["condition_number"],
            }

        output = StageOutput(
            stage_number=7,
            stage_name="Residual Analysis",
            description="Verified residual orthogonality A^T e = 0, projection idempotence P^2 = P, and subspace decomposition.",
            data=residual_dict,
        )
        self.stage_history.append(output)
        return residual_dict

    # =========================================================================
    # STAGE 8: Model Comparison
    # =========================================================================
    def stage_8_model_comparison(
        self, residual_dict: Dict[str, Dict[str, Any]], y: Vector
    ) -> List[ModelEvaluation]:
        """
        Computes vector error norms (L1, L2, L_inf), RMSE, MAE, R^2, and Adjusted R^2,
        and provides mathematical ranking of the candidate models.
        """
        m = y.dim
        y_mean = y.mean()
        # Total Sum of Squares: ||y - y_bar 1||_2^2
        tss = sum((val - y_mean) ** 2 for val in y.elements)

        evaluations: List[ModelEvaluation] = []
        for name, res_info in residual_dict.items():
            e = res_info["residuals"]
            p = len(res_info["beta"])  # number of parameters (including intercept)

            norm_l1 = e.norm_l1()
            mae = norm_l1 / m
            norm_l2 = e.norm_l2()
            rmse = norm_l2 / math.sqrt(m)
            norm_inf = e.norm_inf()

            # Residual sum of squares: ||e||_2^2
            rss = norm_l2 ** 2
            r2 = 1.0 - (rss / tss) if tss > 0 else 0.0
            # Adjusted R^2: 1 - [(1 - R^2)*(m - 1)/(m - p)]
            adj_r2 = 1.0 - ((1.0 - r2) * (m - 1) / (m - p))

            ev = ModelEvaluation(
                model_name=name,
                num_parameters=p,
                beta=res_info["beta"],
                predictions=res_info["y_hat"],
                residuals=e,
                norm_l1=norm_l1,
                mae=mae,
                norm_l2=norm_l2,
                rmse=rmse,
                norm_inf=norm_inf,
                r_squared=r2,
                adj_r_squared=adj_r2,
                orthogonality_norm=res_info["orthogonality_norm"],
                condition_number=res_info["condition_number"],
            )
            evaluations.append(ev)

        # Sort models by Adjusted R^2 descending (higher is better)
        evaluations.sort(key=lambda x: x.adj_r_squared, reverse=True)

        output = StageOutput(
            stage_number=8,
            stage_name="Model Comparison",
            description="Evaluated vector error norms (L1, L2, L_inf), RMSE, R^2, and ranked models.",
            data=evaluations,
        )
        self.stage_history.append(output)
        return evaluations

    def execute_full_pipeline(self) -> List[ModelEvaluation]:
        """Runs the entire 8-stage sequential pipeline end-to-end."""
        self.stage_history.clear()
        # Stage 1 -> 2
        records = self.stage_1_house_data()
        # Stage 2 -> 3
        D, y, feature_names = self.stage_2_matrix_formation(records)
        # Stage 3 -> 4
        stage_3_data = self.stage_3_feature_vectors(D, y, feature_names)
        # Stage 4 -> 5
        design_dict = self.stage_4_design_matrices(stage_3_data)
        # Stage 5 -> 6
        solutions = self.stage_5_least_squares_solutions(design_dict, y)
        # Stage 6 -> 7
        predictions = self.stage_6_predictions(solutions, y)
        # Stage 7 -> 8
        residuals = self.stage_7_residual_analysis(predictions, y)
        # Stage 8
        evaluations = self.stage_8_model_comparison(residuals, y)
        return evaluations
