"""Huấn luyện, đánh giá và giải thích mô hình KNN."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_loader import DatasetBundle


LABEL_NAME = {0: "B", 1: "M"}
METRIC_NAME = {"euclidean": "Euclid", "manhattan": "Manhattan"}
WEIGHT_NAME = {"uniform": "Đồng đều", "distance": "Theo khoảng cách"}


@dataclass
class ModelResult:
    pipeline: Pipeline
    train_indices: np.ndarray
    test_indices: np.ndarray
    y_test: np.ndarray
    y_pred: np.ndarray
    predictions: pd.DataFrame
    mistakes: pd.DataFrame
    accuracy: float
    report_dict: dict
    report_text: str
    matrix: np.ndarray
    k: int
    metric: str
    weights: str
    cv_results: pd.DataFrame | None = None
    best_cv_mean: float | None = None
    best_cv_std: float | None = None


def make_pipeline(k: int, metric: str, weights: str) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights)),
        ]
    )


def fixed_split(bundle: DatasetBundle) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(bundle.X))
    return train_test_split(indices, test_size=0.20, random_state=42, stratify=bundle.y)


def _evaluate(
    bundle: DatasetBundle,
    pipeline: Pipeline,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    k: int,
    metric: str,
    weights: str,
    cv_results: pd.DataFrame | None = None,
    best_cv_mean: float | None = None,
    best_cv_std: float | None = None,
) -> ModelResult:
    X_test = bundle.X.iloc[test_indices]
    y_test = bundle.y.iloc[test_indices].to_numpy()
    y_pred = pipeline.predict(X_test).astype(int)
    ids = bundle.ids.iloc[test_indices].to_numpy()
    predictions = pd.DataFrame(
        {"id": ids, "nhan_that": [LABEL_NAME[v] for v in y_test], "nhan_du_doan": [LABEL_NAME[v] for v in y_pred]},
        index=test_indices,
    )
    mistakes = predictions.loc[y_test != y_pred].copy()
    matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
    report_dict = classification_report(
        y_test, y_pred, labels=[0, 1], target_names=["B", "M"], output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_test, y_pred, labels=[0, 1], target_names=["B", "M"], digits=4, zero_division=0
    )
    return ModelResult(
        pipeline=pipeline,
        train_indices=np.asarray(train_indices),
        test_indices=np.asarray(test_indices),
        y_test=y_test,
        y_pred=y_pred,
        predictions=predictions,
        mistakes=mistakes,
        accuracy=float(accuracy_score(y_test, y_pred)),
        report_dict=report_dict,
        report_text=report_text,
        matrix=matrix,
        k=k,
        metric=metric,
        weights=weights,
        cv_results=cv_results,
        best_cv_mean=best_cv_mean,
        best_cv_std=best_cv_std,
    )


def train_model(bundle: DatasetBundle, k: int = 5, metric: str = "euclidean", weights: str = "uniform") -> ModelResult:
    train_indices, test_indices = fixed_split(bundle)
    if not 1 <= k <= len(train_indices):
        raise ValueError(f"K phải từ 1 đến {len(train_indices)}.")
    pipeline = make_pipeline(k, metric, weights)
    pipeline.fit(bundle.X.iloc[train_indices], bundle.y.iloc[train_indices])
    return _evaluate(bundle, pipeline, train_indices, test_indices, k, metric, weights)


def search_model(bundle: DatasetBundle) -> ModelResult:
    train_indices, test_indices = fixed_split(bundle)
    pipeline = make_pipeline(5, "euclidean", "uniform")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        pipeline,
        {
            "knn__n_neighbors": list(range(1, 32, 2)),
            "knn__metric": ["euclidean", "manhattan"],
            "knn__weights": ["uniform", "distance"],
        },
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        return_train_score=False,
        refit=True,
    )
    grid.fit(bundle.X.iloc[train_indices], bundle.y.iloc[train_indices])
    params = grid.best_params_
    cv_frame = pd.DataFrame(grid.cv_results_)
    best_row = cv_frame.iloc[int(grid.best_index_)]
    return _evaluate(
        bundle,
        grid.best_estimator_,
        train_indices,
        test_indices,
        int(params["knn__n_neighbors"]),
        str(params["knn__metric"]),
        str(params["knn__weights"]),
        cv_results=cv_frame,
        best_cv_mean=float(best_row["mean_test_score"]),
        best_cv_std=float(best_row["std_test_score"]),
    )


def neighbor_details(bundle: DatasetBundle, result: ModelResult, original_index: int) -> tuple[pd.DataFrame, dict[str, float]]:
    preprocess = result.pipeline[:-1]
    knn: KNeighborsClassifier = result.pipeline.named_steps["knn"]
    transformed = preprocess.transform(bundle.X.iloc[[original_index]])
    distances, positions = knn.kneighbors(transformed, n_neighbors=result.k, return_distance=True)
    positions = positions[0]
    distances = distances[0]
    train_rows = result.train_indices[positions]
    labels = bundle.y.iloc[train_rows].to_numpy()
    if result.weights == "uniform":
        actual_weights = np.ones_like(distances, dtype=float)
    elif np.any(distances == 0):
        actual_weights = (distances == 0).astype(float)
    else:
        actual_weights = 1.0 / distances
    details = pd.DataFrame(
        {
            "id": bundle.ids.iloc[train_rows].to_numpy(),
            "nhan": [LABEL_NAME[v] for v in labels],
            "khoang_cach": distances,
            "trong_so": actual_weights,
        }
    )
    votes = {
        "B": float(actual_weights[labels == 0].sum()),
        "M": float(actual_weights[labels == 1].sum()),
    }
    return details, votes


def validate_result(result: ModelResult) -> None:
    if int(result.matrix.sum()) != len(result.test_indices):
        raise AssertionError("Tổng ma trận nhầm lẫn không bằng số mẫu test.")
    if len(result.mistakes) != int((result.y_test != result.y_pred).sum()):
        raise AssertionError("Số mẫu sai không khớp bảng dự đoán sai.")
