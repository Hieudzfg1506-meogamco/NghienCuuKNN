"""Chức năng dành cho bác sĩ: dự đoán ca mới, giải thích, lưu lịch sử."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import DatasetBundle
from model import ModelResult

DB_PATH = Path(__file__).resolve().parent / "data" / "history.db"


# ---------------------------------------------------------------- lịch sử ca
def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT, patient_code TEXT, patient_name TEXT, age INTEGER,
            prob_malignant REAL, label TEXT, threshold REAL,
            note TEXT, features TEXT)"""
    )
    return conn


def save_case(code: str, name: str, age: int, outcome: dict, note: str, features: dict) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO cases (created_at, patient_code, patient_name, age, prob_malignant, label, threshold, note, features)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (
                datetime.now().strftime("%Y-%m-%d %H:%M"),
                code, name, age,
                outcome["prob_m"], outcome["label"], outcome["threshold"],
                note, json.dumps(features),
            ),
        )


def load_cases() -> pd.DataFrame:
    with _connect() as conn:
        return pd.read_sql_query(
            "SELECT id, created_at, patient_code, patient_name, age, prob_malignant, label, threshold, note"
            " FROM cases ORDER BY id DESC", conn)


def delete_case(case_id: int) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM cases WHERE id = ?", (case_id,))


# ------------------------------------------------------------------ dự đoán
ALERT = "Cảnh báo: nghi ngờ ác tính"
NO_ALERT = "Chưa có cảnh báo ác tính"


def decide(prob_m: float, threshold: float) -> str:
    """prob_m là tỉ lệ phiếu ác tính của K láng giềng, không phải xác suất mắc bệnh."""
    return ALERT if prob_m >= threshold else NO_ALERT


def consensus(prob_m: float) -> tuple[str, str]:
    """Mức đồng thuận giữa các láng giềng: (nhãn, mức 'high' hoặc 'low')."""
    agree = max(prob_m, 1 - prob_m)
    if agree >= 0.999:
        return "Đồng thuận hoàn toàn", "high"
    if agree >= 0.8:
        return "Đồng thuận cao", "high"
    return "Ranh giới, ý kiến phân tán", "low"


def predict_case(bundle: DatasetBundle, result: ModelResult, row: pd.DataFrame, threshold: float = 0.5) -> dict:
    """Dự đoán một ca. `row` là DataFrame 1 dòng, đúng thứ tự cột của bundle.X."""
    row = row[list(bundle.X.columns)]
    pipe = result.pipeline
    classes = list(pipe.classes_)
    prob_m = float(pipe.predict_proba(row)[0][classes.index(1)])
    knn = pipe.named_steps["knn"]
    transformed = pipe[:-1].transform(row)
    distances, positions = knn.kneighbors(transformed, n_neighbors=result.k, return_distance=True)
    train_rows = result.train_indices[positions[0]]
    labels = bundle.y.iloc[train_rows].to_numpy()
    neighbors = pd.DataFrame(
        {
            "Mã ca tương tự": bundle.ids.iloc[train_rows].to_numpy(),
            "Chẩn đoán đã biết": ["Ác tính (M)" if v == 1 else "Lành tính (B)" for v in labels],
            "Khoảng cách": distances[0],
        }
    )
    consensus_label, consensus_level = consensus(prob_m)
    return {
        "prob_m": prob_m,
        "label": decide(prob_m, threshold),
        "alert": prob_m >= threshold,
        "threshold": threshold,
        "neighbors": neighbors,
        "labels": [int(v) for v in labels],
        "neighbor_index": [int(i) for i in train_rows],
        "n_malignant": int((labels == 1).sum()),
        "k": result.k,
        "consensus": consensus_label,
        "consensus_level": consensus_level,
    }


def predict_batch(bundle: DatasetBundle, result: ModelResult, frame: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    X = frame[list(bundle.X.columns)]
    classes = list(result.pipeline.classes_)
    prob_m = result.pipeline.predict_proba(X)[:, classes.index(1)]
    out = frame.copy()
    out.insert(0, "ti_le_phieu_ac_tinh", np.round(prob_m, 4))
    out.insert(1, "ket_qua", [decide(p, threshold) for p in prob_m])
    return out


def check_columns(bundle: DatasetBundle, frame: pd.DataFrame) -> list[str]:
    return [c for c in bundle.X.columns if c not in frame.columns]


def range_warnings(bundle: DatasetBundle, row: pd.DataFrame, train_indices: np.ndarray) -> list[str]:
    """Cảnh báo chỉ số nằm ngoài khoảng đã gặp trong dữ liệu huấn luyện."""
    train = bundle.X.iloc[train_indices]
    low, high = train.min(), train.max()
    notes = []
    for name in bundle.X.columns:
        value = float(row.iloc[0][name])
        if value < low[name] or value > high[name]:
            notes.append(f"{name} = {value:g} nằm ngoài khoảng dữ liệu huấn luyện ({low[name]:g} – {high[name]:g})")
    return notes


def feature_groups(columns: list[str]) -> dict[str, list[str]]:
    """Nhóm 30 đặc trưng thành mean / se / worst nếu tên cột theo chuẩn Wisconsin."""
    groups = {
        "Giá trị trung bình": [c for c in columns if c.endswith("_mean")],
        "Sai số chuẩn": [c for c in columns if c.endswith("_se")],
        "Giá trị xấu nhất": [c for c in columns if c.endswith("_worst")],
    }
    if sum(len(v) for v in groups.values()) == len(columns):
        return groups
    size = 10
    return {f"Nhóm {i // size + 1}": columns[i:i + size] for i in range(0, len(columns), size)}
