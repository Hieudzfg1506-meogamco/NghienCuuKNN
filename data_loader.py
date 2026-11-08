"""Đọc và kiểm tra dữ liệu Breast Cancer Wisconsin (Diagnostic)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd


class DataValidationError(ValueError):
    """Lỗi dữ liệu có thể trình bày trực tiếp cho người dùng."""


@dataclass
class DatasetBundle:
    source: Path
    raw: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    ids: pd.Series
    report: str
    removed_empty_columns: list[str]
    duplicate_count: int
    missing_count: int


def _read(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() != ".zip":
        raise DataValidationError("Chỉ hỗ trợ tệp CSV hoặc ZIP.")
    try:
        with ZipFile(path) as archive:
            csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if not csv_names:
                raise DataValidationError("Tệp ZIP không chứa tệp CSV.")
            preferred = next((name for name in csv_names if Path(name).name.lower() == "data.csv"), csv_names[0])
            with archive.open(preferred) as stream:
                return pd.read_csv(stream)
    except BadZipFile as exc:
        raise DataValidationError("Tệp ZIP bị hỏng hoặc không đúng định dạng.") from exc


def load_dataset(path: str | Path) -> DatasetBundle:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise DataValidationError(f"Không tìm thấy tệp: {source}")
    try:
        original = _read(source)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise DataValidationError(f"Không thể đọc dữ liệu: {exc}") from exc

    if original.empty:
        raise DataValidationError("Dữ liệu không có dòng nào.")

    empty_columns = [str(col) for col in original.columns if original[col].isna().all()]
    frame = original.drop(columns=empty_columns).copy()
    frame.columns = [str(col).strip() for col in frame.columns]
    if frame.columns.duplicated().any():
        duplicated = frame.columns[frame.columns.duplicated()].tolist()
        raise DataValidationError(f"Tên cột bị trùng: {', '.join(duplicated)}")

    required = {"id", "diagnosis"}
    missing_required = required.difference(frame.columns)
    if missing_required:
        raise DataValidationError(f"Thiếu cột bắt buộc: {', '.join(sorted(missing_required))}")

    labels = frame["diagnosis"].astype("string").str.strip().str.upper()
    invalid_labels = sorted(set(labels.dropna()) - {"B", "M"})
    if labels.isna().any() or invalid_labels:
        detail = ", ".join(map(str, invalid_labels)) or "giá trị trống"
        raise DataValidationError(f"Cột diagnosis chỉ được chứa B hoặc M; phát hiện: {detail}.")

    feature_names = [col for col in frame.columns if col not in {"id", "diagnosis"}]
    if len(feature_names) != 30:
        raise DataValidationError(
            f"Cần đúng 30 đặc trưng sau khi bỏ id, diagnosis và cột trống; hiện có {len(feature_names)}."
        )
    bad_suffix = [name for name in feature_names if not name.endswith(("_mean", "_se", "_worst"))]
    if bad_suffix:
        raise DataValidationError(f"Tên đặc trưng không đúng nhóm _mean/_se/_worst: {', '.join(bad_suffix)}")

    raw_features = frame[feature_names]
    numeric = raw_features.apply(pd.to_numeric, errors="coerce")
    invalid_mask = raw_features.notna() & numeric.isna()
    invalid_count = int(invalid_mask.sum().sum())
    infinite_count = int(np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).sum())
    if invalid_count or infinite_count:
        locations = []
        for row, col in zip(*np.where(invalid_mask.to_numpy())):
            locations.append(f"dòng {row + 2}, cột {feature_names[col]}")
            if len(locations) == 5:
                break
        where = "; ".join(locations)
        raise DataValidationError(
            f"Phát hiện {invalid_count} giá trị không phải số và {infinite_count} giá trị vô hạn. {where}"
        )

    missing_count = int(numeric.isna().sum().sum())
    duplicate_count = int(frame.duplicated().sum())
    counts = labels.value_counts()
    report_lines = [
        f"Nguồn: {source.name}",
        f"Số mẫu: {len(frame)}",
        f"Số đặc trưng số: {len(feature_names)}",
        f"Nhãn B: {int(counts.get('B', 0))}; nhãn M: {int(counts.get('M', 0))}",
        f"Ô thiếu trong đặc trưng: {missing_count}",
        "Giá trị không hợp lệ: 0",
        f"Dòng trùng hoàn toàn: {duplicate_count} (chỉ báo cáo, không xóa)",
        "Cột hoàn toàn trống đã loại: " + (", ".join(empty_columns) if empty_columns else "không có"),
    ]
    clean_raw = pd.concat(
        [frame[["id"]].reset_index(drop=True), labels.rename("diagnosis").reset_index(drop=True), numeric.reset_index(drop=True)],
        axis=1,
    )
    return DatasetBundle(
        source=source,
        raw=clean_raw,
        X=numeric.reset_index(drop=True),
        y=labels.map({"B": 0, "M": 1}).astype(int).reset_index(drop=True),
        ids=frame["id"].reset_index(drop=True),
        report="\n".join(report_lines),
        removed_empty_columns=empty_columns,
        duplicate_count=duplicate_count,
        missing_count=missing_count,
    )
