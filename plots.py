"""Các hàm vẽ biểu đồ, dùng chung cho GUI và xuất báo cáo."""

from __future__ import annotations

from matplotlib.figure import Figure

from data_loader import DatasetBundle
from model import METRIC_NAME, WEIGHT_NAME, ModelResult


def class_count_figure(bundle: DatasetBundle) -> Figure:
    fig = Figure(figsize=(5, 3.5), dpi=100)
    ax = fig.add_subplot(111)
    counts = bundle.raw["diagnosis"].value_counts().reindex(["B", "M"], fill_value=0)
    bars = ax.bar(["B - Lành tính", "M - Ác tính"], counts, color=["#4c9f70", "#d95f59"])
    ax.bar_label(bars)
    ax.set_title("Số lượng mẫu theo lớp")
    ax.set_ylabel("Số mẫu")
    fig.tight_layout()
    return fig


def confusion_figure(result: ModelResult) -> Figure:
    fig = Figure(figsize=(5, 3.5), dpi=100)
    ax = fig.add_subplot(111)
    image = ax.imshow(result.matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(result.matrix[i, j]), ha="center", va="center")
    ax.set_xticks([0, 1], ["B", "M"])
    ax.set_yticks([0, 1], ["B", "M"])
    ax.set_xlabel("Nhãn dự đoán")
    ax.set_ylabel("Nhãn thật")
    ax.set_title("Ma trận nhầm lẫn")
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def cv_figure(result: ModelResult) -> Figure:
    fig = Figure(figsize=(6, 3.5), dpi=100)
    ax = fig.add_subplot(111)
    if result.cv_results is None:
        ax.text(0.5, 0.5, "Chưa thực hiện tìm tham số", ha="center", va="center")
        ax.axis("off")
        return fig
    frame = result.cv_results
    for (metric, weights), group in frame.groupby(["param_knn__metric", "param_knn__weights"]):
        group = group.sort_values("param_knn__n_neighbors")
        ax.plot(
            group["param_knn__n_neighbors"].astype(int),
            group["mean_test_score"],
            marker="o",
            markersize=3,
            label=f"{METRIC_NAME[str(metric)]} - {WEIGHT_NAME[str(weights)]}",
        )
    ax.set_xlabel("K")
    ax.set_ylabel("F1-macro CV trung bình")
    ax.set_title("Kiểm định chéo trên tập train")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def scatter_figure(bundle: DatasetBundle, result: ModelResult | None, x_name: str, y_name: str) -> Figure:
    fig = Figure(figsize=(6, 4), dpi=100)
    ax = fig.add_subplot(111)
    labels = bundle.raw["diagnosis"]
    for label, color in [("B", "#4c9f70"), ("M", "#d95f59")]:
        mask = labels == label
        ax.scatter(bundle.X.loc[mask, x_name], bundle.X.loc[mask, y_name], s=16, alpha=0.6, label=f"Thật: {label}", c=color)
    if result is not None and not result.mistakes.empty:
        wrong = result.mistakes.index
        ax.scatter(bundle.X.loc[wrong, x_name], bundle.X.loc[wrong, y_name], s=90, marker="x", c="black", linewidths=2, label="Test dự đoán sai")
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title("Phân tán 2 đặc trưng (mô hình dùng đủ 30 đặc trưng)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
