"""Số liệu và biểu đồ đánh giá thuật toán KNN (Plotly)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.inspection import permutation_importance
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from data_loader import DatasetBundle
from model import ModelResult, make_pipeline

TEAL, RED, GREEN, GRAY, INK = "#0F8B8D", "#C8553D", "#2E8B57", "#8A9BA3", "#1B2B34"
FONT = "Be Vietnam Pro, Segoe UI, sans-serif"


def _style(fig: go.Figure, title: str, height: int = 360) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=INK)),
        height=height,
        margin=dict(l=10, r=10, t=55, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family=FONT, color=INK),
        legend=dict(orientation="h", y=-0.22),
    )
    fig.update_xaxes(gridcolor="#EDF1F3", zeroline=False)
    fig.update_yaxes(gridcolor="#EDF1F3", zeroline=False)
    return fig


def positive_proba(result: ModelResult, X: pd.DataFrame) -> np.ndarray:
    classes = list(result.pipeline.classes_)
    return result.pipeline.predict_proba(X)[:, classes.index(1)]


def metrics_at(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    """Các chỉ số đánh giá khi coi M (nhãn 1) là lớp dương, cảnh báo nếu prob >= ngưỡng."""
    y = np.asarray(y_true).astype(int)
    pred = np.asarray(prob) >= threshold
    tp = int((pred & (y == 1)).sum())
    fn = int((~pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    tn = int((~pred & (y == 0)).sum())

    def ratio(a: int, b: int) -> float:
        return a / b if b else float("nan")

    recall, precision = ratio(tp, tp + fn), ratio(tp, tp + fp)
    f1 = ratio(2 * precision * recall, precision + recall) if tp else 0.0
    return dict(
        tp=tp, fn=fn, fp=fp, tn=tn,
        accuracy=ratio(tp + tn, len(y)), recall=recall, precision=precision,
        specificity=ratio(tn, tn + fp), f1=f1,
    )


# ------------------------------------------------------------------ biểu đồ
def confusion_figure(m: dict) -> go.Figure:
    z = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
    names = [["TN", "FP"], ["FN", "TP"]]
    notes = [["Lành tính, báo đúng", "Lành tính, báo nhầm"], ["Ác tính, bị bỏ sót", "Ác tính, báo đúng"]]
    text = [[f"<b>{z[i][j]}</b><br>{names[i][j]}<br><span style='font-size:11px'>{notes[i][j]}</span>" for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(
        z=z, x=["Báo lành tính", "Báo ác tính"], y=["Thực tế lành tính", "Thực tế ác tính"],
        text=text, texttemplate="%{text}", showscale=False, xgap=4, ygap=4,
        colorscale=[[0, "#F2F7F8"], [1, TEAL]],
    ))
    fig.update_yaxes(autorange="reversed")
    return _style(fig, "Ma trận nhầm lẫn (tập kiểm thử)", 340)


def roc_figure(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> go.Figure:
    fpr, tpr, _ = roc_curve(y_true, prob)
    area = auc(fpr, tpr)
    m = metrics_at(y_true, prob, threshold)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Đoán ngẫu nhiên", line=dict(color=GRAY, dash="dash")))
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"KNN (AUC = {area:.3f})", line=dict(color=TEAL, width=3), fill="tozeroy", fillcolor="rgba(15,139,141,.10)"))
    fig.add_trace(go.Scatter(x=[1 - m["specificity"]], y=[m["recall"]], mode="markers", name="Ngưỡng đang chọn",
                             marker=dict(size=13, color=RED, line=dict(width=2, color="white"))))
    fig.update_xaxes(title="Tỉ lệ báo nhầm (1 − Specificity)", range=[0, 1])
    fig.update_yaxes(title="Recall – tỉ lệ bắt đúng ca ác tính", range=[0, 1.02])
    return _style(fig, "Đường cong ROC")


def pr_figure(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> go.Figure:
    precision, recall, _ = precision_recall_curve(y_true, prob)
    ap = average_precision_score(y_true, prob)
    m = metrics_at(y_true, prob, threshold)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=f"KNN (AP = {ap:.3f})", line=dict(color=TEAL, width=3), fill="tozeroy", fillcolor="rgba(15,139,141,.10)"))
    if m["precision"] == m["precision"]:
        fig.add_trace(go.Scatter(x=[m["recall"]], y=[m["precision"]], mode="markers", name="Ngưỡng đang chọn",
                                 marker=dict(size=13, color=RED, line=dict(width=2, color="white"))))
    fig.update_xaxes(title="Recall", range=[0, 1.02])
    fig.update_yaxes(title="Precision – tỉ lệ cảnh báo đúng", range=[0, 1.02])
    return _style(fig, "Đường cong Precision – Recall")


def threshold_figure(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> go.Figure:
    grid = np.round(np.linspace(0.05, 1.0, 96), 2)
    rows = [metrics_at(y_true, prob, t) for t in grid]
    fig = go.Figure()
    for key, label, color in [("recall", "Recall (bắt đúng ca ác tính)", RED), ("precision", "Precision (cảnh báo đúng)", TEAL),
                              ("specificity", "Specificity (nhận đúng ca lành tính)", GREEN), ("accuracy", "Accuracy", GRAY)]:
        fig.add_trace(go.Scatter(x=grid, y=[r[key] for r in rows], mode="lines", name=label, line=dict(color=color, width=2.5)))
    fig.add_vline(x=threshold, line_dash="dot", line_color=INK, annotation_text="Ngưỡng đang chọn", annotation_position="top")
    fig.update_xaxes(title="Ngưỡng cảnh báo (tỉ lệ phiếu ác tính tối thiểu)", range=[0.05, 1], tickformat=".0%")
    fig.update_yaxes(title="Giá trị chỉ số", range=[0, 1.05], tickformat=".0%")
    return _style(fig, "Đánh đổi khi thay đổi ngưỡng cảnh báo", 400)


def vote_hist_figure(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> go.Figure:
    y = np.asarray(y_true).astype(int)
    fig = go.Figure()
    for value, name, color in [(0, "Thực tế lành tính", GREEN), (1, "Thực tế ác tính", RED)]:
        fig.add_trace(go.Histogram(x=prob[y == value], name=name, marker_color=color, opacity=0.75,
                                   xbins=dict(start=-0.05, end=1.05, size=0.1)))
    fig.add_vline(x=threshold, line_dash="dot", line_color=INK)
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="Tỉ lệ phiếu ác tính của K ca giống nhất", tickformat=".0%")
    fig.update_yaxes(title="Số ca")
    return _style(fig, "Phân bố tỉ lệ phiếu ác tính trên tập kiểm thử")


def k_sweep(bundle: DatasetBundle, result: ModelResult) -> pd.DataFrame:
    """Kiểm định chéo 5-fold trên tập huấn luyện cho nhiều giá trị K (không dùng tập test)."""
    X = bundle.X.iloc[result.train_indices]
    y = bundle.y.iloc[result.train_indices]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for k in range(1, 32, 2):
        pred = cross_val_predict(make_pipeline(k, result.metric, result.weights), X, y, cv=cv)
        m = metrics_at(y.to_numpy(), pred.astype(float), 0.5)
        rows.append(dict(K=k, accuracy=m["accuracy"], recall=m["recall"], precision=m["precision"]))
    return pd.DataFrame(rows)


def k_figure(frame: pd.DataFrame, current_k: int) -> go.Figure:
    fig = go.Figure()
    for key, label, color in [("recall", "Recall (M)", RED), ("precision", "Precision (M)", TEAL), ("accuracy", "Accuracy", GRAY)]:
        fig.add_trace(go.Scatter(x=frame["K"], y=frame[key], mode="lines+markers", name=label, line=dict(color=color, width=2.5)))
    fig.add_vline(x=current_k, line_dash="dot", line_color=INK, annotation_text=f"K đang dùng = {current_k}", annotation_position="top")
    fig.update_xaxes(title="K (số láng giềng)", dtick=2)
    fig.update_yaxes(title="Điểm kiểm định chéo 5-fold", tickformat=".0%")
    return _style(fig, "Ảnh hưởng của K đến hiệu năng (chỉ dùng tập huấn luyện)", 380)


def importance_frame(bundle: DatasetBundle, result: ModelResult) -> pd.DataFrame:
    X_test = bundle.X.iloc[result.test_indices]
    r = permutation_importance(result.pipeline, X_test, result.y_test, scoring="balanced_accuracy",
                               n_repeats=10, random_state=42, n_jobs=1)
    frame = pd.DataFrame({"feature": X_test.columns, "mean": r.importances_mean, "std": r.importances_std})
    return frame.sort_values("mean", ascending=False).reset_index(drop=True)


def importance_figure(frame: pd.DataFrame, top: int = 10) -> go.Figure:
    top_rows = frame.head(top).iloc[::-1]
    fig = go.Figure(go.Bar(x=top_rows["mean"], y=top_rows["feature"], orientation="h", marker_color=TEAL,
                           error_x=dict(type="data", array=top_rows["std"], color=GRAY)))
    fig.update_xaxes(title="Độ chính xác cân bằng giảm khi xáo trộn chỉ số")
    return _style(fig, f"{top} chỉ số ảnh hưởng nhiều nhất đến kết quả", 400)


def feature_hist(bundle: DatasetBundle, train_indices: np.ndarray, feature: str, value: float) -> go.Figure:
    X = bundle.X.iloc[train_indices][feature]
    y = bundle.y.iloc[train_indices].to_numpy()
    fig = go.Figure()
    for v, name, color in [(0, "Lành tính (B)", GREEN), (1, "Ác tính (M)", RED)]:
        fig.add_trace(go.Histogram(x=X[y == v], name=name, marker_color=color, opacity=0.65, nbinsx=30))
    fig.add_vline(x=value, line_width=3, line_color=INK, annotation_text="Ca đang xét", annotation_position="top")
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title=feature)
    fig.update_yaxes(title="Số ca")
    return _style(fig, f"Chỉ số {feature} của ca đang xét so với dữ liệu huấn luyện", 340)


# ------------------------------------------------- biểu đồ phân tán 2 chỉ số
def separation_pair(bundle: DatasetBundle, train_indices: np.ndarray) -> tuple[str, str]:
    """Chọn sẵn 2 chỉ số tách hai nhóm B/M tốt nhất và không quá tương quan với nhau."""
    X = bundle.X.iloc[train_indices]
    y = bundle.y.iloc[train_indices].to_numpy()
    gap = (X[y == 1].mean() - X[y == 0].mean()).abs() / X.std().replace(0, np.nan)
    ranked = gap.fillna(0).sort_values(ascending=False).index.tolist()
    corr = X.corr().abs()
    for name in ranked[1:]:
        if corr.loc[ranked[0], name] < 0.9:
            return ranked[0], name
    return ranked[0], ranked[1]


def _axis_title(name: str) -> str:
    from glossary import feature_info

    return f"{name} · {feature_info(name)['name']}"


def _label(v: int) -> str:
    return "ác tính (M)" if v == 1 else "lành tính (B)"


def case_scatter(bundle: DatasetBundle, result: ModelResult, fx: str, fy: str,
                 row: pd.DataFrame, neighbor_index: list[int]) -> go.Figure:
    """Dữ liệu huấn luyện theo 2 chỉ số, đánh dấu ca đang xét và K láng giềng."""
    train = bundle.X.iloc[result.train_indices]
    y = bundle.y.iloc[result.train_indices].to_numpy()
    fig = go.Figure()
    for v, name, color in [(0, "Lành tính (B)", GREEN), (1, "Ác tính (M)", RED)]:
        mask = y == v
        fig.add_trace(go.Scatter(x=train[fx][mask], y=train[fy][mask], mode="markers", name=name,
                                 marker=dict(size=7, color=color, opacity=0.45)))
    if neighbor_index:
        nb = bundle.X.iloc[neighbor_index]
        fig.add_trace(go.Scatter(x=nb[fx], y=nb[fy], mode="markers", name=f"{len(nb)} ca giống nhất",
                                 marker=dict(size=15, symbol="circle", color="rgba(0,0,0,0)", line=dict(width=2.5, color=INK))))
    fig.add_trace(go.Scatter(x=[float(row.iloc[0][fx])], y=[float(row.iloc[0][fy])], mode="markers", name="Ca đang xét",
                             marker=dict(size=20, color=INK, symbol="star", line=dict(width=1, color="white"))))
    fig.update_xaxes(title=_axis_title(fx))
    fig.update_yaxes(title=_axis_title(fy))
    return _style(fig, "Vị trí ca đang xét theo 2 chỉ số đã chọn", 430)


def eval_scatter(bundle: DatasetBundle, result: ModelResult, fx: str, fy: str, prob: np.ndarray,
                 threshold: float, show_train: bool = True) -> go.Figure:
    """Ca kiểm thử theo 2 chỉ số, tô màu theo chẩn đoán thật và đánh dấu ca bị dự đoán sai."""
    test = bundle.X.iloc[result.test_indices]
    ids = bundle.ids.iloc[result.test_indices].to_numpy()
    y = np.asarray(result.y_test).astype(int)
    pred = np.asarray(prob) >= threshold
    hover = [f"Mã {i}<br>Thực tế: {_label(t)}<br>Tỉ lệ phiếu ác tính: {p:.0%}" for i, t, p in zip(ids, y, prob)]
    hover = np.array(hover)
    fig = go.Figure()
    if show_train:
        train = bundle.X.iloc[result.train_indices]
        fig.add_trace(go.Scatter(x=train[fx], y=train[fy], mode="markers", name="Tập huấn luyện (nền)", hoverinfo="skip",
                                 marker=dict(size=5, color="#B8C4CA", opacity=0.45)))
    ok = pred == (y == 1)
    for v, name, color in [(0, "Kiểm thử: lành tính, đoán đúng", GREEN), (1, "Kiểm thử: ác tính, đoán đúng", RED)]:
        mask = ok & (y == v)
        fig.add_trace(go.Scatter(x=test[fx][mask], y=test[fy][mask], mode="markers", name=name, hovertext=hover[mask], hoverinfo="text",
                                 marker=dict(size=9, color=color, opacity=0.85, line=dict(width=1, color="white"))))
    fn = (~ok) & (y == 1)
    fp = (~ok) & (y == 0)
    fig.add_trace(go.Scatter(x=test[fx][fn], y=test[fy][fn], mode="markers", name=f"FN: ác tính bị bỏ sót ({int(fn.sum())})",
                             hovertext=hover[fn], hoverinfo="text",
                             marker=dict(size=15, symbol="x", color=RED, line=dict(width=3, color=INK))))
    fig.add_trace(go.Scatter(x=test[fx][fp], y=test[fy][fp], mode="markers", name=f"FP: lành tính bị báo nhầm ({int(fp.sum())})",
                             hovertext=hover[fp], hoverinfo="text",
                             marker=dict(size=14, symbol="diamond", color=GREEN, line=dict(width=2.5, color=INK))))
    fig.update_xaxes(title=_axis_title(fx))
    fig.update_yaxes(title=_axis_title(fy))
    return _style(fig, "Ca kiểm thử theo 2 chỉ số đã chọn, đánh dấu các ca dự đoán sai", 480)
