"""Giao diện Streamlit: hỗ trợ bác sĩ tham khảo kết quả KNN trong sàng lọc ung thư vú."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA

import charts
import doctor
import glossary
from data_loader import load_dataset
from model import METRIC_NAME, WEIGHT_NAME, search_model, train_model, validate_result
from plots import class_count_figure, cv_figure

HERE = Path(__file__).resolve().parent
TEAL, RED, GREEN = charts.TEAL, charts.RED, charts.GREEN
DISCLAIMER = (
    "Công cụ hỗ trợ tham khảo, xây dựng cho mục đích nghiên cứu học thuật. "
    "Chưa được kiểm định lâm sàng và **không thay thế** chẩn đoán của bác sĩ."
)
VOTE_NOTE = (
    "**Cách hiểu kết quả:** KNN không tính xác suất mắc bệnh. Phần mềm tìm các ca đã có chẩn đoán giống ca này nhất, "
    "rồi đếm xem bao nhiêu ca trong đó là ác tính. \"Cả K ca đều ác tính\" nghĩa là ca này giống các ca ác tính đã biết, "
    "**không** có nghĩa là chắc chắn 100% bị ung thư."
)

st.set_page_config(page_title="KNN – Hỗ trợ sàng lọc ung thư vú", page_icon="🩺", layout="wide")
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&display=swap');
    html, body, [class*="css"], .stMarkdown, button, input, textarea {{ font-family: 'Be Vietnam Pro', 'Segoe UI', sans-serif; }}
    .block-container {{ padding-top: 1.4rem; max-width: 1280px; }}
    #MainMenu, footer {{ visibility: hidden; }}
    h1, h2, h3, h4 {{ color: #12343B; letter-spacing: -0.01em; }}

    section[data-testid="stSidebar"] {{ background: #FFFFFF; border-right: 1px solid #E1E8EB; }}
    .brand {{ display:flex; align-items:center; gap:10px; padding: 4px 0 14px 0; }}
    .brand .logo {{ background: linear-gradient(135deg, {TEAL}, #145A66); color: white; width: 40px; height: 40px;
                    border-radius: 12px; display:flex; align-items:center; justify-content:center; font-size: 20px; }}
    .brand b {{ color:#12343B; font-size: 1.02rem; display:block; line-height:1.2; }}
    .brand span {{ color:#6B7C85; font-size:.78rem; }}

    .hero {{ background: linear-gradient(120deg, {TEAL} 0%, #145A66 70%, #0E3F4D 100%); color: white;
             padding: 22px 28px; border-radius: 18px; margin-bottom: 18px;
             box-shadow: 0 8px 24px rgba(15,139,141,.18); }}
    .hero h2 {{ color: white; margin: 0; font-size: 1.55rem; }}
    .hero p {{ margin: 6px 0 0 0; opacity: .92; font-size: .95rem; }}

    .step {{ display:flex; align-items:center; gap:10px; margin: 18px 0 8px 0; }}
    .step .n {{ background:{TEAL}; color:white; width:28px; height:28px; border-radius:50%; display:flex;
                align-items:center; justify-content:center; font-weight:700; font-size:.9rem; }}
    .step .t {{ font-size:1.18rem; font-weight:600; color:#12343B; }}

    .result {{ border-radius: 18px; padding: 22px 26px; color: white; margin: 4px 0 10px 0;
               box-shadow: 0 8px 22px rgba(0,0,0,.10); }}
    .result.bad {{ background: linear-gradient(120deg, #C8553D, #A8402B); }}
    .result.good {{ background: linear-gradient(120deg, #2E8B57, #226B43); }}
    .result .head {{ font-size: 1.55rem; font-weight: 700; margin: 0; }}
    .result .sub {{ opacity: .95; margin-top: 4px; }}
    .pill {{ display:inline-block; padding: 3px 12px; border-radius: 999px; font-size:.82rem; font-weight:600;
             background: rgba(255,255,255,.22); margin-top: 10px; }}

    .card {{ background:#fff; border:1px solid #E1E8EB; border-radius:14px; padding:14px 18px; margin: 6px 0 10px 0; }}
    .card .lbl {{ color:#6B7C85; font-size:.82rem; margin-bottom:6px; }}
    .dot {{ display:inline-block; width:22px; height:22px; border-radius:50%; margin: 2px 5px 2px 0; }}
    .dot.m {{ background:{RED}; }} .dot.b {{ background:{GREEN}; }}

    div[data-testid="stMetric"] {{ background:#fff; border:1px solid #E1E8EB; border-left: 4px solid {TEAL};
             border-radius:12px; padding: 10px 16px; box-shadow: 0 1px 2px rgba(18,52,59,.05); }}
    div[data-testid="stMetricValue"] {{ color:#12343B; }}

    button[data-baseweb="tab"] {{ font-weight:600; }}
    .stButton > button, .stDownloadButton > button {{ border-radius: 10px; font-weight: 600; }}
    .stButton > button[kind="primary"] {{ background:{TEAL}; border-color:{TEAL}; padding: .55rem 1.6rem; }}
    div[data-testid="stExpander"] {{ border-radius: 12px; border-color:#E1E8EB; background:#fff; }}
    .foot {{ color:#6B7C85; font-size:.82rem; margin-top: 12px; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def step_header(n: int, text: str) -> None:
    st.markdown(f'<div class="step"><div class="n">{n}</div><div class="t">{text}</div></div>', unsafe_allow_html=True)


# ----------------------------------------------------------------- tải dữ liệu
def _find_default_data() -> Path | None:
    for name in ("archive.zip", "data.csv"):
        if (HERE / name).exists():
            return HERE / name
    return None


@st.cache_resource(show_spinner="Đang đọc và kiểm tra dữ liệu...")
def get_bundle(path: str):
    return load_dataset(Path(path))


@st.cache_resource(show_spinner="Đang huấn luyện mô hình...")
def get_model(path: str, k: int, metric: str, weights: str):
    return train_model(get_bundle(path), k, metric, weights)


@st.cache_resource(show_spinner="Đang tìm tham số tốt nhất (GridSearchCV, có thể mất vài chục giây)...")
def get_best_model(path: str):
    return search_model(get_bundle(path))


@st.cache_resource(show_spinner="Đang chạy kiểm định chéo cho nhiều giá trị K...")
def get_k_sweep(key: str, _bundle, _result):
    return charts.k_sweep(_bundle, _result)


@st.cache_resource(show_spinner="Đang tính mức quan trọng của từng chỉ số...")
def get_importance(key: str, _bundle, _result):
    return charts.importance_frame(_bundle, _result)


PAGE_INFO = {
    "chan-doan": ("Hỗ trợ sàng lọc ung thư vú", "Nhập chỉ số tế bào học, nhận kết quả tham khảo kèm các ca tương tự để đối chiếu."),
    "hang-loat": ("Dự đoán hàng loạt", "Tải lên file CSV nhiều ca và nhận bảng kết quả tham khảo."),
    "lich-su": ("Lịch sử ca khám", "Xem lại, tìm kiếm và xuất các ca đã lưu."),
    "danh-gia": ("Đánh giá thuật toán KNN", "Các số liệu và biểu đồ cho biết mô hình đáng tin đến đâu và sai ở đâu."),
    "thuat-ngu": ("Từ điển thuật ngữ", "Giải thích 30 chỉ số tế bào học và các thuật ngữ tiếng Anh dùng trong phần mềm."),
}


def keep(key: str, default) -> None:
    """Giữ giá trị ô nhập khi người dùng chuyển trang (Streamlit xoá state của widget không được vẽ)."""
    store = st.session_state.setdefault("_keep", {})
    if key not in st.session_state:
        st.session_state[key] = store.get(key, default)


def remember(key: str) -> None:
    st.session_state.setdefault("_keep", {})[key] = st.session_state[key]


with st.sidebar:
    data_path = _find_default_data()
    uploaded = st.file_uploader("Dữ liệu huấn luyện (.zip / .csv)", type=["zip", "csv"])
    if uploaded is not None:
        tmp = Path(tempfile.gettempdir()) / uploaded.name
        tmp.write_bytes(uploaded.getvalue())
        data_path = tmp
    st.divider()
    st.markdown("**Cấu hình mô hình**")
    for key, default in (("k_val", 5), ("metric_val", "euclidean"), ("weights_val", "uniform")):
        st.session_state.setdefault(key, default)
    use_best = st.toggle("Dùng tham số tốt nhất (GridSearch)", key="use_best",
                         help="Tự tìm K, cách đo khoảng cách và cách bỏ phiếu tối ưu theo điểm F1-macro trên tập huấn luyện. "
                              "Các ô bên dưới sẽ hiển thị giá trị tìm được.")
    best_found = None
    if use_best and data_path is not None:
        try:
            best_found = get_best_model(str(data_path))
        except Exception as exc:
            st.error(f"Không chạy được GridSearch: {exc}")
        if best_found is not None:  # ô bên dưới hiện đúng giá trị tối ưu
            st.session_state["k_val"] = best_found.k
            st.session_state["metric_val"] = best_found.metric
            st.session_state["weights_val"] = best_found.weights
    elif st.session_state.get("_was_best"):  # vừa tắt: trả về giá trị người dùng chỉnh tay trước đó
        for key, value in st.session_state.get("_manual", {}).items():
            st.session_state[key] = value
    st.session_state["_was_best"] = use_best
    k = st.slider("K láng giềng", 1, 31, step=2, key="k_val", disabled=use_best,
                  help="Số ca tương tự nhất dùng để bỏ phiếu. K lẻ để tránh hoà phiếu.")
    metric = st.selectbox("Khoảng cách", list(METRIC_NAME), format_func=METRIC_NAME.get, key="metric_val", disabled=use_best,
                          help="Euclid: khoảng cách đường chim bay. Manhattan: tổng chênh lệch từng chỉ số.")
    weights = st.selectbox("Bỏ phiếu", list(WEIGHT_NAME), format_func=WEIGHT_NAME.get, key="weights_val", disabled=use_best,
                           help="Đồng đều: mỗi láng giềng 1 phiếu. Theo khoảng cách: ca càng gần càng nặng phiếu.")
    if use_best and best_found is not None and best_found.best_cv_mean is not None:
        st.caption(f"Đã tự chọn: K = {best_found.k}, {METRIC_NAME[best_found.metric]}, {WEIGHT_NAME[best_found.weights]} "
                   f"(F1-macro kiểm định chéo = {best_found.best_cv_mean:.3f}).")
    if not use_best:
        st.session_state["_manual"] = {"k_val": k, "metric_val": metric, "weights_val": weights}
    threshold = st.slider(
        "Ngưỡng cảnh báo ác tính", 0.10, 0.90, 0.50, 0.05, format="%.2f",
        help="Cảnh báo khi tỉ lệ láng giềng ác tính ≥ ngưỡng. Hạ ngưỡng giúp bắt thêm ca ác tính "
             "(Recall tăng, ít bỏ sót hơn) nhưng sẽ báo nhầm nhiều hơn.",
    )
    st.caption(DISCLAIMER)

def pct(x: float) -> str:
    return "—" if x != x else f"{x:.1%}"


def axis_pickers(prefix: str) -> tuple[str, str]:
    """Hai ô chọn chỉ số cho trục X và Y; giữ lựa chọn khi chuyển trang."""
    for key, default in ((f"{prefix}_x", default_pair[0]), (f"{prefix}_y", default_pair[1])):
        keep(key, default)
        if st.session_state[key] not in columns:
            st.session_state[key] = default
    fmt = lambda c: f"{c}  ·  {glossary.feature_info(c)['name']}"
    c1, c2 = st.columns(2)
    fx = c1.selectbox("Trục ngang (X)", columns, format_func=fmt, key=f"{prefix}_x")
    fy = c2.selectbox("Trục dọc (Y)", columns, format_func=fmt, key=f"{prefix}_y")
    remember(f"{prefix}_x")
    remember(f"{prefix}_y")
    return fx, fy


def pca_figure(row: pd.DataFrame) -> go.Figure:
    pre = result.pipeline[:-1]
    train_z = pre.transform(bundle.X.iloc[result.train_indices])
    pca = PCA(n_components=2).fit(train_z)
    pts = pca.transform(train_z)
    labels = bundle.y.iloc[result.train_indices].to_numpy()
    fig = go.Figure()
    for value, name, color in [(0, "Lành tính (B)", GREEN), (1, "Ác tính (M)", RED)]:
        mask = labels == value
        fig.add_trace(go.Scatter(x=pts[mask, 0], y=pts[mask, 1], mode="markers", name=name,
                                 marker=dict(size=7, color=color, opacity=0.45)))
    new = pca.transform(pre.transform(row))
    fig.add_trace(go.Scatter(x=new[:, 0], y=new[:, 1], mode="markers", name="Ca đang xét",
                             marker=dict(size=20, color="#1B2B34", symbol="star", line=dict(width=1, color="white"))))
    fig.update_xaxes(title="Thành phần 1")
    fig.update_yaxes(title="Thành phần 2")
    return charts._style(fig, "Vị trí ca đang xét so với dữ liệu huấn luyện (PCA 2D, chỉ minh hoạ)", 400)


# ------------------------------------------------------------ trang chẩn đoán
def page_diagnose() -> None:
    groups = doctor.feature_groups(columns)
    test_pos = {str(bundle.ids.iloc[int(i)]): int(i) for i in result.test_indices}

    def load_sample() -> None:
        pos = test_pos.get(str(st.session_state["sample_pick"]))
        if pos is None:
            return
        for feat, value in bundle.X.iloc[pos].items():
            st.session_state[f"f_{feat}"] = float(value) if value == value else float(medians[feat])
        st.session_state.pop("case", None)

    step_header(1, "Thông tin bệnh nhân")
    for key, default in (("p_code", ""), ("p_name", ""), ("p_age", 40), ("p_note", "")):
        keep(key, default)
    c1, c2, c3 = st.columns([1, 2, 1])
    code = c1.text_input("Mã bệnh nhân", key="p_code")
    name = c2.text_input("Họ tên", key="p_name")
    age = c3.number_input("Tuổi", 0, 120, key="p_age", help="Chỉ để ghi hồ sơ, không ảnh hưởng đến kết quả của mô hình.")

    step_header(2, "Chỉ số tế bào học")
    st.caption("Rê chuột vào dấu (?) bên cạnh mỗi ô để xem giải thích chỉ số. Các chỉ số được tính từ ảnh nhân tế bào "
               "lấy bằng chọc hút kim nhỏ (FNA). Xem thêm ở trang **Từ điển thuật ngữ**.")
    test_ids = ["—"] + list(test_pos)[:200]
    st.selectbox("Nạp nhanh một ca mẫu từ tập kiểm thử (để thử nghiệm)", test_ids, key="sample_pick", on_change=load_sample,
                 help="Các ca này mô hình chưa từng dùng để huấn luyện, nên phù hợp để thử khả năng dự đoán.")

    tabs = st.tabs(list(groups))
    for tab, (group, feats) in zip(tabs, groups.items()):
        with tab:
            if group in glossary.GROUP_DESC:
                st.caption(glossary.GROUP_DESC[group])
            cols = st.columns(2)
            for i, feat in enumerate(feats):
                key = f"f_{feat}"
                keep(key, float(medians[feat]))
                info = glossary.feature_info(feat)
                cols[i % 2].number_input(f"{feat}  ·  {info['base']}", min_value=0.0, format="%.5f", key=key,
                                         help=glossary.feature_help(feat))

    note = st.text_area("Ghi chú của bác sĩ (tuỳ chọn)", height=70, key="p_note")
    for key in [f"f_{f}" for f in columns] + ["p_code", "p_name", "p_age", "p_note"]:
        remember(key)
    if st.button("🔍 Phân tích ca này", type="primary"):
        values = {f: st.session_state[f"f_{f}"] for f in columns}
        row = pd.DataFrame([values])
        st.session_state["case"] = {
            "row": row, "values": values, "note": note, "code": code, "name": name, "age": int(age),
            "outcome": doctor.predict_case(bundle, result, row, threshold),
            "warnings": doctor.range_warnings(bundle, row, result.train_indices),
        }

    case = st.session_state.get("case")
    if not case:
        return
    out = case["outcome"]
    alert = out["prob_m"] >= threshold
    out["label"], out["threshold"], out["alert"] = doctor.decide(out["prob_m"], threshold), threshold, alert

    step_header(3, "Kết quả tham khảo")
    cls, head = ("bad", "Cảnh báo: nghi ngờ ác tính") if alert else ("good", "Chưa có cảnh báo ác tính")
    st.markdown(
        f'<div class="result {cls}"><p class="head">{head}</p>'
        f'<div class="sub">{out["n_malignant"]}/{out["k"]} ca giống nhất trong dữ liệu là ác tính · ngưỡng cảnh báo {threshold:.0%}</div>'
        f'<span class="pill">{out["consensus"]}</span></div>',
        unsafe_allow_html=True,
    )
    dots = "".join(f'<span class="dot {"m" if v == 1 else "b"}"></span>' for v in out["labels"])
    st.markdown(
        f'<div class="card"><div class="lbl">Chẩn đoán đã biết của {out["k"]} ca giống nhất '
        f'(<span style="color:{RED}">●</span> ác tính, <span style="color:{GREEN}">●</span> lành tính)</div>{dots}</div>',
        unsafe_allow_html=True,
    )
    if out["consensus_level"] == "low":
        st.warning("Các ca tương tự không thống nhất, ca này nằm ở vùng ranh giới giữa hai nhóm. "
                   "Kết quả kém chắc chắn, cần bác sĩ đánh giá thêm.")
    st.info(VOTE_NOTE)

    m1, m2, m3 = st.columns(3)
    m1.metric("Láng giềng ác tính", f"{out['n_malignant']} / {out['k']}",
              help="Số ca ác tính trong K ca đã biết giống ca này nhất.")
    m2.metric("Tỉ lệ phiếu ác tính", f"{out['prob_m']:.0%}",
              help="Tỉ lệ phiếu của K láng giềng (có tính trọng số nếu chọn bỏ phiếu theo khoảng cách). "
                   "Đây không phải xác suất mắc bệnh.")
    m3.metric("Recall (M) của mô hình", pct(metrics["recall"]),
              help="Đo trên tập kiểm thử, với ngưỡng cảnh báo hiện tại: trong các ca ác tính thật, mô hình bắt đúng bao nhiêu. "
                   "Đây là hiệu năng chung của mô hình, không phải độ chắc chắn của riêng ca này.")

    if case["warnings"]:
        with st.expander(f"⚠️ {len(case['warnings'])} chỉ số nằm ngoài khoảng dữ liệu huấn luyện (kết quả kém tin cậy)"):
            for text in case["warnings"]:
                st.write("• " + text)

    t1, t2, t3 = st.tabs(["Các ca tương tự", "Vị trí trên biểu đồ", "So sánh từng chỉ số"])
    with t1:
        st.caption("Các ca trong dữ liệu có chỉ số giống ca đang xét nhất. Khoảng cách càng nhỏ thì càng giống (0 nghĩa là trùng hoàn toàn).")
        st.dataframe(out["neighbors"].style.format({"Khoảng cách": "{:.3f}"}), hide_index=True, width="stretch")
    with t2:
        modes = ["Chọn 2 chỉ số", "PCA (nén 30 chỉ số)"]
        keep("sc_mode", modes[0])
        mode = st.radio("Cách vẽ", modes, horizontal=True, key="sc_mode")
        remember("sc_mode")
        if mode == modes[0]:
            fx, fy = axis_pickers("dg")
            if fx == fy:
                st.warning("Hãy chọn hai chỉ số khác nhau cho hai trục.")
            else:
                st.plotly_chart(charts.case_scatter(bundle, result, fx, fy, case["row"], out.get("neighbor_index", [])), width="stretch")
                st.caption("Vòng tròn đen là các ca giống ca đang xét nhất. Khoảng cách của KNN được tính trên đủ 30 chỉ số đã chuẩn hoá, "
                           "nên khi chỉ nhìn 2 chỉ số, các ca này có thể không nằm sát ngôi sao. Thử đổi sang cặp chỉ số khác để xem.")
        else:
            st.plotly_chart(pca_figure(case["row"]), width="stretch")
            st.caption("30 chỉ số được nén thành 2 trục chỉ để vẽ. Khoảng cách thật được tính trên đủ 30 chỉ số.")
    with t3:
        feat = st.selectbox("Chọn chỉ số", columns, format_func=lambda c: f"{c}  ·  {glossary.feature_info(c)['name']}")
        st.plotly_chart(charts.feature_hist(bundle, result.train_indices, feat, case["values"][feat]), width="stretch")
        st.caption(glossary.feature_info(feat)["desc"])
    st.markdown(f'<div class="foot">{DISCLAIMER}</div>', unsafe_allow_html=True)

    if st.button("💾 Lưu ca vào lịch sử"):
        doctor.save_case(case["code"], case["name"], case["age"], out, case["note"], case["values"])
        st.success("Đã lưu ca vào lịch sử.")


# --------------------------------------------------------------- hàng loạt
def page_batch() -> None:
    st.write(f"File cần có đủ {len(columns)} cột đặc trưng giống dữ liệu huấn luyện (có thể kèm cột `id`).")
    template = bundle.X.iloc[result.test_indices[:3]].to_csv(index=False).encode("utf-8-sig")
    st.download_button("Tải file mẫu", template, "mau_nhap_lieu.csv", "text/csv")
    file = st.file_uploader("Chọn file CSV", type=["csv"], key="batch")
    if file is None:
        return
    frame = pd.read_csv(file)
    missing = doctor.check_columns(bundle, frame)
    if missing:
        st.error(f"Thiếu {len(missing)} cột: {', '.join(missing[:8])}{'...' if len(missing) > 8 else ''}")
        return
    out = doctor.predict_batch(bundle, result, frame, threshold)
    n_bad = int((out["ti_le_phieu_ac_tinh"] >= threshold).sum())
    a, b, c = st.columns(3)
    a.metric("Số ca", len(out))
    b.metric("Có cảnh báo ác tính", n_bad)
    c.metric("Chưa có cảnh báo", len(out) - n_bad)
    st.dataframe(out.sort_values("ti_le_phieu_ac_tinh", ascending=False), width="stretch", height=380)
    st.caption("ti_le_phieu_ac_tinh là tỉ lệ K láng giềng ác tính, không phải xác suất mắc bệnh.")
    st.download_button("Tải kết quả (CSV)", out.to_csv(index=False).encode("utf-8-sig"), "ket_qua_du_doan.csv", "text/csv")


# ------------------------------------------------------------------- lịch sử
def page_history() -> None:
    st.caption("Dữ liệu lưu cục bộ trên máy này (data/history.db). Cần có quy trình bảo mật riêng nếu dùng với dữ liệu thật.")
    cases = doctor.load_cases()
    if cases.empty:
        st.info("Chưa có ca nào được lưu.")
        return
    keyword = st.text_input("Tìm theo mã hoặc tên bệnh nhân")
    if keyword:
        mask = cases["patient_code"].fillna("").str.contains(keyword, case=False) | cases["patient_name"].fillna("").str.contains(keyword, case=False)
        cases = cases[mask]
    st.dataframe(cases.rename(columns={
        "id": "STT", "created_at": "Thời gian", "patient_code": "Mã BN", "patient_name": "Họ tên", "age": "Tuổi",
        "prob_malignant": "Tỉ lệ phiếu ác tính", "label": "Kết quả", "threshold": "Ngưỡng", "note": "Ghi chú"}),
        hide_index=True, width="stretch")
    st.download_button("Xuất lịch sử (CSV)", cases.to_csv(index=False).encode("utf-8-sig"), "lich_su_ca_kham.csv", "text/csv")
    with st.expander("Xoá một ca"):
        case_id = st.number_input("STT cần xoá", min_value=1, step=1)
        if st.button("Xoá"):
            doctor.delete_case(int(case_id))
            st.rerun()


# --------------------------------------------------------------------- đánh giá
def page_evaluate() -> None:
    st.caption(f"Mô hình: K = {result.k} · khoảng cách {METRIC_NAME[result.metric]} · bỏ phiếu {WEIGHT_NAME[result.weights]} · "
               f"huấn luyện {len(result.train_indices)} ca, kiểm thử {len(result.test_indices)} ca (chia 80/20) · "
               f"ngưỡng cảnh báo {threshold:.0%}")
    m = metrics
    a, b, c = st.columns(3)
    a.metric("Recall (M)", pct(m["recall"]), help="Trong các ca ác tính thật, mô hình bắt đúng bao nhiêu. Càng cao càng ít bỏ sót. Recall = TP / (TP + FN).")
    b.metric("Precision (M)", pct(m["precision"]), help="Trong các ca bị cảnh báo ác tính, bao nhiêu ca thật sự ác tính. Precision = TP / (TP + FP).")
    c.metric("Specificity", pct(m["specificity"]), help="Trong các ca lành tính thật, mô hình nhận ra đúng bao nhiêu. Specificity = TN / (TN + FP).")
    d, e, f = st.columns(3)
    d.metric("Accuracy", pct(m["accuracy"]), help="Tỉ lệ ca đoán đúng trên tổng số ca kiểm thử.")
    e.metric("F1 (M)", pct(m["f1"]), help="Trung bình điều hoà của Precision và Recall.")
    from sklearn.metrics import roc_auc_score
    f.metric("AUC", f"{roc_auc_score(result.y_test, p_test):.3f}",
             help="Diện tích dưới đường cong ROC. 1,0 là phân biệt hoàn hảo, 0,5 là đoán ngẫu nhiên.")
    st.info(f"Quy ước M (ác tính) là lớp dương. Đáng chú ý nhất là **FN = {m['fn']}**: số ca ác tính bị bỏ sót, "
            f"cần giảm tối đa trong sàng lọc. Ngoài ra có FP = {m['fp']} ca lành tính bị cảnh báo nhầm. "
            f"Các số liệu thay đổi theo thanh **ngưỡng cảnh báo** ở thanh bên.")

    t = st.tabs(["Tổng quan", "Phân tán 2 chỉ số", "ROC và Precision–Recall", "Ngưỡng cảnh báo", "Chọn K", "Chỉ số quan trọng", "Ca dự đoán sai", "Dữ liệu"])
    with t[0]:
        l, r = st.columns(2)
        l.plotly_chart(charts.confusion_figure(m), width="stretch")
        r.plotly_chart(charts.vote_hist_figure(result.y_test, p_test, threshold), width="stretch")
        st.caption("**Ma trận nhầm lẫn** (trái) đếm số ca đúng/sai theo từng loại. **Biểu đồ phân bố** (phải) cho thấy phần lớn ca có tỉ lệ phiếu "
                   "rất gần 0% hoặc 100%, tức KNN thường rất \"quyết đoán\"; chỉ vài ca ở giữa là ca ranh giới. "
                   "Vì vậy con số này không nên đọc như xác suất mắc bệnh.")
    with t[1]:
        fx, fy = axis_pickers("ev")
        show_train = st.checkbox("Hiện tập huấn luyện làm nền", value=True)
        if fx == fy:
            st.warning("Hãy chọn hai chỉ số khác nhau cho hai trục.")
        else:
            st.plotly_chart(charts.eval_scatter(bundle, result, fx, fy, p_test, threshold, show_train), width="stretch")
            st.caption("Chấm xanh/đỏ là các ca kiểm thử theo chẩn đoán thật. Dấu **✕** là ca ác tính bị bỏ sót (FN), dấu **◆** là ca lành tính bị báo nhầm (FP). "
                       "Rê chuột vào từng điểm để xem mã ca và tỉ lệ phiếu. Mô hình dùng đủ 30 chỉ số, còn biểu đồ chỉ cho nhìn qua 2 chỉ số, "
                       "nên một ca sai có thể trông bình thường ở cặp này nhưng lại lệch ở các chỉ số khác. Các ca sai thường nằm ở vùng hai nhóm chồng lấn nhau.")

    with t[2]:
        l, r = st.columns(2)
        l.plotly_chart(charts.roc_figure(result.y_test, p_test, threshold), width="stretch")
        r.plotly_chart(charts.pr_figure(result.y_test, p_test, threshold), width="stretch")
        st.caption("Chấm đỏ là vị trí của ngưỡng đang chọn. Đường cong càng ôm sát góc trên bên trái (ROC) hoặc góc trên bên phải (PR) thì mô hình càng tốt. "
                   "Với KNN, đường cong có dạng bậc thang vì tỉ lệ phiếu chỉ nhận một số giá trị rời rạc.")
    with t[3]:
        st.plotly_chart(charts.threshold_figure(result.y_test, p_test, threshold), width="stretch")
        st.caption("Kéo thanh ngưỡng ở thanh bên: hạ ngưỡng thì Recall tăng (ít bỏ sót) nhưng Precision và Specificity giảm (báo nhầm nhiều hơn). "
                   "Việc chọn ngưỡng là quyết định lâm sàng, cần cân nhắc giữa nguy cơ bỏ sót và chi phí xét nghiệm thêm.")
    with t[4]:
        frame = get_k_sweep(model_key, bundle, result)
        st.plotly_chart(charts.k_figure(frame, result.k), width="stretch")
        st.caption("Điểm được tính bằng kiểm định chéo 5-fold trên tập huấn luyện, không dùng tập kiểm thử nên không gây thiên lệch khi chọn K. "
                   "K nhỏ dễ bị nhiễu, K quá lớn có thể làm lẫn ca ác tính vào nhóm lành tính.")
        if result.cv_results is not None:
            with st.expander("Kết quả GridSearchCV"):
                st.pyplot(cv_figure(result))
    with t[5]:
        imp = get_importance(model_key, bundle, result)
        st.plotly_chart(charts.importance_figure(imp), width="stretch")
        st.caption("Mỗi chỉ số được xáo trộn ngẫu nhiên 10 lần trên tập kiểm thử; độ chính xác cân bằng giảm càng nhiều thì chỉ số càng quan trọng. "
                   "Thanh sai số cho biết mức dao động giữa các lần xáo trộn.")
        top = imp.head(5)["feature"].tolist()
        st.write("**Giải thích nhanh các chỉ số đứng đầu:**")
        for feat in top:
            info = glossary.feature_info(feat)
            st.write(f"• `{feat}` ({info['name']}): {info['desc']}")
    with t[6]:
        pred = p_test >= threshold
        wrong = pd.DataFrame({
            "Mã ca": bundle.ids.iloc[result.test_indices].to_numpy(),
            "Thực tế": ["Ác tính (M)" if v == 1 else "Lành tính (B)" for v in result.y_test],
            "Mô hình báo": ["Ác tính (M)" if v else "Lành tính (B)" for v in pred],
            "Tỉ lệ phiếu ác tính": p_test,
            "Loại sai": ["FN – bỏ sót ca ác tính" if (y == 1) else "FP – báo nhầm ca lành tính" for y in result.y_test],
        })[pred != (result.y_test == 1)]
        if wrong.empty:
            st.success("Không có ca kiểm thử nào bị dự đoán sai ở ngưỡng hiện tại.")
        else:
            st.dataframe(wrong.sort_values("Loại sai", ascending=False).style.format({"Tỉ lệ phiếu ác tính": "{:.0%}"}),
                         hide_index=True, width="stretch")
            st.caption("Các ca này đáng xem lại nhất: ca FN là ác tính nhưng mô hình báo lành tính.")
    with t[7]:
        st.pyplot(class_count_figure(bundle))
        with st.expander("Báo cáo kiểm tra dữ liệu"):
            st.text(bundle.report)
        with st.expander("Xem dữ liệu huấn luyện"):
            st.dataframe(bundle.raw, width="stretch", height=300)
    with st.expander("Báo cáo phân loại (dạng văn bản, ở ngưỡng mặc định của scikit-learn)"):
        st.code(result.report_text)


# -------------------------------------------------------------------- thuật ngữ
def page_glossary() -> None:
    query = st.text_input("Tìm thuật ngữ hoặc chỉ số", placeholder="Ví dụ: recall, radius, ROC...").strip().lower()
    st.subheader("30 chỉ số tế bào học")
    st.caption("Mỗi đặc trưng gốc (10 loại) xuất hiện ở 3 nhóm: trung bình (_mean), sai số chuẩn (_se) và xấu nhất (_worst).")
    rows = []
    for col in columns:
        info = glossary.feature_info(col)
        rows.append({"Tên cột": col, "Tên tiếng Việt": info["name"], "Ý nghĩa": info["desc"]})
    frame = pd.DataFrame(rows)
    if query:
        frame = frame[frame.apply(lambda r: query in " ".join(r.astype(str)).lower(), axis=1)]
    st.dataframe(frame, hide_index=True, width="stretch", height=min(35 * (len(frame) + 1) + 3, 420))
    with st.expander("Ba nhóm giá trị _mean, _se, _worst nghĩa là gì?"):
        for key, (name, text) in glossary.SUFFIX.items():
            st.write(f"• **{name}** (`_{key}`): {text}.")

    st.subheader("Thuật ngữ mô hình và đánh giá")
    shown = 0
    for term, vi, text in glossary.TERMS:
        if query and query not in f"{term} {vi} {text}".lower():
            continue
        shown += 1
        with st.expander(f"**{term}**  ·  {vi}"):
            st.write(text)
    if query and not shown and frame.empty:
        st.info("Không tìm thấy thuật ngữ nào khớp.")


# ------------------------------------------------------------------ điều hướng
nav = st.navigation(
    [
        st.Page(page_diagnose, title="Chẩn đoán ca mới", icon="🩺", url_path="chan-doan", default=True),
        st.Page(page_batch, title="Dự đoán hàng loạt", icon="📋", url_path="hang-loat"),
        st.Page(page_history, title="Lịch sử ca khám", icon="🗂️", url_path="lich-su"),
        st.Page(page_evaluate, title="Đánh giá thuật toán", icon="📊", url_path="danh-gia"),
        st.Page(page_glossary, title="Từ điển thuật ngữ", icon="📖", url_path="thuat-ngu"),
    ]
)
title, subtitle = PAGE_INFO.get(nav.url_path, PAGE_INFO["chan-doan"])
st.markdown(f'<div class="hero"><h2>{title}</h2><p>{subtitle}</p></div>', unsafe_allow_html=True)

if data_path is None:
    st.info("Chưa có dữ liệu huấn luyện. Hãy đặt `archive.zip` cạnh `app.py` hoặc tải lên ở thanh bên.")
    st.stop()

try:
    bundle = get_bundle(str(data_path))
    result = get_best_model(str(data_path)) if use_best else get_model(str(data_path), k, metric, weights)
    validate_result(result)
except Exception as exc:  # hiển thị lỗi dữ liệu thân thiện
    st.error(f"Không thể nạp dữ liệu hoặc huấn luyện mô hình: {exc}")
    st.stop()

columns = list(bundle.X.columns)
medians = bundle.X.iloc[result.train_indices].median()
model_key = f"{data_path}|{result.k}|{result.metric}|{result.weights}"
default_pair = charts.separation_pair(bundle, result.train_indices)
p_test = charts.positive_proba(result, bundle.X.iloc[result.test_indices])
metrics = charts.metrics_at(result.y_test, p_test, threshold)

nav.run()
