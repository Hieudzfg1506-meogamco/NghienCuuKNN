"""Giao diện Tkinter cho ứng dụng phân loại KNN."""

from __future__ import annotations

import queue
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from data_loader import DataValidationError, DatasetBundle, load_dataset
from model import METRIC_NAME, WEIGHT_NAME, ModelResult, neighbor_details, search_model, train_model, validate_result
from plots import class_count_figure, confusion_figure, cv_figure, scatter_figure


class KNNApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Phân loại dữ liệu bằng K-Nearest Neighbors")
        self.geometry("1200x800")
        self.minsize(980, 680)
        self.bundle: DatasetBundle | None = None
        self.result: ModelResult | None = None
        self._canvases: dict[str, FigureCanvasTkAgg] = {}
        self._work_queue: queue.Queue = queue.Queue()
        self._busy = False
        self._build_style()
        self._build_ui()
        self.after(100, self._poll_work)
        default_zip = Path(__file__).resolve().parent / "archive.zip"
        if default_zip.exists():
            self.after(250, lambda: self._start_load(default_zip))

    def _build_style(self) -> None:
        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"))
        style.configure("Heading.TLabel", font=("Segoe UI", 11, "bold"))
        style.configure("Status.TLabel", padding=5)

    def _build_ui(self) -> None:
        top = ttk.Frame(self, padding=(12, 8))
        top.pack(fill="x")
        ttk.Label(top, text="Phân loại dữ liệu bằng K-Nearest Neighbors", style="Title.TLabel").pack(side="left")
        self.status_var = tk.StringVar(value="Sẵn sàng")
        ttk.Label(top, textvariable=self.status_var, style="Status.TLabel").pack(side="right")
        self.progress = ttk.Progressbar(self, mode="indeterminate", length=160)
        self.progress.pack(fill="x", padx=12)

        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True, padx=12, pady=8)
        self.data_tab = ttk.Frame(self.tabs, padding=8)
        self.train_tab = ttk.Frame(self.tabs, padding=8)
        self.chart_tab = ttk.Frame(self.tabs, padding=8)
        self.tabs.add(self.data_tab, text="Dữ liệu")
        self.tabs.add(self.train_tab, text="Huấn luyện và đánh giá")
        self.tabs.add(self.chart_tab, text="Biểu đồ và dự đoán sai")
        self._build_data_tab()
        self._build_train_tab()
        self._build_chart_tab()

    @staticmethod
    def _tree(parent, columns: list[str], height: int = 10) -> tuple[ttk.Frame, ttk.Treeview]:
        frame = ttk.Frame(parent)
        tree = ttk.Treeview(frame, columns=columns, show="headings", height=height)
        vertical = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        horizontal = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        tree.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=120, minwidth=80, anchor="center")
        return frame, tree

    def _build_data_tab(self) -> None:
        controls = ttk.Frame(self.data_tab)
        controls.pack(fill="x")
        self.open_button = ttk.Button(controls, text="Mở dữ liệu", command=self._choose_data)
        self.open_button.pack(side="left")
        self.summary_var = tk.StringVar(value="Chưa tải dữ liệu")
        ttk.Label(controls, textvariable=self.summary_var, style="Heading.TLabel").pack(side="left", padx=16)
        self.validation_text = tk.Text(self.data_tab, height=8, wrap="word", font=("Consolas", 9))
        self.validation_text.pack(fill="x", pady=(8, 6))
        sub_tabs = ttk.Notebook(self.data_tab)
        sub_tabs.pack(fill="both", expand=True)
        preview_page, stats_page = ttk.Frame(sub_tabs), ttk.Frame(sub_tabs)
        sub_tabs.add(preview_page, text="Xem dữ liệu")
        sub_tabs.add(stats_page, text="Thống kê đặc trưng")
        self.preview_host = preview_page
        self.stats_frame, self.stats_tree = self._tree(
            stats_page, ["Đặc trưng", "Trung bình", "Độ lệch chuẩn", "Nhỏ nhất", "Lớn nhất"], 15
        )
        self.stats_frame.pack(fill="both", expand=True)

    def _build_train_tab(self) -> None:
        settings = ttk.LabelFrame(self.train_tab, text="Cấu hình", padding=10)
        settings.pack(fill="x")
        ttk.Label(settings, text="K:").grid(row=0, column=0, padx=4)
        self.k_var = tk.IntVar(value=5)
        ttk.Spinbox(settings, from_=1, to=99, increment=1, textvariable=self.k_var, width=7).grid(row=0, column=1, padx=4)
        ttk.Label(settings, text="Khoảng cách:").grid(row=0, column=2, padx=(16, 4))
        self.metric_var = tk.StringVar(value="Euclid")
        ttk.Combobox(settings, values=["Euclid", "Manhattan"], textvariable=self.metric_var, state="readonly", width=12).grid(row=0, column=3)
        ttk.Label(settings, text="Bỏ phiếu:").grid(row=0, column=4, padx=(16, 4))
        self.weight_var = tk.StringVar(value="Đồng đều")
        ttk.Combobox(settings, values=["Đồng đều", "Theo khoảng cách"], textvariable=self.weight_var, state="readonly", width=18).grid(row=0, column=5)
        self.run_button = ttk.Button(settings, text="Chạy KNN", command=self._run_manual)
        self.run_button.grid(row=0, column=6, padx=(20, 4))
        self.search_button = ttk.Button(settings, text="Tìm tham số phù hợp", command=self._run_search)
        self.search_button.grid(row=0, column=7, padx=4)
        self.export_button = ttk.Button(settings, text="Xuất báo cáo", command=self._export, state="disabled")
        self.export_button.grid(row=0, column=8, padx=4)
        ttk.Label(
            self.train_tab,
            text="Accuracy: tỷ lệ đúng tổng thể. Precision: độ chính xác của dự đoán từng lớp. "
            "Recall: tỷ lệ mẫu thật được tìm đúng. F1: cân bằng Precision và Recall. Support: số mẫu thật.",
            wraplength=1100,
        ).pack(fill="x", pady=8)
        self.result_text = tk.Text(self.train_tab, wrap="none", font=("Consolas", 10))
        ybar = ttk.Scrollbar(self.train_tab, orient="vertical", command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=ybar.set)
        self.result_text.pack(side="left", fill="both", expand=True)
        ybar.pack(side="right", fill="y")

    def _build_chart_tab(self) -> None:
        top = ttk.Frame(self.chart_tab)
        top.pack(fill="x")
        ttk.Label(top, text="Trục X:").pack(side="left")
        self.x_var = tk.StringVar()
        self.x_combo = ttk.Combobox(top, textvariable=self.x_var, state="readonly", width=25)
        self.x_combo.pack(side="left", padx=4)
        ttk.Label(top, text="Trục Y:").pack(side="left", padx=(10, 0))
        self.y_var = tk.StringVar()
        self.y_combo = ttk.Combobox(top, textvariable=self.y_var, state="readonly", width=25)
        self.y_combo.pack(side="left", padx=4)
        ttk.Button(top, text="Cập nhật phân tán", command=self._refresh_scatter).pack(side="left", padx=8)
        ttk.Label(top, text="Biểu đồ 2D chỉ minh họa; mô hình sử dụng toàn bộ 30 đặc trưng.").pack(side="left", padx=10)

        paned = ttk.Panedwindow(self.chart_tab, orient="vertical")
        paned.pack(fill="both", expand=True, pady=(8, 0))
        plot_area = ttk.Frame(paned)
        wrong_area = ttk.Frame(paned)
        paned.add(plot_area, weight=3)
        paned.add(wrong_area, weight=2)
        self.plot_tabs = ttk.Notebook(plot_area)
        self.plot_tabs.pack(fill="both", expand=True)
        self.plot_hosts = {}
        for key, title in [("class", "Phân bố lớp"), ("confusion", "Ma trận nhầm lẫn"), ("cv", "Kiểm định chéo"), ("scatter", "Phân tán")]:
            host = ttk.Frame(self.plot_tabs)
            self.plot_tabs.add(host, text=title)
            self.plot_hosts[key] = host

        wrong_left = ttk.LabelFrame(wrong_area, text="Mẫu test dự đoán sai", padding=4)
        wrong_left.pack(side="left", fill="both", expand=True)
        frame, self.wrong_tree = self._tree(wrong_left, ["id", "Nhãn thật", "Nhãn dự đoán"], 7)
        frame.pack(fill="both", expand=True)
        self.wrong_tree.bind("<<TreeviewSelect>>", self._show_mistake)
        detail_tabs = ttk.Notebook(wrong_area)
        detail_tabs.pack(side="left", fill="both", expand=True, padx=(8, 0))
        feature_page, neighbor_page = ttk.Frame(detail_tabs), ttk.Frame(detail_tabs)
        detail_tabs.add(feature_page, text="30 đặc trưng gốc")
        detail_tabs.add(neighbor_page, text="K láng giềng")
        frame, self.feature_tree = self._tree(feature_page, ["Đặc trưng", "Giá trị"], 7)
        frame.pack(fill="both", expand=True)
        frame, self.neighbor_tree = self._tree(neighbor_page, ["id", "Nhãn", "Khoảng cách", "Trọng số"], 6)
        frame.pack(fill="both", expand=True)
        self.vote_var = tk.StringVar(value="Chọn một mẫu sai để xem giải thích.")
        ttk.Label(neighbor_page, textvariable=self.vote_var, wraplength=500).pack(fill="x")

    def _set_busy(self, busy: bool, text: str = "Sẵn sàng") -> None:
        self._busy = busy
        self.status_var.set(text)
        state = "disabled" if busy else "normal"
        self.open_button.configure(state=state)
        self.run_button.configure(state=state)
        self.search_button.configure(state=state)
        if busy:
            self.progress.start(12)
        else:
            self.progress.stop()
            if self.result is not None:
                self.export_button.configure(state="normal")

    def _background(self, label: str, task, success) -> None:
        if self._busy:
            return
        self._set_busy(True, label)
        def worker() -> None:
            try:
                self._work_queue.put(("ok", task(), success))
            except Exception as exc:  # chuyển lỗi về luồng giao diện
                self._work_queue.put(("error", exc, traceback.format_exc()))
        threading.Thread(target=worker, daemon=True).start()

    def _poll_work(self) -> None:
        try:
            kind, payload, extra = self._work_queue.get_nowait()
        except queue.Empty:
            self.after(100, self._poll_work)
            return
        self._set_busy(False)
        if kind == "ok":
            try:
                extra(payload)
            except Exception as exc:
                messagebox.showerror("Lỗi hiển thị", str(exc))
        else:
            title = "Dữ liệu không hợp lệ" if isinstance(payload, DataValidationError) else "Không thể xử lý"
            messagebox.showerror(title, str(payload))
        self.after(100, self._poll_work)

    def _choose_data(self) -> None:
        path = filedialog.askopenfilename(title="Chọn dữ liệu", filetypes=[("Dữ liệu", "*.zip *.csv"), ("ZIP", "*.zip"), ("CSV", "*.csv")])
        if path:
            self._start_load(Path(path))

    def _start_load(self, path: Path) -> None:
        self._background("Đang đọc và kiểm tra dữ liệu...", lambda: load_dataset(path), self._display_dataset)

    def _clear_tree(self, tree: ttk.Treeview) -> None:
        tree.delete(*tree.get_children())

    def _display_dataset(self, bundle: DatasetBundle) -> None:
        self.bundle = bundle
        self.result = None
        self.export_button.configure(state="disabled")
        counts = bundle.raw["diagnosis"].value_counts()
        self.summary_var.set(f"{len(bundle.X)} mẫu | 30 đặc trưng | B: {counts.get('B', 0)} | M: {counts.get('M', 0)}")
        self.validation_text.delete("1.0", "end")
        self.validation_text.insert("1.0", bundle.report)
        for child in self.preview_host.winfo_children():
            child.destroy()
        columns = list(bundle.raw.columns)
        frame, tree = self._tree(self.preview_host, columns, 16)
        frame.pack(fill="both", expand=True)
        for col in columns:
            tree.column(col, width=125, anchor="center")
        for row in bundle.raw.itertuples(index=False, name=None):
            tree.insert("", "end", values=["" if value != value else value for value in row])
        self._clear_tree(self.stats_tree)
        stats = bundle.X.describe().T
        for name, row in stats.iterrows():
            self.stats_tree.insert("", "end", values=(name, f"{row['mean']:.6g}", f"{row['std']:.6g}", f"{row['min']:.6g}", f"{row['max']:.6g}"))
        names = list(bundle.X.columns)
        self.x_combo.configure(values=names)
        self.y_combo.configure(values=names)
        self.x_var.set(names[0])
        self.y_var.set(names[1])
        self._show_figure("class", class_count_figure(bundle))
        self._show_figure("scatter", scatter_figure(bundle, None, names[0], names[1]))
        self._show_figure("confusion", confusion_figure_placeholder())
        self._show_figure("cv", cv_figure_placeholder())
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", "Dữ liệu đã sẵn sàng. Chọn cấu hình và nhấn Chạy KNN, hoặc tìm tham số phù hợp.")
        self._clear_tree(self.wrong_tree)
        self.status_var.set(f"Đã tải {bundle.source.name}")

    def _run_manual(self) -> None:
        if self.bundle is None:
            messagebox.showwarning("Chưa có dữ liệu", "Hãy mở dữ liệu trước.")
            return
        try:
            k = int(self.k_var.get())
        except (ValueError, tk.TclError):
            messagebox.showerror("K không hợp lệ", "K phải là một số nguyên dương.")
            return
        metric = "euclidean" if self.metric_var.get() == "Euclid" else "manhattan"
        weights = "uniform" if self.weight_var.get() == "Đồng đều" else "distance"
        self._background("Đang huấn luyện và đánh giá KNN...", lambda: train_model(self.bundle, k, metric, weights), self._display_result)

    def _run_search(self) -> None:
        if self.bundle is None:
            messagebox.showwarning("Chưa có dữ liệu", "Hãy mở dữ liệu trước.")
            return
        self._background("Đang GridSearchCV (5 fold), vui lòng chờ...", lambda: search_model(self.bundle), self._display_result)

    def _display_result(self, result: ModelResult) -> None:
        validate_result(result)
        self.result = result
        self.k_var.set(result.k)
        self.metric_var.set(METRIC_NAME[result.metric])
        self.weight_var.set(WEIGHT_NAME[result.weights])
        tn, fp, fn, tp = result.matrix.ravel()
        lines = [
            "CẤU HÌNH ĐÃ CHẠY",
            f"K = {result.k}; khoảng cách = {METRIC_NAME[result.metric]}; bỏ phiếu = {WEIGHT_NAME[result.weights]}",
            "Chia dữ liệu: 80/20, random_state=42, stratify=y",
            f"Train: {len(result.train_indices)} mẫu; Test: {len(result.test_indices)} mẫu",
        ]
        if result.best_cv_mean is not None:
            lines += [
                "",
                "KẾT QUẢ CHỌN THAM SỐ (chỉ trên tập train)",
                f"F1-macro CV trung bình = {result.best_cv_mean:.6f}; độ lệch chuẩn = {result.best_cv_std:.6f}",
                "Điểm test không được dùng để chọn tham số.",
            ]
        lines += [
            "",
            "KẾT QUẢ TRÊN TẬP TEST",
            f"Accuracy = {result.accuracy:.6f}",
            f"Quy ước M là lớp dương: TP={tp}, TN={tn}, FP={fp}, FN={fn}",
            "",
            "CLASSIFICATION REPORT",
            result.report_text,
            "Lưu ý: đây là đánh giá mô hình học máy trên dữ liệu, không phải chẩn đoán y khoa.",
        ]
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", "\n".join(lines))
        self._show_figure("confusion", confusion_figure(result))
        self._show_figure("cv", cv_figure(result))
        self._refresh_scatter()
        self._clear_tree(self.wrong_tree)
        if result.mistakes.empty:
            self.vote_var.set("Không có mẫu test nào bị dự đoán sai.")
        else:
            for original_index, row in result.mistakes.iterrows():
                self.wrong_tree.insert("", "end", iid=str(original_index), values=(row["id"], row["nhan_that"], row["nhan_du_doan"]))
            self.vote_var.set(f"Có {len(result.mistakes)} mẫu sai. Chọn một dòng để xem chi tiết.")
        self.export_button.configure(state="normal")
        self.tabs.select(self.train_tab)
        self.status_var.set("Hoàn tất huấn luyện và đánh giá")

    def _show_figure(self, key: str, figure) -> None:
        old = self._canvases.pop(key, None)
        if old is not None:
            old.get_tk_widget().destroy()
        canvas = FigureCanvasTkAgg(figure, master=self.plot_hosts[key])
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self._canvases[key] = canvas

    def _refresh_scatter(self) -> None:
        if self.bundle is None or not self.x_var.get() or not self.y_var.get():
            return
        self._show_figure("scatter", scatter_figure(self.bundle, self.result, self.x_var.get(), self.y_var.get()))

    def _show_mistake(self, _event=None) -> None:
        if self.bundle is None or self.result is None or not self.wrong_tree.selection():
            return
        original_index = int(self.wrong_tree.selection()[0])
        self._clear_tree(self.feature_tree)
        for name, value in self.bundle.X.iloc[original_index].items():
            self.feature_tree.insert("", "end", values=(name, value))
        details, votes = neighbor_details(self.bundle, self.result, original_index)
        self._clear_tree(self.neighbor_tree)
        for row in details.itertuples(index=False):
            self.neighbor_tree.insert("", "end", values=(row.id, row.nhan, f"{row.khoang_cach:.6f}", f"{row.trong_so:.6f}"))
        predicted = self.result.predictions.loc[original_index, "nhan_du_doan"]
        self.vote_var.set(
            f"Tổng phiếu/trọng số: B={votes['B']:.6f}, M={votes['M']:.6f}. Mô hình chọn {predicted} dựa trên "
            f"{self.result.k} láng giềng sau đúng bước điền thiếu và chuẩn hóa đã fit từ train. "
            "Đây là giải thích thuật toán, không phải kết luận nguyên nhân y khoa."
        )

    def _export(self) -> None:
        if self.bundle is None or self.result is None:
            return
        selected = filedialog.askdirectory(title="Chọn thư mục xuất báo cáo")
        if not selected:
            return
        try:
            output = Path(selected)
            self.result.predictions.reset_index(drop=True).to_csv(output / "du_doan_tap_test.csv", index=False, encoding="utf-8-sig")
            self.result.mistakes.reset_index(drop=True).to_csv(output / "du_doan_sai.csv", index=False, encoding="utf-8-sig")
            if self.result.cv_results is not None:
                self.result.cv_results.to_csv(output / "ket_qua_grid_search.csv", index=False, encoding="utf-8-sig")
            (output / "bao_cao.txt").write_text(self.result_text.get("1.0", "end-1c"), encoding="utf-8")
            class_count_figure(self.bundle).savefig(output / "phan_bo_lop.png", dpi=160)
            confusion_figure(self.result).savefig(output / "ma_tran_nham_lan.png", dpi=160)
            cv_figure(self.result).savefig(output / "kiem_dinh_cheo.png", dpi=160)
            scatter_figure(self.bundle, self.result, self.x_var.get(), self.y_var.get()).savefig(output / "bieu_do_phan_tan.png", dpi=160)
            messagebox.showinfo("Xuất thành công", f"Đã lưu báo cáo vào:\n{output}")
        except OSError as exc:
            messagebox.showerror("Không thể xuất", str(exc))


def confusion_figure_placeholder():
    from matplotlib.figure import Figure
    fig = Figure(figsize=(5, 3.5), dpi=100)
    ax = fig.add_subplot(111)
    ax.text(0.5, 0.5, "Chưa chạy mô hình", ha="center", va="center")
    ax.axis("off")
    return fig


def cv_figure_placeholder():
    from matplotlib.figure import Figure
    fig = Figure(figsize=(6, 3.5), dpi=100)
    ax = fig.add_subplot(111)
    ax.text(0.5, 0.5, "Chưa thực hiện tìm tham số", ha="center", va="center")
    ax.axis("off")
    return fig
