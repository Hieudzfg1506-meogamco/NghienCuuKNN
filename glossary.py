"""Giải thích tiếng Việt cho 30 chỉ số tế bào học và các thuật ngữ của mô hình."""

from __future__ import annotations

# Mô tả theo tài liệu của bộ dữ liệu Wisconsin Diagnostic Breast Cancer (WDBC).
FEATURES = {
    "radius": ("Bán kính", "Khoảng cách trung bình từ tâm nhân tế bào đến các điểm trên viền. Nhân càng to thì giá trị càng lớn."),
    "texture": ("Kết cấu", "Độ lệch chuẩn của mức xám trong ảnh nhân tế bào. Giá trị cao nghĩa là nhân kém đồng đều."),
    "perimeter": ("Chu vi", "Độ dài đường viền của nhân tế bào."),
    "area": ("Diện tích", "Diện tích của nhân tế bào."),
    "smoothness": ("Độ mịn viền", "Mức biến thiên cục bộ của độ dài bán kính. Viền càng nhấp nhô thì giá trị càng lớn."),
    "compactness": ("Độ đặc", "Tính theo công thức chu vi² / diện tích − 1. Hình dạng càng méo, kém tròn thì giá trị càng lớn."),
    "concavity": ("Độ lõm", "Mức độ nghiêm trọng của các chỗ lõm vào trên đường viền nhân."),
    "concave points": ("Số điểm lõm", "Số lượng các chỗ lõm trên đường viền nhân tế bào."),
    "symmetry": ("Độ đối xứng", "Mức đối xứng của hình dạng nhân tế bào."),
    "fractal dimension": ("Chiều fractal", "Đo độ phức tạp, ngoằn ngoèo của đường viền (xấp xỉ \"đường bờ biển\" − 1)."),
}

SUFFIX = {
    "mean": ("Trung bình", "giá trị trung bình trên tất cả nhân tế bào trong ảnh"),
    "se": ("Sai số chuẩn", "mức dao động của chỉ số giữa các nhân tế bào (standard error)"),
    "worst": ("Xấu nhất", "giá trị lớn nhất, tính bằng trung bình của 3 nhân có giá trị lớn nhất"),
}

GROUP_DESC = {
    "Giá trị trung bình": "Mỗi chỉ số được lấy trung bình trên tất cả nhân tế bào trong ảnh.",
    "Sai số chuẩn": "Cho biết chỉ số dao động nhiều hay ít giữa các nhân tế bào.",
    "Giá trị xấu nhất": "Giá trị lớn nhất của chỉ số (trung bình 3 nhân lớn nhất), phản ánh vùng bất thường nhất.",
}


def parse(col: str) -> tuple[str, str]:
    """Tách tên cột thành (đặc trưng gốc, hậu tố)."""
    text = col.lower().replace("_", " ").strip()
    for suffix in ("mean", "se", "worst"):
        if text.endswith(" " + suffix):
            return text[: -len(suffix) - 1].strip(), suffix
    return text, ""


def feature_info(col: str) -> dict[str, str]:
    base, suffix = parse(col)
    name, desc = FEATURES.get(base, (col, "Chỉ số tế bào học của mẫu."))
    group, group_desc = SUFFIX.get(suffix, ("", ""))
    full = f"{name} ({group.lower()})" if group else name
    return {"name": full, "base": name, "group": group, "desc": desc, "group_desc": group_desc}


def feature_help(col: str) -> str:
    info = feature_info(col)
    text = f"**{info['base']}** – {info['desc']}"
    if info["group"]:
        text += f"\n\nNhóm *{info['group']}*: {info['group_desc']}."
    return text


# (thuật ngữ, tên tiếng Việt, giải thích)
TERMS = [
    ("Benign (B)", "Lành tính", "Khối u không xâm lấn mô xung quanh và không di căn."),
    ("Malignant (M)", "Ác tính", "Khối u ung thư, có thể xâm lấn mô xung quanh và di căn."),
    ("FNA – Fine Needle Aspiration", "Chọc hút tế bào bằng kim nhỏ",
     "Kỹ thuật lấy tế bào từ khối u bằng kim nhỏ. Ảnh nhân tế bào được số hoá để tính ra 30 chỉ số dùng trong phần mềm này."),
    ("KNN – K-Nearest Neighbors", "K láng giềng gần nhất",
     "Thuật toán tìm K ca đã có chẩn đoán giống ca đang xét nhất rồi lấy kết quả theo đa số. Không có bước \"học\" phức tạp, kết quả giải thích được bằng chính các ca tương tự."),
    ("K", "Số láng giềng", "Số ca tương tự được dùng để bỏ phiếu. K nhỏ thì nhạy với nhiễu, K lớn thì kết quả mượt hơn nhưng có thể bỏ sót ca hiếm."),
    ("Euclidean / Manhattan", "Khoảng cách Euclid / Manhattan",
     "Hai cách đo độ \"xa\" giữa hai ca. Euclid là đường chim bay, Manhattan là tổng chênh lệch trên từng chỉ số."),
    ("Uniform / Distance weights", "Bỏ phiếu đồng đều / theo khoảng cách",
     "Đồng đều: mỗi láng giềng 1 phiếu như nhau. Theo khoảng cách: ca càng gần thì phiếu càng nặng."),
    ("Standardization (StandardScaler)", "Chuẩn hoá dữ liệu",
     "Đưa các chỉ số về cùng thang đo (trung bình 0, độ lệch chuẩn 1) để chỉ số có số lớn như diện tích không lấn át chỉ số nhỏ như độ mịn."),
    ("Imputation (median)", "Điền giá trị thiếu", "Nếu thiếu một chỉ số, mô hình thay bằng trung vị của dữ liệu huấn luyện."),
    ("Train / Test split", "Tập huấn luyện / kiểm thử",
     "Dữ liệu chia 80% để mô hình tra cứu (train) và 20% giữ lại để đo hiệu năng thật (test). Ca trong tập test mô hình chưa từng \"thấy\"."),
    ("Cross-validation (5-fold)", "Kiểm định chéo",
     "Chia tập huấn luyện thành 5 phần, lần lượt dùng mỗi phần để kiểm tra phần còn lại. Dùng để chọn tham số mà không đụng vào tập test."),
    ("GridSearchCV", "Tìm tham số tốt nhất", "Thử nhiều tổ hợp K, khoảng cách, cách bỏ phiếu và chọn tổ hợp có điểm kiểm định chéo cao nhất."),
    ("Confusion matrix", "Ma trận nhầm lẫn", "Bảng đếm số ca đoán đúng và đoán sai theo từng loại (TP, TN, FP, FN)."),
    ("TP – True Positive", "Dương tính thật", "Ca ác tính được báo đúng là ác tính."),
    ("TN – True Negative", "Âm tính thật", "Ca lành tính được báo đúng là lành tính."),
    ("FP – False Positive", "Dương tính giả", "Ca lành tính bị báo nhầm là ác tính (gây lo lắng và xét nghiệm thêm không cần thiết)."),
    ("FN – False Negative", "Âm tính giả", "Ca ác tính bị bỏ sót, báo là lành tính. Đây là loại sai nguy hiểm nhất trong sàng lọc."),
    ("Accuracy", "Độ chính xác chung", "Tỉ lệ ca đoán đúng trên tổng số ca. Có thể gây ảo tưởng khi số ca hai loại chênh lệch."),
    ("Recall / Sensitivity", "Độ nhạy",
     "Trong các ca ác tính thật, mô hình phát hiện được bao nhiêu phần trăm. Recall = TP / (TP + FN). Càng cao thì càng ít bỏ sót."),
    ("Precision", "Độ chính xác của cảnh báo",
     "Trong các ca bị báo ác tính, bao nhiêu phần trăm thật sự ác tính. Precision = TP / (TP + FP)."),
    ("Specificity", "Độ đặc hiệu", "Trong các ca lành tính thật, mô hình nhận ra được bao nhiêu phần trăm. Specificity = TN / (TN + FP)."),
    ("F1-score", "Điểm F1", "Trung bình điều hoà của Precision và Recall, dùng khi cần cân bằng giữa hai chỉ số."),
    ("ROC curve / AUC", "Đường cong ROC / diện tích dưới đường cong",
     "ROC cho thấy Recall tăng thế nào khi chấp nhận thêm báo nhầm. AUC càng gần 1 thì mô hình càng phân biệt tốt hai nhóm; 0,5 là đoán ngẫu nhiên."),
    ("Precision–Recall curve / AP", "Đường cong Precision–Recall", "Cho thấy sự đánh đổi giữa Precision và Recall. AP là diện tích dưới đường cong này."),
    ("Threshold", "Ngưỡng cảnh báo",
     "Tỉ lệ phiếu ác tính tối thiểu để phần mềm đưa ra cảnh báo. Hạ ngưỡng thì bắt được nhiều ca ác tính hơn (Recall tăng) nhưng báo nhầm nhiều hơn."),
    ("Vote share / Consensus", "Tỉ lệ phiếu / Mức đồng thuận",
     "Trong K ca giống nhất, tỉ lệ ca ác tính là tỉ lệ phiếu. Đây không phải xác suất mắc bệnh. Đồng thuận cao nghĩa là các ca tương tự cùng một loại; thấp nghĩa là ca nằm ở ranh giới."),
    ("PCA", "Phân tích thành phần chính",
     "Kỹ thuật nén 30 chỉ số xuống 2 trục để vẽ lên mặt phẳng. Chỉ để minh hoạ vị trí tương đối, khoảng cách thật được tính trên đủ 30 chỉ số."),
    ("Permutation importance", "Mức quan trọng của chỉ số",
     "Xáo trộn từng chỉ số và xem độ chính xác giảm bao nhiêu. Giảm càng nhiều thì chỉ số đó càng ảnh hưởng đến kết quả."),
]
