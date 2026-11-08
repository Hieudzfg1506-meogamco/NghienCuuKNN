# Hỗ trợ sàng lọc ung thư vú bằng KNN – Breast Cancer Wisconsin

Ứng dụng web chạy cục bộ bằng Python và Streamlit, dùng dữ liệu thật trong `archive.zip` (tệp `data.csv`). Chương trình đọc và kiểm tra dữ liệu, huấn luyện K-Nearest Neighbors, tìm tham số bằng kiểm định chéo, đánh giá trên tập test, và có giao diện cho bác sĩ nhập ca mới để nhận kết quả tham khảo.

> Kết quả của chương trình chỉ phục vụ học tập và nghiên cứu về học máy, **không phải chẩn đoán y khoa** và chưa được kiểm định lâm sàng.

## Chức năng

| Trang | Nội dung |
|---|---|
| Chẩn đoán ca mới | Nhập thông tin bệnh nhân và 30 chỉ số tế bào học (có chú thích từng chỉ số), nhận cảnh báo kèm số láng giềng ác tính, mức đồng thuận, bảng các ca tương tự, biểu đồ vị trí (chọn 2 chỉ số hoặc PCA) và so sánh từng chỉ số |
| Dự đoán hàng loạt | Tải CSV nhiều ca, nhận bảng kết quả và tải về |
| Lịch sử ca khám | Lưu, tìm kiếm, xuất CSV và xoá các ca (lưu cục bộ bằng SQLite) |
| Đánh giá thuật toán | Recall, Precision, Specificity, Accuracy, F1, AUC; ma trận nhầm lẫn; phân bố tỉ lệ phiếu; phân tán 2 chỉ số; ROC và Precision–Recall; ảnh hưởng của ngưỡng cảnh báo; chọn K bằng kiểm định chéo; mức quan trọng của chỉ số; danh sách ca dự đoán sai |
| Từ điển thuật ngữ | Giải thích 30 chỉ số và các thuật ngữ tiếng Anh dùng trong phần mềm |

Thanh bên cho chọn K, khoảng cách, cách bỏ phiếu, bật GridSearch (các ô tự hiển thị tham số tìm được) và chỉnh **ngưỡng cảnh báo ác tính**.

## Yêu cầu

- Windows và Python 3.10 trở lên.
- Các thư viện trong `requirements.txt`: numpy, pandas, scikit-learn, matplotlib, streamlit, plotly. Không cần Tkinter.

## Cài đặt và chạy

**Cách nhanh:** nhấp đúp `run_app.bat`. Lần đầu tệp này tự tạo `.venv`, cài thư viện, rồi chạy `streamlit run app.py`; các lần sau chạy thẳng. Trình duyệt tự mở `http://localhost:8501`.

**Cách thủ công** (PowerShell, tại thư mục dự án):

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Ghi chú:

- Lần đầu Streamlit có thể hỏi email, nhấn Enter để bỏ qua.
- Không đóng cửa sổ dòng lệnh khi đang dùng app. Nếu mở nhiều lần, app có thể chạy ở các cổng 8502, 8503...; nên đóng các cửa sổ cũ.
- `.venv` phụ thuộc bản Python dùng để tạo nó, không chép `.venv` giữa các máy.
- Tuỳ chọn: tạo `.streamlit/config.toml` để đổi màu giao diện.

### Dữ liệu huấn luyện

Đặt `archive.zip` (hoặc `data.csv`) cạnh `app.py`, hoặc tải lên ở ô "Dữ liệu huấn luyện" trên thanh bên. ZIP phải chứa CSV (ưu tiên tên `data.csv`).

## Quy trình mô hình

- Mã hoá B = 0, M = 1; M là lớp dương khi ghi TP/TN/FP/FN.
- Cột `id` chỉ dùng tra cứu, không đưa vào mô hình. `diagnosis` là nhãn. Cột hoàn toàn trống bị loại và được nêu trong báo cáo kiểm tra dữ liệu.
- Chia train/test 80/20 với `random_state=42`, `stratify=y`.
- Pipeline: `SimpleImputer(median)` → `StandardScaler` → `KNeighborsClassifier`. Pipeline chỉ fit trên train nên không rò rỉ thống kê từ test.
- GridSearchCV chạy trên train với `StratifiedKFold` 5 fold, thử K lẻ 1–31, Euclid/Manhattan và uniform/distance; tiêu chí là F1-macro.
- Biểu đồ chọn K dùng kiểm định chéo 5 fold trên tập train, không dùng tập test.

### Cách hiểu kết quả

Giao diện không hiển thị "xác suất mắc bệnh". KNN chỉ cho biết trong K ca đã có chẩn đoán giống ca đang xét nhất có bao nhiêu ca ác tính (tỉ lệ phiếu). Phần mềm đưa ra **cảnh báo** khi tỉ lệ này đạt ngưỡng cảnh báo (mặc định 50%) và gắn nhãn mức đồng thuận: hoàn toàn, cao, hoặc ranh giới. Hạ ngưỡng làm tăng Recall (ít bỏ sót ca ác tính) nhưng làm tăng số ca báo nhầm. Các chỉ số ở trang Đánh giá được tính theo ngưỡng đang chọn.

## Dữ liệu bệnh nhân và quyền riêng tư

Lịch sử ca khám được lưu ở `data/history.db` trên máy chạy app, chưa mã hoá và chưa có đăng nhập. Chỉ dùng dữ liệu mô phỏng hoặc dữ liệu đã ẩn danh. **Không đưa `data/` lên GitHub**: thêm dòng `data/` vào `.gitignore`.

## Kiểm thử

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_core.py
```

`test_core.py` kiểm tra phần dữ liệu và mô hình: dữ liệu thật có 569 mẫu, 30 đặc trưng, phân bố 357 B/212 M; loại đúng cột trống; không dùng `id`/`diagnosis` làm đặc trưng; phép chia tái lập; tổng ma trận nhầm lẫn và bảng dự đoán khớp số mẫu test. Tầng giao diện (`app.py`, `doctor.py`, `charts.py`) chưa có kiểm thử tự động.

## Cấu trúc mã nguồn

| Tệp | Vai trò |
|---|---|
| `app.py` | Giao diện Streamlit, điều hướng 5 trang |
| `doctor.py` | Dự đoán ca mới và hàng loạt, mức đồng thuận, lịch sử SQLite |
| `charts.py` | Chỉ số đánh giá và biểu đồ Plotly |
| `glossary.py` | Giải thích tiếng Việt cho chỉ số và thuật ngữ |
| `data_loader.py` | Đọc ZIP/CSV, kiểm tra và chuẩn bị dữ liệu |
| `model.py` | Chia dữ liệu, Pipeline, GridSearchCV, đánh giá |
| `plots.py` | Biểu đồ matplotlib (phân bố lớp, kết quả CV) |
| `test_core.py` | Kiểm thử dữ liệu và mô hình |
| `run_app.bat` | Tạo `.venv` lần đầu và mở app (giữ định dạng xuống dòng CRLF) |
| `main.py`, `ui.py` | Giao diện Tkinter bản đầu, không còn dùng trong app hiện tại |

## Phiên bản đã dùng

Python 3.13.7, streamlit 1.66.1, plotly 7.1.0, pandas 3.1.1, scikit-learn 1.9.1, numpy 2.5.4, matplotlib 3.11.2.
