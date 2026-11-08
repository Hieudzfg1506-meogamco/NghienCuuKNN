# Ứng dụng phân loại KNN – Breast Cancer Wisconsin

Ứng dụng desktop nhỏ bằng Python/Tkinter, dùng dữ liệu thật trong `archive.zip` (tệp `data.csv`). Chương trình đọc và kiểm tra dữ liệu, huấn luyện K-Nearest Neighbors, tìm tham số bằng kiểm định chéo, đánh giá tập test, phân tích các mẫu dự đoán sai và xuất báo cáo.

> Kết quả của chương trình chỉ phục vụ học tập về học máy, không phải chẩn đoán y khoa.

## Yêu cầu

- Windows và Python 3.10 trở lên (bản Python có Tkinter).
- Các thư viện trong `requirements.txt`.

## Cài đặt và chạy

Môi trường `.venv` đã được tạo tại thư mục dự án. **Nhấp đúp `run_app.bat` để mở ứng dụng**. Tệp này tự chuyển đến thư mục chương trình, dùng Python trong `.venv` và giữ cửa sổ thông báo nếu ứng dụng gặp lỗi. Không cần kích hoạt môi trường hoặc thay đổi PATH.

Để chạy từ PowerShell tại thư mục dự án:

```powershell
.\.venv\Scripts\python.exe -E main.py
```

Khi tạo lại môi trường (hoặc chép dự án sang máy khác), dùng một bản Python có Tkinter đã được kiểm tra. Trên máy hiện tại, Python 3.13.7 được launcher `py` nhận diện:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import tkinter, matplotlib, pandas, sklearn; print('Import thành công')"
.\.venv\Scripts\python.exe -E main.py
```

Luôn dùng `.venv\Scripts\python.exe -m pip` để cài thư viện và cùng Python đó để chạy chương trình. Tùy chọn `-E` khi khởi động bỏ qua biến môi trường `PYTHONHOME`/`PYTHONPATH` của tiến trình; không sửa các biến của Windows. `.venv` phụ thuộc bản Python dùng để tạo nó, nên không chép `.venv` giữa các máy.

### Kết quả kiểm tra môi trường ngày 09/10/2026

- `python` trỏ tới bản cài Python 3.13.7 tại `%LOCALAPPDATA%\Programs\Python\Python313\python.exe`.
- `pip` trên PATH thuộc bản Python 3.13; `py` là launcher trong `%LOCALAPPDATA%\Programs\Python\Launcher\py.exe` và nhận diện được Python 3.13 khi chạy ngoài sandbox.
- `PYTHONHOME` và `PYTHONPATH` không được đặt ở phạm vi tiến trình, người dùng hoặc hệ thống.
- Cảnh báo `Failed to find real location` và việc `py` không tìm thấy Python chỉ xuất hiện trong sandbox của phiên kiểm tra. Ngoài sandbox, bản cài Python hoạt động và tạo `.venv` kèm `pip` thành công; không cần Repair/cài lại hay sửa PATH hệ thống.
- Lỗi thiếu `matplotlib` xảy ra vì thư viện chưa được cài vào môi trường dùng để chạy ứng dụng. Các thư viện được cài bằng chính Python của `.venv`.
- Đã kiểm tra thành công: Tkinter 8.6, Matplotlib 3.11.2, pandas 2.3.3 và scikit-learn 1.9.1. `pip check` không phát hiện phụ thuộc bị lỗi; cả 3 kiểm thử trong `test_core.py` đều đạt.
- Đã chạy `main.py` bằng `.venv`, xác nhận tải 569 mẫu/30 đặc trưng, tạo đủ 3 tab và 4 vùng biểu đồ. Đã mở cửa sổ ứng dụng qua `run_app.bat` và xác nhận tiến trình tiếp tục chạy. Launcher đã được kiểm tra trả mã lỗi 1 và dừng bằng `pause` khi thiếu `.venv`.
- `run_app.bat` dùng UTF-8 và xuống dòng CRLF của Windows. Nếu chỉnh sửa tệp này, giữ định dạng xuống dòng CRLF để `cmd.exe` đọc đúng.

Khi khởi động, ứng dụng tự đọc `archive.zip` cạnh `main.py`. Nút **Mở dữ liệu** cho phép chọn một ZIP hoặc CSV khác. ZIP phải chứa CSV (ưu tiên tên `data.csv`).

## Cách sử dụng

1. Ở tab **Dữ liệu**, xem báo cáo kiểm tra, bảng dữ liệu và thống kê 30 đặc trưng.
2. Ở tab **Huấn luyện và đánh giá**, chọn K, khoảng cách và cách bỏ phiếu rồi nhấn **Chạy KNN**; hoặc nhấn **Tìm tham số phù hợp** để chạy GridSearchCV.
3. Ở tab **Biểu đồ và dự đoán sai**, xem phân bố lớp, ma trận nhầm lẫn, kết quả CV, biểu đồ phân tán và chi tiết các láng giềng của mẫu sai.
4. Nhấn **Xuất báo cáo**, chọn thư mục đích để lưu CSV, TXT và PNG.

## Quy trình mô hình

- Mã hóa `B = 0`, `M = 1`; `M` là lớp dương khi ghi TP/TN/FP/FN.
- Cột `id` chỉ dùng tra cứu, không đưa vào mô hình. `diagnosis` là nhãn. Cột hoàn toàn trống được loại và được nêu trong báo cáo.
- Chia train/test 80/20 với `random_state=42`, `stratify=y`.
- Pipeline: `SimpleImputer(median)` → `StandardScaler` → `KNeighborsClassifier`. Pipeline chỉ fit trên train nên không rò rỉ thống kê từ test.
- GridSearchCV chạy trên train với StratifiedKFold 5 fold, thử K lẻ 1–31, Euclid/Manhattan và uniform/distance; tiêu chí là F1-macro.

## Kiểm thử

Sau khi cài thư viện:

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_core.py
```

Kiểm thử xác nhận dữ liệu thật có 569 mẫu, 30 đặc trưng, phân bố 357 B/212 M; loại đúng cột trống và không dùng `id`/`diagnosis` làm đặc trưng; phép chia tái lập; tổng ma trận nhầm lẫn và bảng dự đoán khớp số mẫu test.

## Cấu trúc mã nguồn

- `data_loader.py`: đọc ZIP/CSV, kiểm tra và chuẩn bị dữ liệu.
- `model.py`: chia dữ liệu, Pipeline, GridSearchCV, đánh giá và láng giềng.
- `plots.py`: tạo biểu đồ dùng cho giao diện và xuất PNG.
- `ui.py`: giao diện Tkinter ba tab và tác vụ nền.
- `main.py`: điểm khởi động.
- `test_core.py`: kiểm thử phần dữ liệu và mô hình.
- `run_app.bat`: mở ứng dụng bằng môi trường `.venv`, giữ cửa sổ lỗi khi khởi động thất bại.
