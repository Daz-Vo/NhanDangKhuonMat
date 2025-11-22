# Dự án Nhận dạng Khuôn mặt sử dụng PCA (Eigenfaces)

Dự án này là một triển khai thuật toán nhận dạng khuôn mặt bằng phương pháp Phân tích Thành phần chính (Principal Component Analysis - PCA), còn được biết đến với tên gọi Eigenfaces.

## Cấu trúc thư mục

```
PCA_Face_Recognition/
│
├── data/                       # Chứa dữ liệu
│   ├── raw/                    # Dữ liệu thô (ảnh gốc, ví dụ: từ tập ORL)
│   └── processed/              # Dữ liệu đã xử lý (nếu có)
│
├── src/                        # Mã nguồn chính
│   ├── __init__.py
│   ├── data_loader.py          # Tải và tiền xử lý dữ liệu ảnh
│   ├── pca_engine.py           # Lớp thực hiện thuật toán PCA
│   ├── model.py                # Lớp chứa mô hình phân loại (k-NN, SVM)
│   └── visualization.py        # Các hàm để trực quan hóa kết quả
│
├── notebooks/                  # Chứa file mã nguồn để thử nghiệm
│   └── experiment.py           # File để chạy và trực quan hóa từng bước
│
├── models/                     # Lưu các mô hình đã huấn luyện
│
├── realtime_recognition.py     # File chạy nhận dạng khuôn mặt thời gian thực qua webcam
├── main.py                     # File chạy chính của chương trình
├── requirements.txt            # Các thư viện Python cần thiết
└── README.md                   # File hướng dẫn này
```

## Hướng dẫn cài đặt và sử dụng

### 1. Yêu cầu

- Python 3.7+
- `pip`

### 2. Cài đặt thư viện

Clone repository này về máy, sau đó cài đặt các thư viện cần thiết bằng lệnh sau:

```bash
pip install -r requirements.txt
```

### 3. Chuẩn bị dữ liệu

1.  Tải về một tập dữ liệu khuôn mặt. Dự án này được thiết kế để hoạt động tốt với **The ORL Database of Faces**. Bạn có thể tìm và tải về từ nhiều nguồn trên mạng.
2.  Giải nén và đặt các ảnh vào thư mục `data/raw/`.
3.  Cấu trúc thư mục dữ liệu phải theo dạng sau: mỗi người một thư mục con.
    ```
    data/raw/
    ├── s1/
    │   ├── 1.pgm
    │   ├── 2.pgm
    │   └── ...
    ├── s2/
    │   ├── 1.pgm
    │   └── ...
    └── ...
    ```

### 4. Chạy chương trình

Để chạy toàn bộ quy trình (tải dữ liệu, huấn luyện PCA, huấn luyện mô hình phân loại và đánh giá), thực thi file `main.py`:

```bash
python main.py
```

Chương trình sẽ thực hiện các bước sau:

- Tải và chuẩn hóa ảnh từ `data/raw/`.
- Chia dữ liệu thành tập huấn luyện và tập kiểm tra.
- Huấn luyện PCA trên tập huấn luyện và hiển thị các _eigenfaces_.
- Huấn luyện một mô hình phân loại (SVM hoặc k-NN) trên dữ liệu đã giảm chiều.
- Đánh giá độ chính xác của mô hình trên tập kiểm tra.
- Hiển thị một vài ví dụ về kết quả dự đoán.

### 5. Nhận dạng thời gian thực

Sau khi đã huấn luyện mô hình bằng cách chạy `main.py` (điều này sẽ tạo ra các file mô hình trong thư mục `models/`), bạn có thể sử dụng `realtime_recognition.py` để nhận dạng khuôn mặt qua webcam:

```bash
python realtime_recognition.py
```

-   Chương trình sẽ mở webcam của bạn.
-   Nó sẽ phát hiện khuôn mặt, trích xuất đặc trưng và dự đoán danh tính trong thời gian thực.
-   Nhấn phím `q` để thoát chương trình.

### 6. Thử nghiệm từng bước

Nếu bạn muốn hiểu rõ hơn về từng bước của thuật toán, bạn có thể chạy file `notebooks/experiment.py`. File này sẽ thực hiện và trực quan hóa các bước quan trọng như:

- Hiển thị ảnh mẫu.
- Tính toán và hiển thị khuôn mặt trung bình.
- Hiển thị các eigenfaces.
- Tái tạo lại khuôn mặt từ không gian đã giảm chiều.

```bash
python notebooks/experiment.py
```

### Tùy chỉnh

Bạn có thể thay đổi các tham số trong file `main.py` để thử nghiệm:

- `KICH_THUOC_ANH`: Kích thước ảnh sau khi chuẩn hóa.
- `SO_THANH_PHAN_PCA`: Số lượng thành phần chính (eigenfaces) giữ lại. Thay đổi giá trị này sẽ ảnh hưởng đến độ chính xác và tốc độ.
- `LOAI_MO_HINH`: Chọn giữa `'knn'` (mặc định) và `'svm'`.
