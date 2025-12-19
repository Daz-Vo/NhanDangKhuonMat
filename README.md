# Hệ Thống Nhận Diện Khuôn Mặt (PCA & KNN)

> **Đồ án môn học: Xử lý ảnh / Trí tuệ nhân tạo**

Dự án xây dựng hệ thống điểm danh và nhận diện khuôn mặt thời gian thực (Real-time). Hệ thống sử dụng thuật toán **Principal Component Analysis (PCA - Eigenfaces)** để trích xuất đặc trưng và **K-Nearest Neighbors (KNN)** để phân loại.

## 🚀 Tính Năng Chính

1. **Thu thập dữ liệu tự động:** Tool hỗ trợ chụp ảnh mẫu và nhập thông tin cá nhân (MSSV, Lớp) trực tiếp từ màn hình console.
2. **Huấn luyện mô hình (Training):** Tự động quét toàn bộ thư mục dữ liệu, trích xuất Eigenfaces và huấn luyện bộ phân loại KNN.
3. **Nhận diện Real-time:**
   - Tự động căn giữa cửa sổ camera trên màn hình.
   - Hiển thị tên và thông tin chi tiết (MSSV, Lớp...) nếu nhận diện đúng.
   - **Cảnh báo người lạ:** Tự động khoanh vùng **MÀU ĐỎ** và hiện "Unknown" nếu khuôn mặt không khớp với dữ liệu đã học.

---

## 🛠 Yêu Cầu Cài Đặt

Môi trường khuyến nghị: Python 3.8 trở lên.
Cài đặt các thư viện cần thiết bằng lệnh:

```bash
python -m pip install numpy opencv-python scikit-learn

📂 Cấu Trúc Thư Mục
Để hệ thống hoạt động, cấu trúc thư mục phải được sắp xếp như sau:

Plaintext

Project_PCA/
├── data/
│   └── raw/                <-- Nơi chứa ảnh khuôn mặt (được tạo tự động)
│       ├── VoVanDat/
│       │   ├── info.json   <-- File chứa thông tin: MSV, Lớp...
│       │   ├── 0.jpg
│       │   ├── 1.jpg
│       │   └── ...
├── models/                 <-- Nơi chứa file model sau khi train (.pkl)
├── src/                    <-- Mã nguồn xử lý lõi
│   ├── data_loader.py      <-- Đọc và tiền xử lý ảnh
│   ├── pca_engine.py       <-- Class xử lý thuật toán PCA
│   ├── model.py            <-- Class xử lý thuật toán KNN
│   └── ...
├── ThuThapDuLieu.py        <-- [BƯỚC 1] Chạy file này để thêm người mới
├── main.py                 <-- [BƯỚC 2] Chạy file này để huấn luyện
├── NhanDienKM.py           <-- [BƯỚC 3] Chạy file này để nhận diện
└── README.md
📖 Hướng Dẫn Sử Dụng
Bước 1: Thu thập dữ liệu
Thay vì copy ảnh thủ công, hãy dùng tool tự động:

Bash

python ThuThapDuLieu.py
Nhập Tên (Viết liền không dấu, vd: NguyenVanA).

Nhập MSSV, Lớp khi được hỏi.

Nhấn Enter để bật Camera.

Ngồi trước camera, thay đổi nhẹ góc mặt để hệ thống chụp đủ 30 tấm ảnh.

Bước 2: Huấn luyện mô hình (Training)
Sau khi có dữ liệu người mới, cần chạy lệnh này để máy học lại:

Bash

python main.py
Hệ thống sẽ tạo ra các file pca_model.pkl, knn_model.pkl và info_map.pkl trong thư mục models/.

Bước 3: Chạy nhận diện
Khởi động camera để kiểm tra kết quả:

Bash

python NhanDienKM.py
Thoát chương trình: Bấm phím q hoặc nút X (Close) trên cửa sổ.

⚙️ Tinh Chỉnh Độ Chính Xác (Quan Trọng)
Do thuật toán PCA rất nhạy cảm với ánh sáng và thay đổi góc mặt, kết quả tính toán khoảng cách (Distance) có thể biến động lớn.

Nếu hệ thống nhận nhầm người lạ hoặc không nhận ra bạn (báo Unknown), hãy làm theo các bước sau:

Mở file NhanDienKM.py.

Tìm dòng cấu hình:

Python

NGUONG_KHOANG_CACH = 2500
Quan sát Terminal/Console khi chạy chương trình để xem thông số Khoảng cách đo được.

Điều chỉnh:

Nếu Console báo khoảng cách toàn 3000-4000 mà vẫn là bạn -> Tăng số này lên (ví dụ: 4500).

Nếu người lạ vào mà khoảng cách chỉ 1000-2000 -> Giảm số này xuống.

🧠 Nguyên Lý Hoạt Động
1. Tiền xử lý
Ảnh đầu vào được chuyển sang ảnh xám (Grayscale).

Resize đồng bộ về kích thước chuẩn (92x112) để đảm bảo tính nhất quán cho ma trận.

2. Trích xuất đặc trưng (PCA)
Sử dụng thuật toán PCA (Principal Component Analysis) để giảm chiều dữ liệu.

Thay vì xử lý hàng nghìn pixel, mỗi khuôn mặt được nén thành một vector đặc trưng (Eigenface) gồm khoảng 150 thành phần chính.

3. Phân loại (KNN)
Sử dụng thuật toán K-Nearest Neighbors (KNN).

Hệ thống tính Khoảng cách Euclidean giữa vector khuôn mặt hiện tại và các vector đã học.

Nếu Khoảng cách < Ngưỡng: Trả về tên người dùng và hiển thị thông tin.

Nếu Khoảng cách > Ngưỡng: Kết luận là người lạ ("Unknown").
```
