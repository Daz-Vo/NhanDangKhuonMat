# Hệ Thống Nhận Diện Khuôn Mặt (PCA & KNN)

Dự án nhận diện khuôn mặt Realtime sử dụng thuật toán **Eigenfaces (PCA)** để giảm chiều dữ liệu và **KNN (K-Nearest Neighbors)** để phân loại. Hệ thống hỗ trợ hiển thị thông tin chi tiết (MSSV, Lớp...) và tự động cảnh báo người lạ.

## 🚀 Tính Năng Nổi Bật

- **Nhận diện thời gian thực:** Tốc độ phản hồi nhanh qua Webcam.
- **Hiển thị thông tin cá nhân:** Tự động hiện Tên, MSSV, Lớp... từ file cấu hình.
- **Cảnh báo người lạ:** Tự động khoanh vùng **ĐỎ** và hiện "Unknown" nếu khuôn mặt không khớp với dữ liệu.
- **Giao diện thông minh:** Cửa sổ Camera tự động căn giữa màn hình và phóng to.
- **Dễ dàng mở rộng:** Chỉ cần thêm folder ảnh và chạy lại file train.

---

## 🛠 Yêu Cầu Cài Đặt

Đảm bảo bạn đã cài đặt Python (3.8 trở lên). Cài đặt các thư viện cần thiết bằng lệnh sau:

````bash
pip install numpy opencv-python scikit-learn

📂 Cấu Trúc Thư Mục
Project_Folder/
├── data/
│   └── raw/
│       ├── VoVanDat/           <-- Tên thư mục là Tên hiển thị
│       │   ├── info.json       <-- File chứa thông tin chi tiết
│       │   ├── anh1.jpg
│       │   ├── anh2.jpg
│       │   └── ...
│       ├── NguoiKhac/
│       │   ├── info.json
│       │   └── ...
├── models/                     <-- Nơi chứa các file model (.pkl) sau khi train
├── src/                        <-- Source code xử lý chính
│   ├── data_loader.py
│   ├── model.py
│   ├── pca_engine.py
│   └── ...
├── main.py                     <-- File dùng để Huấn Luyện (Training)
├── NhanDienKM.py               <-- File chạy Nhận Diện (Realtime)
└── README.md

📖 Hướng Dẫn Sử Dụng

### Bước 1: Chuẩn bị dữ liệu
Hệ thống yêu cầu mỗi người dùng phải có một thư mục riêng chứa ảnh và file thông tin.

1.  Vào thư mục `data/raw/`.
2.  Tạo thư mục mới với tên của bạn (Viết liền không dấu, ví dụ: `VoVanDat`).
3.  Copy khoảng **10-20 tấm ảnh** khuôn mặt của bạn vào thư mục đó.
4.  Tạo một file tên là `info.json` trong thư mục đó với nội dung như sau:
    ```json
    {
        "MSV": "2100xxxx",
        "Lop": "KTPM16A",
        "Khoa": "CNTT"
    }
    ```

### Bước 2: Huấn luyện mô hình
Mỗi khi thêm người mới hoặc sửa file `info.json`, bạn cần chạy lệnh này để hệ thống học dữ liệu:

```bash
python main.py

### Bước 3: Chạy nhận diện

```bash
python NhanDienKM.py

Thoát chương trình: Bấm phím q hoặc nhấn nút X (Close) trên thanh tiêu đề cửa sổ.



### Bước 4: Tinh chỉnh độ nhạy (Quan trọng)

Nếu hệ thống nhận diện sai (nhận người lạ thành bạn) hoặc không nhận ra bạn (báo Unknown/màu đỏ), hãy làm như sau:

Mở file NhanDienKM.py.
Tìm dòng: NGUONG_KHOANG_CACH = 2500.
Chạy chương trình và nhìn vào màn hình Console (Terminal) để xem "Khoảng cách đo được".
Sửa số 2500 thành giá trị phù hợp:
Tăng lên (ví dụ 3000): Nếu hệ thống quá khắt khe, không nhận ra bạn.
Giảm xuống (ví dụ 2000): Nếu hệ thống dễ tính, nhận nhầm người lạ.
````
