import cv2
import pickle
import numpy as np
import os

# --- Cấu hình ---
KICH_THUOC_ANH = (112, 92) # Phải giống với kích thước khi huấn luyện
DUONG_DAN_PCA = 'models/pca_model.pkl'
DUONG_DAN_PHAN_LOAI = 'models/knn_model.pkl'
DUONG_DAN_DU_LIEU_THO = 'data/raw'

# --- Tải các mô hình và dữ liệu cần thiết ---

# 1. Tải mô hình PCA đã huấn luyện
try:
    with open(DUONG_DAN_PCA, 'rb') as f:
        pca = pickle.load(f)
    print("Tải mô hình PCA thành công.")
except FileNotFoundError:
    print(f"LỖI: Không tìm thấy file mô hình PCA tại '{DUONG_DAN_PCA}'.")
    print("Vui lòng chạy 'main.py' để huấn luyện và lưu mô hình trước.")
    exit()

# 2. Tải mô hình phân loại đã huấn luyện
try:
    with open(DUONG_DAN_PHAN_LOAI, 'rb') as f:
        model = pickle.load(f)
    print("Tải mô hình phân loại thành công.")
except FileNotFoundError:
    print(f"LỖI: Không tìm thấy file mô hình phân loại tại '{DUONG_DAN_PHAN_LOAI}'.")
    exit()

# 3. Tạo dictionary để ánh xạ nhãn số sang tên thư mục (ví dụ: 0 -> 's1')
# Điều này giúp hiển thị tên người thay vì chỉ một con số
ten_nhan = {}
try:
    # Sắp xếp các thư mục để đảm bảo thứ tự nhất quán
    for i, ten_thu_muc in enumerate(sorted(os.listdir(DUONG_DAN_DU_LIEU_THO))):
        if os.path.isdir(os.path.join(DUONG_DAN_DU_LIEU_THO, ten_thu_muc)):
            ten_nhan[i] = ten_thu_muc
    print(f"Đã tạo ánh xạ cho {len(ten_nhan)} người.")
except FileNotFoundError:
    print(f"LỖI: Không tìm thấy thư mục dữ liệu thô tại '{DUONG_DAN_DU_LIEU_THO}'.")
    exit()

# 4. Tải bộ phát hiện khuôn mặt Haar Cascade của OpenCV
# Sử dụng đường dẫn được cung cấp sẵn bởi thư viện cv2
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    print("LỖI: Không thể tải file Haar Cascade để phát hiện khuôn mặt.")
    exit()
print("Tải bộ phát hiện khuôn mặt thành công.")


# --- Bắt đầu nhận dạng qua Webcam ---

# Mở webcam (thiết bị 0)
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("LỖI: Không thể mở webcam.")
    exit()

print("\nBắt đầu nhận dạng... Nhấn 'q' để thoát.")

while True:
    # Đọc từng khung hình từ webcam
    ret, frame = video_capture.read()
    if not ret:
        break

    # Chuyển khung hình sang ảnh xám để phát hiện khuôn mặt
    anh_xam = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Phát hiện các khuôn mặt trong ảnh xám
    cac_khuon_mat = face_cascade.detectMultiScale(
        anh_xam,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Vẽ hình chữ nhật xung quanh các khuôn mặt và nhận dạng
    for (x, y, w, h) in cac_khuon_mat:
        # Cắt vùng chứa khuôn mặt
        khuon_mat_roi = anh_xam[y:y+h, x:x+w]
        
        # Tiền xử lý khuôn mặt giống như khi huấn luyện
        khuon_mat_resized = cv2.resize(khuon_mat_roi, (KICH_THUOC_ANH[1], KICH_THUOC_ANH[0]))
        khuon_mat_flat = khuon_mat_resized.flatten().reshape(1, -1)
        
        # Giảm chiều bằng PCA
        khuon_mat_pca = pca.transform(khuon_mat_flat)
        
        # Dự đoán bằng mô hình phân loại
        nhan_du_doan_so = model.predict(khuon_mat_pca)
        xac_suat_du_doan = model.predict_proba(khuon_mat_pca)
        
        # Lấy tên tương ứng từ nhãn số
        ten_du_doan = ten_nhan.get(nhan_du_doan_so[0], "Khong ro")
        
        # Lấy xác suất cao nhất
        do_tin_cay = np.max(xac_suat_du_doan) * 100
        
        # Hiển thị kết quả
        ket_qua_text = f"{ten_du_doan} ({do_tin_cay:.2f}%)"
        
        # Vẽ hình chữ nhật quanh khuôn mặt
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Viết tên dự đoán lên trên
        cv2.putText(frame, ket_qua_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Hiển thị khung hình kết quả
    cv2.imshow('Nhan dang khuon mat - Nhan q de thoat', frame)

    # Chờ phím 'q' được nhấn để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng webcam và đóng tất cả cửa sổ
video_capture.release()
cv2.destroyAllWindows()
print("Chương trình đã kết thúc.")
