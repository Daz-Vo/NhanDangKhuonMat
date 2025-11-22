import os
import numpy as np
import pickle
from src.data_loader import tai_du_lieu_anh, chia_du_lieu
from src.pca_engine import DongCoPCA
from src.model import MoHinhPhanLoai
from src.visualization import ve_luoi_anh, ve_ket_qua_du_doan

# --- Cấu hình ---
DUONG_DAN_DU_LIEU_THO = 'data/raw'
KICH_THUOC_ANH = (112, 92) # Kích thước chuẩn của ảnh trong tập ORL
SO_THANH_PHAN_PCA = 150   # Số eigenfaces cần giữ lại (có thể điều chỉnh)
LOAI_MO_HINH = 'knn'      # Chọn 'knn' hoặc 'svm'
THAM_SO_KNN = {'n_neighbors': 5}
DUONG_DAN_LUU_PCA = 'models/pca_model.pkl'
DUONG_DAN_LUU_PHAN_LOAI = f'models/{LOAI_MO_HINH}_model.pkl'


def main():
    """
    Hàm chính để chạy toàn bộ quy trình nhận dạng khuôn mặt.
    """
    # --- 1. Tải và chuẩn bị dữ liệu ---
    print("Bắt đầu quá trình tải và xử lý dữ liệu...")
    if not os.path.exists(DUONG_DAN_DU_LIEU_THO):
        print(f"LỖI: Thư mục dữ liệu '{DUONG_DAN_DU_LIEU_THO}' không tồn tại.")
        print("Vui lòng tải tập dữ liệu (ví dụ: ORL) và giải nén vào đó.")
        print("Cấu trúc yêu cầu: data/raw/s1/1.pgm, data/raw/s2/1.pgm, ...")
        return

    du_lieu, nhan, so_lop = tai_du_lieu_anh(DUONG_DAN_DU_LIEU_THO, KICH_THUOC_ANH)
    
    if du_lieu.shape[0] == 0:
        print("LỖI: Không tìm thấy ảnh nào trong thư mục dữ liệu.")
        return
        
    print(f"Tải thành công {du_lieu.shape[0]} ảnh cho {so_lop} người.")

    # Tạo một dictionary để dễ tra cứu tên lớp (ví dụ: 's1', 's2',...)
    ten_cac_lop = {i: f's{i+1}' for i in range(so_lop)}

    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    X_train, X_test, y_train, y_test = chia_du_lieu(du_lieu, nhan)
    print(f"Đã chia dữ liệu: {len(X_train)} mẫu huấn luyện, {len(X_test)} mẫu kiểm tra.")

    # --- 2. Huấn luyện PCA ---
    print("\nBắt đầu huấn luyện PCA...")
    pca = DongCoPCA(so_thanh_phan=SO_THANH_PHAN_PCA)
    
    # Huấn luyện PCA trên tập train và biến đổi cả tập train và test
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    print(f"Đã giảm chiều dữ liệu từ {X_train.shape[1]} xuống {X_train_pca.shape[1]} chiều.")

    # Hiển thị các eigenfaces
    print("Hiển thị các eigenfaces...")
    ve_luoi_anh(
        pca.thanh_phan_chinh.T, 
        KICH_THUOC_ANH, 
        "Các Eigenfaces hàng đầu",
        so_hang=4,
        so_cot=8
    )

    # --- 3. Huấn luyện mô hình phân loại ---
    print(f"\nBắt đầu huấn luyện mô hình phân loại ({LOAI_MO_HINH.upper()})...")
    if LOAI_MO_HINH == 'knn':
        mo_hinh = MoHinhPhanLoai(loai_mo_hinh='knn', **THAM_SO_KNN)
    else:
        mo_hinh = MoHinhPhanLoai(loai_mo_hinh='svm')

    mo_hinh.train(X_train_pca, y_train)

    # --- 4. Đánh giá mô hình ---
    print("\nĐánh giá mô hình trên tập kiểm tra...")
    mo_hinh.evaluate(X_test_pca, y_test)

    # --- 5. Hiển thị kết quả dự đoán ---
    print("Hiển thị một vài kết quả dự đoán...")
    y_pred = mo_hinh.predict(X_test_pca)
    ve_ket_qua_du_doan(X_test, y_test, y_pred, ten_cac_lop, KICH_THUOC_ANH)

    # --- 6. Lưu mô hình (tùy chọn) ---
    # Ghi chú: Trong ứng dụng thực tế, bạn sẽ muốn lưu pca và mo_hinh
    # để có thể tái sử dụng mà không cần huấn luyện lại.
    mo_hinh.luu_mo_hinh(DUONG_DAN_LUU_PHAN_LOAI)

    with open(DUONG_DAN_LUU_PCA, 'wb') as f:
        pickle.dump(pca, f)
    # print(f"Đã lưu mô hình PCA và phân loại tại thư mục '{os.path.dirname(DUONG_DAN_LUU_PCA)}'")


if __name__ == '__main__':
    main()
