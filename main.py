import os
import numpy as np
import pickle
import json  # <--- Nhớ thêm thư viện này
from src.data_loader import tai_du_lieu_anh, chia_du_lieu
from src.pca_engine import DongCoPCA
from src.model import MoHinhPhanLoai

# --- Cấu hình ---
DUONG_DAN_DU_LIEU_THO = 'data/raw'
KICH_THUOC_ANH = (112, 92) 
SO_THANH_PHAN_PCA = 150   
LOAI_MO_HINH = 'knn'      
THAM_SO_KNN = {'n_neighbors': 5}

# Các đường dẫn lưu model
DUONG_DAN_LUU_PCA = 'models/pca_model.pkl'
DUONG_DAN_LUU_PHAN_LOAI = f'models/{LOAI_MO_HINH}_model.pkl'
DUONG_DAN_LUU_NHAN = 'models/label_map.pkl'
DUONG_DAN_LUU_INFO = 'models/info_map.pkl' # <--- File mới chứa thông tin chi tiết

def main():
    print(">>> Bắt đầu quá trình huấn luyện và tổng hợp thông tin...")
    
    if not os.path.exists(DUONG_DAN_DU_LIEU_THO):
        print(f"LỖI: Không tìm thấy thư mục '{DUONG_DAN_DU_LIEU_THO}'")
        return

    # 1. Tải dữ liệu ảnh
    du_lieu, nhan, so_lop = tai_du_lieu_anh(DUONG_DAN_DU_LIEU_THO, KICH_THUOC_ANH)
    if du_lieu.shape[0] == 0: return

    # 2. Xử lý tên lớp và ĐỌC FILE INFO.JSON CỦA TỪNG NGƯỜI
    danh_sach_thu_muc = sorted([d for d in os.listdir(DUONG_DAN_DU_LIEU_THO) 
                                if os.path.isdir(os.path.join(DUONG_DAN_DU_LIEU_THO, d))])
    
    ten_cac_lop = {i: ten for i, ten in enumerate(danh_sach_thu_muc)}
    
    # --- ĐOẠN MỚI: QUÉT FILE JSON ---
    info_map = {} # Dictionary chứa thông tin của tất cả mọi người
    print("-> Đang quét file info.json trong từng thư mục...")
    
    for ten_folder in danh_sach_thu_muc:
        duong_dan_json = os.path.join(DUONG_DAN_DU_LIEU_THO, ten_folder, 'info.json')
        if os.path.exists(duong_dan_json):
            try:
                with open(duong_dan_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    info_map[ten_folder] = data # Gán thông tin vào tên folder
                    print(f"   + Đã đọc thông tin của: {ten_folder}")
            except Exception as e:
                print(f"   ! Lỗi đọc file json của {ten_folder}: {e}")
        else:
            print(f"   - Không tìm thấy info.json cho: {ten_folder}")

    # 3. Train Model (Giữ nguyên)
    X_train, X_test, y_train, y_test = chia_du_lieu(du_lieu, nhan)
    
    print("-> Đang huấn luyện PCA...")
    pca = DongCoPCA(so_thanh_phan=SO_THANH_PHAN_PCA)
    X_train_pca = pca.fit_transform(X_train)

    print(f"-> Đang huấn luyện mô hình {LOAI_MO_HINH.upper()}...")
    if LOAI_MO_HINH == 'knn':
        mo_hinh = MoHinhPhanLoai(loai_mo_hinh='knn', **THAM_SO_KNN)
    else:
        mo_hinh = MoHinhPhanLoai(loai_mo_hinh='svm')
    mo_hinh.train(X_train_pca, y_train)

    # 4. Lưu tất cả
    print("-> Đang lưu các mô hình...")
    mo_hinh.luu_mo_hinh(DUONG_DAN_LUU_PHAN_LOAI)
    with open(DUONG_DAN_LUU_PCA, 'wb') as f: pickle.dump(pca, f)
    with open(DUONG_DAN_LUU_NHAN, 'wb') as f: pickle.dump(ten_cac_lop, f)
        
    # Lưu file info tổng hợp
    with open(DUONG_DAN_LUU_INFO, 'wb') as f:
        pickle.dump(info_map, f)
    print(f"-> Đã lưu file thông tin tổng hợp tại: {DUONG_DAN_LUU_INFO}")

    print("\n>>> HOÀN TẤT!")

if __name__ == '__main__':
    main()