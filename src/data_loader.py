import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def tai_du_lieu_anh(duong_dan_thu_muc, kich_thuoc_anh=(100, 100)):
    """
    Tải tất cả ảnh từ một thư mục, chuyển đổi sang ảnh xám, resize và làm phẳng.

    Args:
        duong_dan_thu_muc (str): Đường dẫn đến thư mục chứa các thư mục con,
                                mỗi thư mục con là một người.
        kich_thuoc_anh (tuple): Kích thước mới của ảnh (chiều rộng, chiều cao).

    Returns:
        tuple: (danh sách ảnh đã làm phẳng, danh sách nhãn tương ứng, số lượng lớp)
    """
    du_lieu_anh = []
    nhan = []
    ten_nhan = {}
    nhan_hien_tai = 0

    # Duyệt qua các thư mục con (mỗi thư mục là một người)
    for ten_thu_muc in sorted(os.listdir(duong_dan_thu_muc)):
        duong_dan_nguoi = os.path.join(duong_dan_thu_muc, ten_thu_muc)
        if not os.path.isdir(duong_dan_nguoi):
            continue

        # Gán nhãn số cho mỗi người
        if ten_thu_muc not in ten_nhan:
            ten_nhan[ten_thu_muc] = nhan_hien_tai
            nhan_hien_tai += 1

        # Đọc từng ảnh trong thư mục của người đó
        for ten_file_anh in os.listdir(duong_dan_nguoi):
            duong_dan_anh = os.path.join(duong_dan_nguoi, ten_file_anh)
            
            # Đọc ảnh bằng OpenCV ở chế độ ảnh xám
            anh = cv2.imread(duong_dan_anh, cv2.IMREAD_GRAYSCALE)
            
            if anh is not None:
                # Resize ảnh về kích thước chuẩn
                anh_resized = cv2.resize(anh, (kich_thuoc_anh[1], kich_thuoc_anh[0]))
                
                # Làm phẳng ảnh thành vector 1D và thêm vào danh sách
                du_lieu_anh.append(anh_resized.flatten())
                nhan.append(ten_nhan[ten_thu_muc])

    so_luong_lop = len(ten_nhan)
    return np.array(du_lieu_anh), np.array(nhan), so_luong_lop

def chia_du_lieu(du_lieu, nhan, kich_thuoc_test=0.2, trang_thai_ngau_nhien=42):
    """
    Chia dữ liệu thành tập huấn luyện và tập kiểm tra.

    Args:
        du_lieu (np.array): Mảng chứa dữ liệu ảnh.
        nhan (np.array): Mảng chứa nhãn.
        kich_thuoc_test (float): Tỷ lệ của tập kiểm tra.
        trang_thai_ngau_nhien (int): Seed để đảm bảo kết quả tái lập được.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        du_lieu, nhan, test_size=kich_thuoc_test, random_state=trang_thai_ngau_nhien, stratify=nhan
    )
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    # Ví dụ cách sử dụng
    # Giả sử bạn có dữ liệu trong thư mục 'data/raw'
    # Cấu trúc:
    # data/raw/s1/1.pgm
    # data/raw/s1/2.pgm
    # ...
    # data/raw/s2/1.pgm
    # ...
    
    duong_dan_du_lieu = '../data/raw' # Thay đổi nếu cần
    
    if os.path.exists(duong_dan_du_lieu):
        du_lieu, nhan, so_lop = tai_du_lieu_anh(duong_dan_du_lieu)
        print(f"Đã tải thành công {du_lieu.shape[0]} ảnh.")
        print(f"Số chiều của mỗi ảnh (đã làm phẳng): {du_lieu.shape[1]}")
        print(f"Số lượng người (lớp): {so_lop}")
        print(f"Kích thước mảng nhãn: {nhan.shape}")

        X_train, X_test, y_train, y_test = chia_du_lieu(du_lieu, nhan)
        print("\nĐã chia dữ liệu:")
        print(f"Tập huấn luyện: {X_train.shape[0]} mẫu")
        print(f"Tập kiểm tra: {X_test.shape[0]} mẫu")
    else:
        print(f"Thư mục dữ liệu không tồn tại: {duong_dan_du_lieu}")
        print("Vui lòng tải tập dữ liệu ORL và giải nén vào 'data/raw'")
