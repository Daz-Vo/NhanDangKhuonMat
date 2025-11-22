# Tệp này dùng để thử nghiệm từng bước của thuật toán,
# tương tự như cách làm việc trong một Jupyter Notebook.

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Thêm thư mục gốc của dự án vào sys.path để có thể import các module từ src
# Điều này cần thiết khi chạy file này trực tiếp từ thư mục notebooks
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.data_loader import tai_du_lieu_anh
from src.pca_engine import DongCoPCA
from src.visualization import ve_luoi_anh

# --- Cấu hình ---
DUONG_DAN_DU_LIEU_THO = os.path.join(project_root, 'data', 'raw')
KICH_THUOC_ANH = (112, 92)
SO_THANH_PHAN_PCA = 150

def buoc_1_tai_va_hien_thi_du_lieu():
    """
    Tải dữ liệu và hiển thị một vài ảnh mẫu.
    """
    print("--- Bước 1: Tải và hiển thị dữ liệu ---")
    if not os.path.exists(DUONG_DAN_DU_LIEU_THO) or not os.listdir(DUONG_DAN_DU_LIEU_THO):
        print(f"Thư mục dữ liệu '{DUONG_DAN_DU_LIEU_THO}' trống hoặc không tồn tại.")
        print("Vui lòng tải tập dữ liệu ORL và đặt vào đó.")
        return None, None
        
    du_lieu, nhan, _ = tai_du_lieu_anh(DUONG_DAN_DU_LIEU_THO, KICH_THUOC_ANH)
    print(f"Đã tải {du_lieu.shape[0]} ảnh.")
    
    # Hiển thị 32 ảnh đầu tiên
    ve_luoi_anh(du_lieu[:32], KICH_THUOC_ANH, "Một vài ảnh từ tập dữ liệu ORL")
    
    return du_lieu, nhan

def buoc_2_thuc_hien_pca(du_lieu):
    """
    Chạy thuật toán PCA và hiển thị khuôn mặt trung bình cùng các eigenfaces.
    """
    print("\n--- Bước 2: Thực hiện PCA ---")
    pca = DongCoPCA(so_thanh_phan=SO_THANH_PHAN_PCA)
    pca.fit(du_lieu)
    
    # Hiển thị khuôn mặt trung bình
    plt.figure(figsize=(4, 4))
    plt.imshow(pca.khuon_mat_trung_binh.reshape(KICH_THUOC_ANH), cmap='gray')
    plt.title("Khuôn mặt trung bình")
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
    # Hiển thị các eigenfaces
    ve_luoi_anh(pca.thanh_phan_chinh.T, KICH_THUOC_ANH, "Các Eigenfaces")
    
    return pca

def buoc_3_tai_tao_khuon_mat(du_lieu, pca):
    """
    Chiếu dữ liệu vào không gian PCA và tái tạo lại để xem sự khác biệt.
    """
    print("\n--- Bước 3: Tái tạo khuôn mặt từ không gian PCA ---")
    
    # Chọn một vài ảnh để thử
    anh_goc = du_lieu[:8]
    
    # Chiếu vào không gian PCA
    du_lieu_bien_doi = pca.transform(anh_goc)
    
    # Tái tạo lại từ không gian PCA
    anh_tai_tao = pca.chieu_nguoc(du_lieu_bien_doi)
    
    # Chuẩn bị danh sách để hiển thị xen kẽ
    danh_sach_hien_thi = np.zeros((16, anh_goc.shape[1]))
    danh_sach_hien_thi[0::2] = anh_goc
    danh_sach_hien_thi[1::2] = anh_tai_tao
    
    # Vẽ hình
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle("Ảnh gốc (trái) và Ảnh tái tạo từ PCA (phải)", fontsize=14)
    for i, ax in enumerate(axes.flat):
        ax.imshow(danh_sach_hien_thi[i].reshape(KICH_THUOC_ANH), cmap='gray')
        if i % 2 == 0:
            ax.set_ylabel(f"Gốc {i//2 + 1}", rotation=90, size='large')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == '__main__':
    # Chạy tuần tự các bước
    du_lieu, nhan = buoc_1_tai_va_hien_thi_du_lieu()
    
    if du_lieu is not None:
        pca_engine = buoc_2_thuc_hien_pca(du_lieu)
        buoc_3_tai_tao_khuon_mat(du_lieu, pca_engine)
