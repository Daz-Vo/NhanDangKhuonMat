import matplotlib.pyplot as plt
import numpy as np

def ve_luoi_anh(danh_sach_anh, kich_thuoc_anh, tieu_de_hinh, so_hang=4, so_cot=8):
    """
    Vẽ một lưới các ảnh.

    Args:
        danh_sach_anh (list or np.array): Danh sách các ảnh đã làm phẳng.
        kich_thuoc_anh (tuple): Kích thước gốc của ảnh (cao, rộng).
        tieu_de_hinh (str): Tiêu đề chung cho cả hình vẽ.
        so_hang (int): Số hàng trong lưới.
        so_cot (int): Số cột trong lưới.
    """
    fig, axes = plt.subplots(so_hang, so_cot, figsize=(so_cot * 1.5, so_hang * 1.5))
    fig.suptitle(tieu_de_hinh, fontsize=16)
    
    for i, ax in enumerate(axes.flat):
        if i < len(danh_sach_anh):
            # Reshape lại ảnh từ vector 1D và hiển thị
            ax.imshow(danh_sach_anh[i].reshape(kich_thuoc_anh), cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            # Ẩn các subplot không dùng đến
            ax.axis('off')
            
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def ve_ket_qua_du_doan(danh_sach_anh_test, nhan_thuc_te, nhan_du_doan, ten_cac_lop, kich_thuoc_anh, so_luong=12):
    """
    Hiển thị một vài ảnh từ tập kiểm tra cùng với nhãn dự đoán và nhãn thực tế.

    Args:
        danh_sach_anh_test (np.array): Mảng chứa dữ liệu ảnh test (đã làm phẳng).
        nhan_thuc_te (np.array): Mảng nhãn thực tế.
        nhan_du_doan (np.array): Mảng nhãn dự đoán.
        ten_cac_lop (dict): Dictionary ánh xạ từ nhãn số sang tên người (nếu có).
        kich_thuoc_anh (tuple): Kích thước gốc của ảnh (cao, rộng).
        so_luong (int): Số lượng ảnh cần hiển thị.
    """
    so_hang = 3
    so_cot = 4
    fig, axes = plt.subplots(so_hang, so_cot, figsize=(12, 9))
    fig.suptitle("Kết quả dự đoán trên tập kiểm tra", fontsize=16)

    # Lấy ngẫu nhiên một vài chỉ số để hiển thị
    indices = np.random.choice(len(danh_sach_anh_test), size=so_luong, replace=False)
    
    for i, ax in enumerate(axes.flat):
        if i < so_luong:
            idx = indices[i]
            ax.imshow(danh_sach_anh_test[idx].reshape(kich_thuoc_anh), cmap='gray')
            
            # Lấy tên từ nhãn
            ten_thuc_te = ten_cac_lop.get(nhan_thuc_te[idx], f"Lớp {nhan_thuc_te[idx]}")
            ten_du_doan = ten_cac_lop.get(nhan_du_doan[idx], f"Lớp {nhan_du_doan[idx]}")

            tieu_de = f"Thực tế: {ten_thuc_te}\nDự đoán: {ten_du_doan}"
            mau_sac = 'green' if nhan_thuc_te[idx] == nhan_du_doan[idx] else 'red'
            
            ax.set_title(tieu_de, color=mau_sac)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')
            
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == '__main__':
    # Ví dụ cách sử dụng
    KICH_THUOC_ANH_GIA = (20, 20)
    SO_ANH_GIA = 16
    
    # Tạo dữ liệu ảnh giả (nhiễu ngẫu nhiên)
    anh_gia = np.random.rand(SO_ANH_GIA, KICH_THUOC_ANH_GIA[0] * KICH_THUOC_ANH_GIA[1])
    
    # Vẽ lưới ảnh
    ve_luoi_anh(anh_gia, KICH_THUOC_ANH_GIA, "Ví dụ hiển thị lưới ảnh", so_hang=4, so_cot=4)

    # Tạo dữ liệu dự đoán giả
    nhan_thuc_te_gia = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
    nhan_du_doan_gia = np.array([0, 1, 3, 3, 0, 1, 2, 3, 1, 1, 2, 3])
    anh_test_gia = np.random.rand(12, KICH_THUOC_ANH_GIA[0] * KICH_THUOC_ANH_GIA[1])
    ten_lop_gia = {0: 'An', 1: 'Bình', 2: 'Cường', 3: 'Dũng'}

    ve_ket_qua_du_doan(anh_test_gia, nhan_thuc_te_gia, nhan_du_doan_gia, ten_lop_gia, KICH_THUOC_ANH_GIA)
