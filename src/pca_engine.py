import numpy as np

class DongCoPCA:
    def __init__(self, so_thanh_phan=None):
        """
        Khởi tạo lớp PCA.

        Args:
            so_thanh_phan (int): Số lượng thành phần chính (eigenfaces) cần giữ lại.
                               Nếu None, sẽ giữ lại tất cả.
        """
        self.so_thanh_phan = so_thanh_phan
        self.khuon_mat_trung_binh = None
        self.thanh_phan_chinh = None # Eigenfaces

    def fit(self, X):
        """
        Huấn luyện mô hình PCA trên dữ liệu X.

        Args:
            X (np.array): Dữ liệu huấn luyện, mỗi hàng là một ảnh đã làm phẳng.
                          Kích thước (so_mau, so_chieu).
        """
        # 1. Tính toán khuôn mặt trung bình
        self.khuon_mat_trung_binh = np.mean(X, axis=0)

        # 2. Trừ khuôn mặt trung bình khỏi dữ liệu
        X_centered = X - self.khuon_mat_trung_binh

        # 3. Tính ma trận hiệp phương sai
        # Để hiệu quả, nếu số mẫu < số chiều, ta dùng "snapshot method"
        # Cov = (1/N) * A.T @ A, với A là dữ liệu đã trừ trung bình
        # Thay vì tính ma trận (số_chiều x số_chiều), ta tính (số_mẫu x số_mẫu)
        # rồi suy ra eigenvector của ma trận gốc.
        L = X_centered @ X_centered.T
        
        # 4. Tính giá trị riêng (eigenvalues) và vector riêng (eigenvectors) của L
        gia_tri_rieng, vector_rieng_L = np.linalg.eigh(L)

        # 5. Suy ra vector riêng của ma trận hiệp phương sai gốc
        vector_rieng_C = X_centered.T @ vector_rieng_L

        # 6. Sắp xếp các vector riêng theo thứ tự giảm dần của giá trị riêng
        idx_sorted = np.argsort(gia_tri_rieng)[::-1]
        vector_rieng_sorted = vector_rieng_C[:, idx_sorted]
        
        # 7. Chuẩn hóa các vector riêng (eigenfaces) để chúng có độ dài đơn vị
        for i in range(vector_rieng_sorted.shape[1]):
            vector_rieng_sorted[:, i] = vector_rieng_sorted[:, i] / np.linalg.norm(vector_rieng_sorted[:, i])

        # 8. Giữ lại số lượng thành phần chính mong muốn
        if self.so_thanh_phan is not None:
            self.thanh_phan_chinh = vector_rieng_sorted[:, :self.so_thanh_phan]
        else:
            self.thanh_phan_chinh = vector_rieng_sorted

    def transform(self, X):
        """
        Chiếu dữ liệu X vào không gian eigenface.

        Args:
            X (np.array): Dữ liệu cần chiếu, mỗi hàng là một ảnh.

        Returns:
            np.array: Dữ liệu đã được chiếu (đã giảm chiều).
        """
        if self.khuon_mat_trung_binh is None or self.thanh_phan_chinh is None:
            raise RuntimeError("Mô hình PCA chưa được huấn luyện. Vui lòng gọi hàm fit() trước.")
        
        # Trừ khuôn mặt trung bình
        X_centered = X - self.khuon_mat_trung_binh
        
        # Chiếu vào không gian con được định nghĩa bởi các thành phần chính
        return X_centered @ self.thanh_phan_chinh

    def fit_transform(self, X):
        """
        Kết hợp cả hai bước fit và transform.
        """
        self.fit(X)
        return self.transform(X)

    def chieu_nguoc(self, X_transformed):
        """
        Tái tạo lại ảnh từ dữ liệu đã giảm chiều.

        Args:
            X_transformed (np.array): Dữ liệu trong không gian eigenface.

        Returns:
            np.array: Dữ liệu ảnh đã được tái tạo.
        """
        if self.khuon_mat_trung_binh is None or self.thanh_phan_chinh is None:
            raise RuntimeError("Mô hình PCA chưa được huấn luyện.")
            
        # Tái tạo lại từ không gian eigenface và cộng lại khuôn mặt trung bình
        return X_transformed @ self.thanh_phan_chinh.T + self.khuon_mat_trung_binh

if __name__ == '__main__':
    # Ví dụ cách sử dụng
    # Tạo dữ liệu giả
    X_gia = np.random.rand(10, 100) # 10 ảnh, mỗi ảnh 100 pixels

    # Khởi tạo và huấn luyện PCA
    pca = DongCoPCA(so_thanh_phan=5)
    X_bien_doi = pca.fit_transform(X_gia)

    print("Kích thước dữ liệu gốc:", X_gia.shape)
    print("Kích thước dữ liệu sau khi giảm chiều:", X_bien_doi.shape)

    # Tái tạo lại ảnh
    X_tai_tao = pca.chieu_nguoc(X_bien_doi)
    print("Kích thước dữ liệu tái tạo:", X_tai_tao.shape)

    # Kiểm tra sai số tái tạo
    sai_so = np.mean((X_gia - X_tai_tao)**2)
    print(f"Sai số tái tạo trung bình: {sai_so:.4f}")
