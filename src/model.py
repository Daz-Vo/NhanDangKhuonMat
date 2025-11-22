from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle

class MoHinhPhanLoai:
    def __init__(self, loai_mo_hinh='knn', **kwargs):
        """
        Khởi tạo lớp phân loại.

        Args:
            loai_mo_hinh (str): 'knn' cho K-Nearest Neighbors hoặc 'svm' cho Support Vector Machine.
            **kwargs: Các tham số cho mô hình (ví dụ: n_neighbors=3 cho KNN).
        """
        if loai_mo_hinh == 'knn':
            self.model = KNeighborsClassifier(**kwargs)
        elif loai_mo_hinh == 'svm':
            # Cấu hình SVM mặc định tốt cho nhận dạng khuôn mặt
            svm_params = {'C': 1000.0, 'gamma': 0.001, 'kernel': 'rbf', 'class_weight': 'balanced'}
            svm_params.update(kwargs) # Cho phép ghi đè tham số mặc định
            self.model = SVC(**svm_params)
        else:
            raise ValueError("Loại mô hình không được hỗ trợ. Vui lòng chọn 'knn' hoặc 'svm'.")
        
        self.loai_mo_hinh = loai_mo_hinh

    def train(self, X_train, y_train):
        """
        Huấn luyện mô hình trên dữ liệu huấn luyện.

        Args:
            X_train (np.array): Dữ liệu đặc trưng của tập huấn luyện.
            y_train (np.array): Nhãn của tập huấn luyện.
        """
        print(f"Bắt đầu huấn luyện mô hình {self.loai_mo_hinh.upper()}...")
        self.model.fit(X_train, y_train)
        print("Huấn luyện hoàn tất.")

    def predict(self, X_test):
        """
        Dự đoán nhãn cho dữ liệu mới.

        Args:
            X_test (np.array): Dữ liệu đặc trưng của tập kiểm tra.

        Returns:
            np.array: Mảng chứa các nhãn được dự đoán.
        """
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        """
        Đánh giá độ chính xác của mô hình.

        Args:
            X_test (np.array): Dữ liệu đặc trưng của tập kiểm tra.
            y_test (np.array): Nhãn thực tế của tập kiểm tra.
        """
        y_pred = self.predict(X_test)
        do_chinh_xac = accuracy_score(y_test, y_pred)
        
        print(f"\n--- Báo cáo đánh giá cho mô hình {self.loai_mo_hinh.upper()} ---")
        print(f"Độ chính xác: {do_chinh_xac:.4f}")
        print("\nBáo cáo phân loại chi tiết:")
        print(classification_report(y_test, y_pred))
        
        return do_chinh_xac

    def luu_mo_hinh(self, duong_dan_file):
        """
        Lưu mô hình đã huấn luyện vào file.

        Args:
            duong_dan_file (str): Đường dẫn để lưu file model (ví dụ: 'models/knn_model.pkl').
        """
        with open(duong_dan_file, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Mô hình đã được lưu tại: {duong_dan_file}")

    def tai_mo_hinh(self, duong_dan_file):
        """
        Tải mô hình từ file.

        Args:
            duong_dan_file (str): Đường dẫn đến file model.
        """
        with open(duong_dan_file, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Mô hình đã được tải từ: {duong_dan_file}")

if __name__ == '__main__':
    # Ví dụ cách sử dụng
    import numpy as np
    
    # Tạo dữ liệu giả
    X_train_gia = np.random.rand(80, 50) # 80 mẫu, 50 chiều
    y_train_gia = np.random.randint(0, 10, 80) # 10 lớp
    X_test_gia = np.random.rand(20, 50)
    y_test_gia = np.random.randint(0, 10, 20)

    # Sử dụng KNN
    knn = MoHinhPhanLoai(loai_mo_hinh='knn', n_neighbors=5)
    knn.train(X_train_gia, y_train_gia)
    knn.evaluate(X_test_gia, y_test_gia)
    knn.luu_mo_hinh('../models/knn_test.pkl')
    
    # Sử dụng SVM
    svm = MoHinhPhanLoai(loai_mo_hinh='svm')
    svm.train(X_train_gia, y_train_gia)
    svm.evaluate(X_test_gia, y_test_gia)
    svm.luu_mo_hinh('../models/svm_test.pkl')
