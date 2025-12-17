# File: src/model.py
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np

class MoHinhPhanLoai:
    def __init__(self, loai_mo_hinh='knn', **kwargs):
        self.loai_mo_hinh = loai_mo_hinh
        if loai_mo_hinh == 'knn':
            # Sử dụng metric 'euclidean' để tính khoảng cách
            self.model = KNeighborsClassifier(metric='euclidean', **kwargs)
        elif loai_mo_hinh == 'svm':
            self.model = SVC(probability=True, **kwargs)
        else:
            raise ValueError("Loại mô hình không hợp lệ. Chọn 'knn' hoặc 'svm'.")
        self.is_trained = False

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, X_test, return_distance=False):
        """
        Dự đoán nhãn cho dữ liệu đầu vào.
        Nếu return_distance=True và mô hình là KNN, trả về thêm khoảng cách đến hàng xóm gần nhất.
        """
        if not self.is_trained:
            raise Exception("Mô hình chưa được huấn luyện.")
        
        if self.loai_mo_hinh == 'knn' and return_distance:
            # kneighbors trả về khoảng cách và chỉ số của k hàng xóm gần nhất
            # Ở đây ta chỉ quan tâm đến hàng xóm gần nhất (n_neighbors=1)
            dist, _ = self.model.kneighbors(X_test, n_neighbors=1)
            y_pred = self.model.predict(X_test)
            # Trả về nhãn dự đoán và khoảng cách (đã được làm phẳng)
            return y_pred, dist.flatten()
        
        return self.model.predict(X_test)

    # Các phương thức khác giữ nguyên
    def evaluate(self, X_test, y_test):
        if not self.is_trained:
            raise Exception("Mô hình chưa được huấn luyện.")
        accuracy = self.model.score(X_test, y_test)
        print(f"Độ chính xác của mô hình {self.loai_mo_hinh.upper()}: {accuracy:.2f}")
        return accuracy

    def luu_mo_hinh(self, đường_dẫn):
        import pickle
        with open(đường_dẫn, 'wb') as f:
            pickle.dump(self, f)
        print(f"Đã lưu mô hình phân loại tại '{đường_dẫn}'")

    @staticmethod
    def tai_mo_hinh(đường_dẫn):
        import pickle
        with open(đường_dẫn, 'rb') as f:
            return pickle.load(f)