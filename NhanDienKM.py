import cv2
import pickle
import numpy as np
import os
import ctypes
import json
from PIL import Image, ImageDraw, ImageFont # Thư viện xử lý ảnh để viết tiếng Việt

from src.pca_engine import DongCoPCA
from src.model import MoHinhPhanLoai

# --- CẤU HÌNH ---
SIZE_OPENCV = (92, 112)
PATH_PCA = 'models/pca_model.pkl'
PATH_MODEL_KNN = 'models/knn_model.pkl'
PATH_LABEL = 'models/label_map.pkl'
PATH_INFO_MAP = 'models/info_map.pkl'
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
WINDOW_NAME = "Nhan Dien Khuon Mat"

# Ngưỡng khoảng cách
NGUONG_KHOANG_CACH = 4900

def load_artifacts():
    print(">>> Đang tải models và data...")
    try:
        with open(PATH_LABEL, 'rb') as f: label_map = pickle.load(f)
        info_map = {}
        if os.path.exists(PATH_INFO_MAP):
            with open(PATH_INFO_MAP, 'rb') as f: info_map = pickle.load(f)
        with open(PATH_PCA, 'rb') as f: pca = pickle.load(f)
        model_path = PATH_MODEL_KNN if os.path.exists(PATH_MODEL_KNN) else 'models/svm_model.pkl'
        with open(model_path, 'rb') as f: classifier = pickle.load(f)
        print("-> Tải thành công!")
        return pca, classifier, label_map, info_map
    except Exception as e:
        print(f"LỖI khi tải file: {e}")
        return None, None, None, None

def center_window(cap):
    try:
        user32 = ctypes.windll.user32
        screen_w, screen_h = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        cam_w, cam_h = int(cap.get(3)), int(cap.get(4))
        scale = 1.5; new_w, new_h = int(cam_w * scale), int(cam_h * scale)
        x, y = (screen_w - new_w) // 2, (screen_h - new_h) // 2
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, new_w, new_h); cv2.moveWindow(WINDOW_NAME, x, y)
    except: cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

# --- HÀM MỚI: VIẾT TIẾNG VIỆT ---
def vietnamese_text(img, text, position, text_color, font_size=20, outline_color=(0,0,0)):
    """
    Hàm hỗ trợ viết tiếng Việt lên ảnh OpenCV
    """
    # 1. Chuyển từ ảnh OpenCV (numpy array) sang PIL Image
    cv2_im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im_rgb)
    draw = ImageDraw.Draw(pil_im)

    # 2. Load Font (Dùng Arial có sẵn trong Windows)
    try:
        # Đường dẫn font trên Windows
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        # Nếu không tìm thấy arial, dùng font mặc định (sẽ không đẹp bằng)
        font = ImageFont.load_default()

    # 3. Vẽ viền đen (bằng cách vẽ chữ màu đen lệch đi 1 chút nhiều lần)
    x, y = position
    outline_range = 2 # Độ dày viền
    if outline_color:
        for dx in range(-outline_range, outline_range+1):
            for dy in range(-outline_range, outline_range+1):
                if dx != 0 or dy != 0:
                    draw.text((x+dx, y+dy), text, font=font, fill=outline_color)

    # 4. Vẽ chữ chính
    draw.text(position, text, font=font, fill=text_color)

    # 5. Chuyển lại về ảnh OpenCV
    return cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

# ... (Các phần import và hàm load_artifacts, vietnamese_text giữ nguyên) ...

def main():
    pca, classifier, label_map, info_map = load_artifacts()
    if pca is None: return
    
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): print("LỖI: Không thể mở Camera."); return
    
    center_window(cap)
    
    print(">>> Camera sẵn sàng. Đang chạy chế độ ĐA ĐỐI TƯỢNG.")
    print(f">>> Ngưỡng khoảng cách: {NGUONG_KHOANG_CACH}")

    while True:
        ret, frame = cap.read()
        if not ret: break
        try:
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1: break
        except: pass
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # --- CẢI TIẾN 1: Tăng minSize để bỏ qua người ở xa ---
        # minSize=(30, 30) -> Quá nhỏ, dễ nhiễu
        # minSize=(80, 80) -> Chỉ nhận diện người đứng gần, rõ nét
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=6,     # Tăng lên 6 để lọc bớt nhiễu (khắt khe hơn)
            minSize=(80, 80)    # Chỉ nhận diện khuôn mặt to (gần camera)
        )

        # Duyệt qua từng khuôn mặt trong đám đông
        for (x, y, w, h) in faces:
            box_color = (0, 255, 0) 
            name = "Unknown"
            
            try:
                # Cắt vùng mặt
                roi_gray = gray[y:y+h, x:x+w]
                
                # --- CẢI TIẾN 2: Cân bằng sáng cho vùng mặt này ---
                # Giúp PCA nhìn rõ mặt dù ánh sáng không đều
                roi_gray = cv2.equalizeHist(roi_gray)

                # Resize và PCA
                roi_resized = cv2.resize(roi_gray, SIZE_OPENCV)
                roi_flat = roi_resized.reshape(1, -1)
                roi_pca = pca.transform(roi_flat)
                
                # Dự đoán
                y_pred, distance = classifier.predict(roi_pca, return_distance=True)
                predicted_id = y_pred[0]
                dist_val = distance[0]
                
                # Logic xác định người lạ
                if dist_val > NGUONG_KHOANG_CACH:
                    name = "Người Lạ"
                    box_color = (0, 0, 255) # Đỏ
                else:
                    name = label_map.get(predicted_id, "Unknown")
                    if name == "Unknown": box_color = (0, 0, 255)

            except Exception as e:
                box_color = (0, 0, 255); name = "Lỗi"

            # Vẽ khung và thông tin
            cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
            
            # Hiển thị tên
            frame = vietnamese_text(frame, name, (x, y - 30), (255, 255, 255), font_size=24)

            # Chỉ hiện thông tin chi tiết nếu là người quen
            # (Tránh rối mắt khi có quá nhiều người lạ)
            if name not in ["Người Lạ", "Lỗi", "Unknown"] and name in info_map:
                details = info_map[name]
                current_y = y + h + 10 
                for key, value in details.items():
                    text_info = f"{key}: {value}"
                    frame = vietnamese_text(frame, text_info, (x, current_y), (255, 255, 0), font_size=18)
                    current_y += 25

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()