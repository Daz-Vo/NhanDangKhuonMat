import cv2
import pickle
import numpy as np
import os
import ctypes
import json
from PIL import Image, ImageDraw, ImageFont 

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
NGUONG_KHOANG_CACH = 3000

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
        print(f"LỖI tải file: {e}")
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

def vietnamese_text(img, text, position, text_color, font_size=20, outline_color=(0,0,0)):
    cv2_im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im_rgb)
    draw = ImageDraw.Draw(pil_im)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    x, y = position
    if outline_color:
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx != 0 or dy != 0:
                    draw.text((x+dx, y+dy), text, font=font, fill=outline_color)
    draw.text(position, text, font=font, fill=text_color)
    return cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

def main():
    pca, classifier, label_map, info_map = load_artifacts()
    if pca is None: return
    
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): print("LỖI: Không thể mở Camera."); return
    
    center_window(cap)
    
    print(">>> Camera sẵn sàng. Chế độ: GƯƠNG (MIRROR).")
    print(f">>> Ngưỡng khoảng cách: {NGUONG_KHOANG_CACH}")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # --- ĐỒNG BỘ: LẬT ẢNH (MIRROR) ---
        # Phải giống hệt file thu thập dữ liệu
        frame = cv2.flip(frame, 1) 
        # ---------------------------------

        try:
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1: break
        except: pass
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Cấu hình phát hiện (Đã tối ưu cho nhiều người)
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=6,     
            minSize=(80, 80)    
        )

        for (x, y, w, h) in faces:
            box_color = (0, 255, 0) 
            name = "Unknown"
            
            try:
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.equalizeHist(roi_gray) # Cân bằng sáng

                roi_resized = cv2.resize(roi_gray, SIZE_OPENCV)
                roi_flat = roi_resized.reshape(1, -1)
                roi_pca = pca.transform(roi_flat)
                
                y_pred, distance = classifier.predict(roi_pca, return_distance=True)
                predicted_id = y_pred[0]
                dist_val = distance[0]
                
                if dist_val > NGUONG_KHOANG_CACH:
                    name = "Người Lạ"
                    box_color = (0, 0, 255)
                else:
                    name = label_map.get(predicted_id, "Unknown")
                    if name == "Unknown": box_color = (0, 0, 255)

            except Exception as e:
                box_color = (0, 0, 255); name = "Lỗi"

            cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
            frame = vietnamese_text(frame, name, (x, y - 30), (255, 255, 255), font_size=24)

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