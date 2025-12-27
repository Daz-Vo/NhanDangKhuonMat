import cv2
import os
import time
import json

# --- CẤU HÌNH ---
THU_MUC_DU_LIEU = 'data/raw'
SO_LUONG_ANH_CAN_CHUP = 60  # Chụp 60 tấm
KHOANG_CACH_CHUP = 0.2      # Tốc độ chụp nhanh (0.2s)

# Load bộ phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def nhap_thong_tin_bo_sung(ten_nguoi):
    """Hàm hỏi người dùng nhập thông tin chi tiết"""
    print(f"\n--- NHẬP THÔNG TIN CHI TIẾT CHO: {ten_nguoi} ---")
    print("(Nếu không muốn nhập, hãy bấm Enter để bỏ qua)")
    
    ho_ten = input(">> Nhập Họ Tên: ").strip()
    if not ho_ten: ho_ten = "Chua cap nhat"

    msv = input(">> Nhập Mã Sinh Viên (MSV): ").strip()
    if not msv: msv = "Chua cap nhat"
    
    lop = input(">> Nhập Lớp: ").strip()
    if not lop: lop = "Chua cap nhat"
    
    nam_sinh = input(">> Nhập Năm Sinh/Ghi Chú: ").strip()
    if not nam_sinh: nam_sinh = ""

    # Tạo dictionary dữ liệu
    data_info = {
        "Họ Tên": ho_ten,
        "MSV": msv,
        "Lop": lop,
        "Ghi Chu": nam_sinh
    }
    return data_info

def tao_thu_muc_va_luu_json(ten_nguoi, data_info):
    duong_dan_thu_muc = os.path.join(THU_MUC_DU_LIEU, ten_nguoi)
    
    if not os.path.exists(duong_dan_thu_muc):
        os.makedirs(duong_dan_thu_muc)
    
    duong_dan_json = os.path.join(duong_dan_thu_muc, 'info.json')
    try:
        with open(duong_dan_json, 'w', encoding='utf-8') as f:
            json.dump(data_info, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Lỗi lưu JSON: {e}")
    
    return duong_dan_thu_muc

def main():
    print("=== TOOL THU THẬP DỮ LIỆU (CHẾ ĐỘ GƯƠNG) ===")
    
    while True:
        ten_nguoi = input("1. Nhập tên folder (Viết liền không dấu, VD: VoVanDat): ").strip()
        if ten_nguoi: break
        print("Tên không được để trống!")

    thong_tin = nhap_thong_tin_bo_sung(ten_nguoi)
    save_path = tao_thu_muc_va_luu_json(ten_nguoi, thong_tin)

    print("\n------------------------------------------------")
    print(f"Chuẩn bị chụp {SO_LUONG_ANH_CAN_CHUP} ảnh.")
    input(">>> Nhấn Enter để BẬT CAMERA...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): print("Lỗi Camera"); return

    count = 0
    last_capture_time = time.time()
    
    # Tính index để không ghi đè
    existing_files = [f for f in os.listdir(save_path) if f.endswith(('.jpg', '.png'))]
    start_index = len(existing_files) + 1

    print(">>> Đang chụp... Hãy di chuyển khuôn mặt nhẹ nhàng.")

    while count < SO_LUONG_ANH_CAN_CHUP:
        ret, frame = cap.read()
        if not ret: break
        
        # --- ĐỒNG BỘ: LẬT ẢNH (MIRROR) ---
        frame = cv2.flip(frame, 1) 
        # ---------------------------------

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

        if len(faces) == 1:
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            current_time = time.time()
            if current_time - last_capture_time > KHOANG_CACH_CHUP:
                count += 1
                roi_face = frame[y:y+h, x:x+w]
                
                img_name = os.path.join(save_path, f"{start_index + count - 1}.jpg")
                cv2.imwrite(img_name, roi_face)
                print(f" -> Đã lưu: {count}/{SO_LUONG_ANH_CAN_CHUP}")
                last_capture_time = current_time
        
        cv2.putText(frame, f"Tien do: {count}/{SO_LUONG_ANH_CAN_CHUP}", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Thu Thap Du Lieu", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    if count >= 1:
        print("\n>>> HOÀN TẤT! Nhớ chạy lại 'python main.py' để huấn luyện lại!")

if __name__ == "__main__":
    main()