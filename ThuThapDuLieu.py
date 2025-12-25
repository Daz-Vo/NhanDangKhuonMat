import cv2
import os
import time
import json

# --- CẤU HÌNH ---
THU_MUC_DU_LIEU = 'data/raw'
SO_LUONG_ANH_CAN_CHUP = 60  # Số lượng ảnh cần thu thập
KHOANG_CACH_CHUP = 0.2      # Giây (Thời gian nghỉ giữa 2 lần chụp)

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
    """Tạo thư mục và lưu file info.json với dữ liệu đã nhập"""
    duong_dan_thu_muc = os.path.join(THU_MUC_DU_LIEU, ten_nguoi)
    
    # 1. Tạo thư mục
    if not os.path.exists(duong_dan_thu_muc):
        os.makedirs(duong_dan_thu_muc)
        print(f"-> Đã tạo thư mục: {duong_dan_thu_muc}")
    else:
        print(f"-> Thư mục '{ten_nguoi}' đã tồn tại. Sẽ cập nhật lại info.json và thêm ảnh.")

    # 2. Lưu file info.json
    duong_dan_json = os.path.join(duong_dan_thu_muc, 'info.json')
    try:
        with open(duong_dan_json, 'w', encoding='utf-8') as f:
            json.dump(data_info, f, indent=4, ensure_ascii=False)
        print("-> Đã lưu thông tin vào file 'info.json'.")
    except Exception as e:
        print(f"Lỗi khi lưu file json: {e}")
    
    return duong_dan_thu_muc

def main():
    print("=== CHƯƠNG TRÌNH THU THẬP DỮ LIỆU & TẠO PROFILE ===")
    
    # --- BƯỚC 1: NHẬP TÊN VÀ THÔNG TIN ---
    while True:
        ten_nguoi = input("1. Nhập tên của bạn (Viết liền không dấu, VD: VoVanDat): ").strip()
        if ten_nguoi:
            break
        print("Tên không được để trống!")

    # Gọi hàm nhập thông tin bổ sung
    thong_tin = nhap_thong_tin_bo_sung(ten_nguoi)
    
    # Tạo thư mục và lưu json
    save_path = tao_thu_muc_va_luu_json(ten_nguoi, thong_tin)

    # --- BƯỚC 2: CHỤP ẢNH ---
    print("\n------------------------------------------------")
    print(f"Chuan bi chup {SO_LUONG_ANH_CAN_CHUP} tam anh cho: {ten_nguoi}")
    input(">>> Nhấn Enter để BẬT CAMERA và bắt đầu chụp...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("LỖI: Không thể mở Camera.")
        return

    count = 0
    last_capture_time = time.time()
    
    # Tính toán tên file tiếp theo để không ghi đè ảnh cũ
    existing_files = [f for f in os.listdir(save_path) if f.endswith(('.jpg', '.png'))]
    start_index = len(existing_files) + 1

    print("\n>>> Đang chụp... Hãy nhìn vào camera và đổi góc mặt nhẹ.")
    print(">>> Nhấn 'q' để dừng sớm.\n")

    while count < SO_LUONG_ANH_CAN_CHUP:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1) # Lật ảnh gương
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

        if len(faces) == 1:
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            current_time = time.time()
            if current_time - last_capture_time > KHOANG_CACH_CHUP:
                count += 1
                roi_face = frame[y:y+h, x:x+w]
                
                # Lưu ảnh
                img_name = os.path.join(save_path, f"{start_index + count - 1}.jpg")
                cv2.imwrite(img_name, roi_face)
                print(f"   + Đã lưu: {count}/{SO_LUONG_ANH_CAN_CHUP}")
                
                last_capture_time = current_time
        
        # Hiển thị số lượng
        cv2.putText(frame, f"Tien do: {count}/{SO_LUONG_ANH_CAN_CHUP}", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("Thu Thap Du Lieu", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if count >= 1:
        print(f"\n>>> HOÀN TẤT! Đã lưu {count} ảnh và file thông tin.")
        print(">>> ĐỪNG QUÊN: Chạy lại 'python main.py' để cập nhật dữ liệu mới này!")
    else:
        print("\n>>> Đã hủy bỏ quá trình.")

if __name__ == "__main__":
    main()