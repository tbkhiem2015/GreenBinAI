"""
config.py — Cấu hình trung tâm của ứng dụng demo Plastic Waste Detection.

Toàn bộ đường dẫn / ngưỡng mặc định được gom về đây để dễ điều chỉnh và
dễ chuyển sang Raspberry Pi 4 (chỉ cần đổi giá trị ở đây, không phải sửa code).
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Đường dẫn project
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

# File weights GỐC của repository has-bi/Plastic-Waste-Detection-YOLOv5s-Raspberry-Pi4
MODEL_PATH = BASE_DIR / "models" / "best.pt"

# Framework YOLOv5 được lấy NGUYÊN VẸN từ repository gốc (chứa models/ + utils/)
# detector.py sẽ import `models.yolo.attempt_load` và `utils.general.non_max_suppression`
# CHÍNH TỪ thư mục này -> đảm bảo tương thích 100% với best.pt.
YOLOV5_DIR = BASE_DIR / "yolov5"
DATA_YAML = YOLOV5_DIR / "data.yaml"

ASSETS_DIR = BASE_DIR / "assets"      # ảnh test mẫu
OUTPUT_DIR = BASE_DIR / "outputs"     # ảnh kết quả được lưu vào đây

# ---------------------------------------------------------------------------
# Tham số inference
# ---------------------------------------------------------------------------
DEFAULT_IMGSZ = 640        # kích thước input model (bội số của 32). Pi 4: dùng 320 để nhanh hơn.
DEFAULT_CONF_THRES = 0.25  # ngưỡng confidence mặc định
DEFAULT_IOU_THRES = 0.45   # ngưỡng IoU cho NMS
MAX_DET = 300              # số object tối đa mỗi lần inference

# Ảnh tĩnh quá lớn sẽ được thu nhỏ trước khi inference (tăng tốc, đủ cho demo)
MAX_IMAGE_SIDE = 1600

# Webcam
WEBCAM_INDEX = 0           # ID camera (0 = camera mặc định). Đổi thành 1/2 nếu có nhiều camera.
WEBCAM_WIDTH = 1280        # yêu cầu độ phân giải (OpenCV sẽ dùng giá trị gần nhất được hỗ trợ)
WEBCAM_HEIGHT = 720

# Tên các class FALLBACK CUỐI CÙNG theo thứ tự trong data.yaml của repository gốc:
#   cable, plastic bag, plastic bottle, plastic cup, soap bottle, sterofoam
# LƯU Ý: ứng dụng ưu tiên đọc class names NHÚNG TRỰC TIẾP trong best.pt
# (model.names) -> đây mới là mapping thật lúc train. Giá trị này chỉ dùng khi
# không đọc được names từ checkpoint.
FALLBACK_CLASS_NAMES = [
    "cable",
    "plastic bag",
    "plastic bottle",
    "plastic cup",
    "soap bottle",
    "sterofoam",
]

# Màu vẽ bounding box (BGR) cho từng class
CLASS_COLORS = {
    "cable": (255, 165, 0),
    "plastic bag": (80, 220, 60),
    "plastic bottle": (60, 200, 250),
    "plastic cup": (0, 170, 255),
    "soap bottle": (200, 100, 255),
    "sterofoam": (100, 100, 255),
}
DEFAULT_COLOR = (46, 204, 113)

# Địa chỉ tải tài nguyên (dùng bởi setup_model.py / test_model.py)
REPO_URL = "https://github.com/has-bi/Plastic-Waste-Detection-YOLOv5s-Raspberry-Pi4"
REPO_ZIP_URL = f"{REPO_URL}/archive/refs/heads/main.zip"
SAMPLE_IMAGE_URLS = [
    # Ảnh chai nhựa rác thải (Wikimedia Commons, giấy phép CC BY-SA 4.0)
    "https://upload.wikimedia.org/wikipedia/commons/b/bf/Waste_Plastic_Bottles.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/5/55/Empty_Plastic_Bottle.jpg",
]

# ---------------------------------------------------------------------------
# Firebase — lưu kết quả detection lên Realtime Database
# Cách cấu hình xem README.md mục "Lưu kết quả lên Firebase"
# ---------------------------------------------------------------------------
FIREBASE_ENABLED = True
FIREBASE_DB_URL = "https://binai-652a7-default-rtdb.asia-southeast1.firebasedatabase.app"   # Firebase Realtime Database của project binai-652a7
FIREBASE_DB_SECRET = ""   # Database secret (nếu rules yêu cầu auth). "" nếu rules public
FIREBASE_NODE = "detections"  # node cha trên Realtime Database

# Class nào được tính là "plastic" (cable = dây điện → KHÔNG tính là plastic)
PLASTIC_CLASSES = {
    "plastic bag",
    "plastic bottle",
    "plastic cup",
    "soap bottle",
    "sterofoam",
}

# Webcam: chỉ đẩy lên Firebase khi KẾT QUẢ THAY ĐỔI và cách nhau ít nhất X giây
WEBCAM_FIREBASE_MIN_INTERVAL_SEC = 5.0

