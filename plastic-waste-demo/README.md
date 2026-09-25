# ♻️ Plastic Waste Detection — YOLOv5s AI Demo

Ứng dụng demo **GUI kiểm thử model nhận diện rác thải** bằng chính file
`best.pt` của repository gốc:

> **[has-bi/Plastic-Waste-Detection-YOLOv5s-Raspberry-Pi4](https://github.com/has-bi/Plastic-Waste-Detection-YOLOv5s-Raspberry-Pi4)**

Ứng dụng **KHÔNG train lại model, KHÔNG dùng model khác, KHÔNG giả lập kết quả** —
mọi detection đều đến từ inference thật của `best.pt` thông qua framework YOLOv5
kèm trong repository gốc (`models.yolo.attempt_load` + `utils.general.non_max_suppression`).

## ✨ Tính năng

- **Mode 1 — Nhận diện ảnh**: chọn file `.jpg/.jpeg/.png/.webp`, chạy inference,
  hiển thị bounding box + class + confidence, lưu ảnh kết quả vào `outputs/`.
- **Mode 2 — Nhận diện webcam realtime**: Start/Stop Camera, bounding box +
  class + confidence + **FPS** (thread riêng, GUI không bị treo).
- Chỉnh được **Confidence threshold** (mặc định `0.25`), **Input size**
  (`320/480/640` — chọn `320` cho Raspberry Pi) và **inference interval** (mỗi N khung).
- Class names được đọc **trực tiếp từ trong checkpoint** (`model.names`) —
  mapping thật lúc train, fallback về `data.yaml` của repo gốc:
  `cable, plastic bag, plastic bottle, plastic cup, soap bottle, sterofoam`.

## 📂 Cấu trúc project

```text
plastic-waste-demo/
├── app.py               # GUI (Tkinter, dark theme)
├── detector.py          # Load best.pt 1 lần + pipeline inference
├── config.py            # Toàn bộ cấu hình (đường dẫn, ngưỡng, màu class...)
├── firebase_store.py    # Đẩy kết quả detection lên Firebase Realtime Database
├── test_model.py        # Script kiểm tra model bằng CLI (chạy TRƯỚC GUI)
├── setup_model.py       # Tải best.pt + framework YOLOv5 từ repository gốc
├── run_gui.bat          # Double-click để mở GUI (tự dùng venv nếu có)
├── run_test.bat         # Double-click để test model bằng CLI
├── _smoke_gui.py        # (tuỳ chọn) smoke-test GUI không cần tương tác
├── requirements.txt
├── README.md
├── models/
│   └── best.pt          # Model gốc của repository (setup_model.py tự tải)
├── yolov5/              # Framework YOLOv5 của repo gốc: models/ + utils/ + data.yaml
├── assets/              # Ảnh test mẫu
└── outputs/             # Ảnh kết quả
```

**Vì sao có thư mục `yolov5/`?** — `best.pt` là checkpoint YOLOv5 được save với
`torch.save`, khi unpickle Python cần đúng các module `models.yolo`,
`models.common`, `utils.general`... Repository gốc chứa NGUYÊN BỘ framework YOLOv5
này, nên `setup_model.py` copy `models/` + `utils/` từ repo vào `yolov5/` để
`detector.py` import **chính framework của repo** — tương thích 100% với
`best.pt`, không cần clone yolov5 của Ultralytics, chạy offline.

---

## 🛠️ Installation (Windows 11 + Python 3.10)

### 1. Kiểm tra Python

```bat
python --version
```
Kết quả mong đợi: `Python 3.10.x` (3.10–3.12 đều chạy được; khuyến nghị 3.10).

### 2. Tạo virtual environment

```bat
cd plastic-waste-demo
python -m venv venv
venv\Scripts\activate
```

### 3. Cài dependencies

```bat
pip install --upgrade pip
pip install -r requirements.txt
```

**PyTorch:** trên Windows, `pip install torch==2.2.2` mặc định đã là bản **CPU**
(gọn, không cần GPU). Nếu bạn muốn cài rõ ràng bản CPU:

```bat
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
```

Nếu máy có NVIDIA GPU (tuỳ chọn, không bắt buộc):

```bat
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
```

### 4. Đặt best.pt + framework YOLOv5 (từ repository gốc)

```bat
python setup_model.py
```

Script sẽ tự động:

1. Tải ZIP repository `has-bi/Plastic-Waste-Detection-YOLOv5s-Raspberry-Pi4` từ GitHub
2. Copy `best.pt` → `models/best.pt` (~14 MB)
3. Copy `models/` + `utils/` (framework YOLOv5 của repo) → `yolov5/`
4. Copy `data.yaml` → `yolov5/data.yaml`
5. Tải 1–2 ảnh test mẫu (chai nhựa) vào `assets/`

**Cách thủ công** (nếu không dùng script):

```bat
git clone https://github.com/has-bi/Plastic-Waste-Detection-YOLOv5s-Raspberry-Pi4.git
copy Plastic-Waste-Detection-YOLOv5s-Raspberry-Pi4\best.pt models\best.pt
xcopy Plastic-Waste-Detection-YOLOv5s-Raspberry-Pi4\models yolov5\models\ /E /I
xcopy Plastic-Waste-Detection-YOLOv5s-Raspberry-Pi4\utils yolov5\utils\ /E /I
copy Plastic-Waste-Detection-YOLOv5s-Raspberry-Pi4\data.yaml yolov5\data.yaml
```
(hoặc `python setup_model.py --repo Plastic-Waste-Detection-YOLOv5s-Raspberry-Pi4`)

### 5. Chạy ứng dụng

```bat
python app.py
```

---

## 🧪 Kiểm tra model trước khi dùng GUI

```bat
python test_model.py
```

Script sẽ:

1. Load `best.pt` (một lần duy nhất)
2. Load ảnh test trong `assets/` (tự tải ảnh mẫu chai nhựa nếu thư mục trống,
   hoặc chỉ định: `python test_model.py --image "D:\anh\chai_nhua.jpg"`)
3. Chạy inference thật
4. In detection ra terminal:

```text
Detected objects:
  plastic bottle    92.4%   bbox=[120,88,410,623]
  plastic bag       87.1%   bbox=[...]
```

5. Lưu ảnh kết quả vào `outputs/test_result.jpg`

Tùy chọn: `--imgsz 320` (nhanh hơn), `--conf 0.4`, `--iou 0.45`, `--no-save`.

**Chỉ khi bước này in ra detection và lưu được ảnh kết quả** thì pipeline
inference đã OK — khi đó mới dùng GUI.

## 🖼️ Cách test bằng ảnh (GUI)

1. Chạy `python app.py` → đợi dòng status *"Model sẵn sàng"*.
2. Bấm **📁 Chọn ảnh** → chọn file `.jpg/.jpeg/.png/.webp`.
3. Xem ảnh đã vẽ bounding box + nhãn `class confidence%`.
4. Danh sách detection hiện ở panel bên phải; ảnh kết quả tự lưu vào `outputs/`.
5. Kéo slider **Confidence** để tăng/giảm ngưỡng rồi chọn lại ảnh để so sánh.

## 📷 Cách test bằng webcam (GUI)

1. Bấm **📷 Start Camera** → cấp quyền truy cập camera nếu Windows hỏi.
2. Đưa chai nhựa/túi nilon... vào khung hình → box + class + confidence hiện ngay.
3. **FPS** hiển thị ở panel kết quả.
4. Bấm **⏹ Stop Camera** để dừng; **📸 Lưu khung hình** để lưu ảnh hiện tại.
5. Nếu máy có nhiều camera, đổi `WEBCAM_INDEX = 1` trong `config.py`.

## 🔥 Lưu kết quả lên Firebase

Sau **mỗi lần phân tích xong**, app đẩy 1 bản ghi lên **Firebase Realtime Database**:

```json
{
  "timestamp": "2026-09-20 09:37:15",
  "epoch": 1758339435.123,
  "source": "image",
  "is_plastic": true,
  "class_name": "plastic bottle",
  "confidence": 0.58,
  "confidence_percent": 58.0,
  "num_objects": 1,
  "all_detections": [ {"class_name": "plastic bottle", "confidence": 0.58, "bbox": [4,0,565,416]} ]
}
```

Giải thích các trường:

| Trường | Ý nghĩa |
|---|---|
| `timestamp` / `epoch` | Thời gian phân tích (định dạng người đọc + epoch để sort) |
| `is_plastic` | **plastic / ko plastic** — `true` với `plastic bag`, `plastic bottle`, `plastic cup`, `soap bottle`, `sterofoam`; `false` với `cable` hoặc không phát hiện gì |
| `class_name` | Class điểm cao nhất (null nếu không phát hiện) |
| `confidence` | Độ tin cậy 0–1 (kèm `confidence_percent` 0–100) |
| `source` | `"image"` (Chọn ảnh) hoặc `"webcam"` |

### Cấu hình Firebase (5 phút)

1. Vào [Firebase Console](https://console.firebase.google.com) → **Add project** (hoặc dùng project có sẵn).
2. **Build → Realtime Database → Create Database** → chọn region → **Start in test mode**.
3. Copy **Database URL** dạng `https://<project-id>-default-rtdb.firebaseio.com`.
4. Mở `config.py`, điền:

   ```python
   FIREBASE_ENABLED = True
   FIREBASE_DB_URL = "https://<project-id>-default-rtdb.firebaseio.com"
   FIREBASE_DB_SECRET = ""    # điền Database Secret nếu rules yêu cầu auth
   ```

5. (Tuỳ chọn bảo mật) Nếu **không** dùng test mode, vào tab **Rules** đặt:

   ```json
   { "rules": { ".read": false, ".write": false, "detections": { ".read": true, ".write": true } } }
   ```
   rồi lấy **Database secret**: Project Settings → Service accounts → Database secrets.
6. Chạy lại `python app.py` — sau mỗi lần nhận diện ảnh, trạng thái dưới cùng hiện **`🔥 Firebase: đã lưu`**.

### Hành vi đẩy dữ liệu

| Chế độ | Khi nào đẩy |
|---|---|
| 🖼️ Chọn ảnh | **Mỗi lần** nhận diện xong → 1 bản ghi |
| 📷 Webcam | Chỉ khi **kết quả thay đổi** (đổi class/điểm số, hoặc mất dấu vật) và cách lần đẩy trước ≥ `WEBCAM_FIREBASE_MIN_INTERVAL_SEC` (mặc định 5s) → tránh spam DB |

Xem dữ liệu: Firebase Console → Realtime Database → node `detections/`, mỗi bản ghi 1 key tự sinh. Module đẩy dữ liệu nằm ở `firebase_store.py` — tái sử dụng được cho script riêng.

> ⚠️ **Lưu ý bảo mật:** chế độ test mode / rules public chỉ dùng cho demo nội bộ. Khi đưa lên production, dùng rules có auth + secret.

## 🤖 Model

| Mục | Giá trị |
|---|---|
| File weights | `best.pt` — **nguyên bản** từ repository gốc |
| Kiến trúc | YOLOv5s (PyTorch) |
| Cách load | `models.yolo.attempt_load()` — chính code của repository gốc |
| Post-process | `utils.general.non_max_suppression` (NMS, IoU 0.45) |
| Classes (6) | `cable`, `plastic bag`, `plastic bottle`, `plastic cup`, `soap bottle`, `sterofoam` |
| Thứ tự class | Đọc trực tiếp từ checkpoint (`model.names`); fallback theo `data.yaml` |

> ⚠️ Lưu ý: file `classes.txt` của repo gốc có **thứ tự khác** với `data.yaml`.
> Ứng dụng ưu tiên mapping nhúng trong `best.pt` (mapping thật lúc train).

## ⚙️ Các tham số trong `config.py`

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `DEFAULT_IMGSZ` | 640 | Kích thước input (320/480/640 — bội số của 32) |
| `DEFAULT_CONF_THRES` | 0.25 | Ngưỡng confidence |
| `DEFAULT_IOU_THRES` | 0.45 | Ngưỡng IoU cho NMS |
| `WEBCAM_INDEX` | 0 | ID camera (0 = mặc định) |
| `MAX_IMAGE_SIDE` | 1600 | Thu nhỏ ảnh lớn trước khi inference |
| `WEBCAM_WIDTH/HEIGHT` | 1280×720 | Độ phân giải yêu cầu từ webcam |

## 🐞 Troubleshooting (lỗi thường gặp)

| Hiện tượng | Nguyên nhân / Cách xử lý |
|---|---|
| `Không tìm thấy models/best.pt` | Chưa chạy `python setup_model.py` — hoặc copy tay `best.pt` vào `models/` |
| `Load model best.pt THẤT BẠT` / `No module named ...` | Chưa cài đủ dependencies (framework YOLOv5 của repo cần `ultralytics==8.0.43`, `psutil`, `setuptools<81`) → chạy `pip install -r requirements.txt`. torch ≥ 2.6 vẫn chạy được — `detector.py` đã tự set `weights_only=False` |
| `No such file or directory: 'D:\BinAI\...\best.pt'` (thất dấu nháy) | YOLOv5 `attempt_download()` xoá dấu `'` khỏi đường dẫn — thư mục project chứa dấu nháy. `detector.py` đã vô hiệu hoá hàm này; **không** đặt project trong thư mục có `'` hoặc dấu tiếng Việt là an toàn nhất |
| `Import nhầm module YOLOv5` | Thư mục `yolov5/models/` không đúng nguồn repo. Xoá `yolov5/` và chạy lại `setup_model.py` |
| `Không thể mở webcam` | Camera bị app khác chiếm, hoặc index sai → đổi `WEBCAM_INDEX` trong `config.py`; kiểm tra Windows Settings → Privacy → Camera |
| Ảnh không hợp lệ | File không phải ảnh chuẩn → dùng `.jpg/.jpeg/.png/.webp` |
| Không nhận được vật thể | Hạ confidence (0.15–0.25), dùng ảnh chụp gần, đủ sáng, vật chiếm ≥ 1/4 khung hình |
| FPS thấp (Windows CPU) | Giảm `Input size` xuống 320/480; hoặc tăng "Infer mỗi N khung" lên 2–3 |
| Lỗi `matplotlib`/`pandas` import | Chưa cài đủ requirements → `pip install -r requirements.txt` |
| Tkinter báo lỗi font "Segoe UI" | Chỉ thấy trên hệ ngoài Windows → app tự fallback font hệ thống (trên Pi dùng default) |
| `Illegal instruction` (crash ~1 phút sau khi chạy, ngay sau `Fusing layers...`) trên Pi | SIGILL từ OpenBLAS/NumPy dispatch sai tập lệnh CPU cho Cortex-A72 → xem mục "Illegal instruction trên Pi 4" bên dưới |

## 🍓 Chuyển sang Raspberry Pi 4

Code được viết **không phụ thuộc API riêng của Windows** (chỉ dùng OpenCV,
Tkinter, NumPy, PyTorch — tất cả chạy trên Linux/ARM):

```text
USB Webcam → Raspberry Pi 4 → best.pt (YOLOv5s) → Detection → Tkinter GUI
```

1. Cài Python ≥ 3.9 trên Pi OS / Ubuntu (repo gốc chạy Ubuntu trên Pi 4).

   ```bash
   cd plastic-waste-demo
   python3 -m venv venv
   source venv/bin/activate
   # torch/torchvision CPU cho ARM (aarch64):
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   # nếu PyPI không có wheel aarch64, dùng piwheels:
   # pip install torch torchvision --extra-index-url https://www.piwheels.org/simple
   pip install -r requirements.txt
   python setup_model.py
   python app.py
   ```

2. **Tối ưu cho Pi 4** (đã có sẵn trong app):
   - Đặt **Input size = 320** ( Combobox trong GUI) — nhanh ~3–4 lần so với 640.
   - Tăng **"Infer mỗi N khung"** lên 2–5: vẫn hiển thị video mượt, chỉ chạy
     inference đúng tần suất (không inference dư thừa, không ghi đĩa mỗi frame).
   - Model load đúng **1 lần** khi khởi động — không reload mỗi frame.
   - Không dùng CUDA — tự chạy CPU (`torch.cuda.is_available()` tự phát hiện).
3. Hiệu năng tham khảo (từ README repo gốc): ~**2 FPS @ 640px** trên Pi 4B;
   với `imgsz=320` + interval 2–3, trải nghiệm demo realtime ở mức khả dụng.
4. **Servo SG92R phân loại kết quả** (`servo_controller.py`):
   - Đấu servo: dây tín hiệu (thường màu vàng/cam) → **GPIO17** (BCM, PIN vật lý
     **11**); dây nguồn 5V → PIN 2/4; dây GND → PIN 6/9/14... (bất kỳ chân GND nào).
   - Nhận diện ra **plastic** → servo quay **phải** `SERVO_ANGLE_OFFSET` độ (mặc
     định 60°), giữ `SERVO_HOLD_SECONDS` giây (mặc định 2s), rồi về vị trí cũ.
   - Nhận diện ra **không phải plastic** → servo quay **trái** tương tự.
   - Cài `pip install RPi.GPIO` (chỉ chạy trên Pi/Linux — trong `requirements.txt`
     đã đánh dấu marker để không cài nhầm trên Windows).
   - Chỉnh góc/thời gian/chân GPIO trong `config.py` (`SERVO_*`); đặt
     `SERVO_ENABLED = False` để tắt hẳn khi demo trên PC không có GPIO.
   - Ở chế độ webcam, servo chỉ trigger khi **kết quả phân loại thay đổi** (tránh
     quay liên tục khi cùng 1 vật đứng yên trước camera); ở chế độ ảnh, mỗi lần
     chọn ảnh luôn trigger 1 lần theo kết quả.

### ⚠️ `Illegal instruction` trên Pi 4

Crash (SIGILL) thường xảy ra ngay sau dòng `Fusing layers...` (bước `model.fuse()`
trong `attempt_load`) — là lỗi kinh điển của OpenBLAS/NumPy trên ARM: thư viện tự
phát hiện sai loại lõi CPU và dùng tập lệnh mà Cortex-A72 không hỗ trợ, chứ
không phải lỗi ở code app hay ở model.

1. Xác nhận đang chạy Pi OS **64-bit** (bắt buộc, không dùng bản 32-bit):
   ```bash
   uname -m   # phải in ra aarch64 — nếu ra armv7l, phải cài lại Pi OS 64-bit
   ```
2. Ép OpenBLAS nhận đúng loại lõi và tắt đa luồng (thường fix được ngay):
   ```bash
   export OPENBLAS_CORETYPE=ARMV8
   export OPENBLAS_NUM_THREADS=1
   export OMP_NUM_THREADS=1
   python app.py
   ```
   Nếu hết lỗi, thêm 3 dòng `export` này vào `~/.bashrc` (hoặc đầu `run_gui.bat`
   tương đương trên Pi) để không phải gõ lại mỗi lần chạy.
3. Nếu vẫn crash: cài lại `torch`/`numpy` đúng wheel `aarch64` chính thức (không
   qua piwheels) rồi cài lại requirements:
   ```bash
   pip uninstall -y torch torchvision numpy
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```

### 🚀 Tự khởi động cùng Raspberry Pi 4 (autostart)

Thay vì phải gõ tay `cd` → `source venv` → `export ...` → `python app.py` mỗi
lần bật máy, dùng script `run_pi.sh` (đã gộp sẵn các bước đó, kèm log ra
`outputs/run.log` và tự khởi động lại nếu app bị crash) + autostart theo
desktop session của Pi OS.

1. Bật auto-login vào Desktop (bắt buộc — autostart chỉ chạy khi có desktop
   session, không chạy ở màn hình đăng nhập):
   ```bash
   sudo raspi-config
   # System Options -> Boot / Auto Login -> Desktop Autologin
   ```
2. Cấp quyền thực thi cho script:
   ```bash
   cd ~/GreenBinAI/plastic-waste-demo
   chmod +x run_pi.sh
   ```
3. Copy file mẫu `greenbinai-autostart.desktop` vào thư mục autostart, sửa lại
   đường dẫn `Exec=` cho đúng vị trí thật của project (nếu username/thư mục
   khác `pi`/`GreenBinAI`):
   ```bash
   mkdir -p ~/.config/autostart
   cp greenbinai-autostart.desktop ~/.config/autostart/
   nano ~/.config/autostart/greenbinai-autostart.desktop   # sửa dòng Exec= nếu cần
   ```
4. Khởi động lại để kiểm tra:
   ```bash
   sudo reboot
   ```
   App sẽ tự mở sau khi desktop load xong (delay 5s theo `X-GNOME-Autostart-Delay`
   để camera/USB kịp sẵn sàng). Xem log tại `outputs/run.log` nếu app không lên.
5. Tắt autostart: xoá file khỏi `~/.config/autostart/`:
   ```bash
   rm ~/.config/autostart/greenbinai-autostart.desktop
   ```

## 📄 License / Credits

- Model & framework YOLOv5: repository
  [has-bi/Plastic-Waste-Detection-YOLOv5s-Raspberry-Pi4](https://github.com/has-bi/Plastic-Waste-Detection-YOLOv5s-Raspberry-Pi4)
  (dataset Roboflow *plastic-waste-detection-x8dvc*, CC BY 4.0).
- YOLOv5: Ultralytics, AGPL-3.0.
- Ảnh test mẫu: Wikimedia Commons (CC BY-SA).
- Demo app này chỉ dùng để kiểm thử model — không phục vụ mục đích thương mại.
