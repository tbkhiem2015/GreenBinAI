"""
detector.py — Pipeline inference cho model best.pt của repository gốc
has-bi/Plastic-Waste-Detection-YOLOv5s-Raspberry-Pi4.

Nguyên tắc (QUAN TRỌNG):
  * Model được load ĐÚNG CÁCH của repository gốc:
        from models.yolo import attempt_load      # framework YOLOv5 kèm trong repo
        from utils.general import non_max_suppression
  * Model chỉ load MỘT LẦN duy nhất (khi khởi tạo WasteDetector).
  * Mọi kết detection đều đến từ inference thật của best.pt — không có giả lập.
  * Không phụ thuộc CUDA/GPU (tự chọn cpu hoặc cuda nếu có).

Khác biệt so với code gốc trong repo (đã ghi rõ trong README):
  * Repo gốc resize ảnh thẳng về size×size (bóp méo) rồi scale box bằng
    scale_coords giả định letterbox -> box lệch trên ảnh không vuông.
    Ở đây dùng letterbox chuẩn YOLOv5 (pad màu 114) và scale box ngược CHÍNH XÁC
    theo (ratio, padding) -> đúng cả ảnh đứng lẫn ảnh ngang.
"""

from __future__ import annotations

import importlib
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

import config

# YOLOv5 chỉ được import một lần (thread-safe) — không load model mỗi frame.
_YOLOV5_LOCK = threading.Lock()
_YOLOV5_READY = False


def _ensure_yolov5_framework(framework_dir: Path) -> None:
    """Đưa framework YOLOv5 (kèm trong repository gốc) vào sys.path đúng 1 lần."""
    global _YOLOV5_READY
    with _YOLOV5_LOCK:
        if _YOLOV5_READY:
            return
        fw = Path(framework_dir).resolve()
        yolo_py = fw / "models" / "yolo.py"
        general_py = fw / "utils" / "general.py"
        if not yolo_py.exists() or not general_py.exists():
            raise FileNotFoundError(
                "Không tìm thấy framework YOLOv5 của repository gốc.\n"
                f"  Thiếu: {yolo_py if not yolo_py.exists() else general_py}\n"
                "Hãy chạy lệnh:  python setup_model.py\n"
                "(hoặc clone repository gốc vào thư mục yolov5/)"
            )
        sys.path.insert(0, str(fw))

        # Kiểm tra không import nhầm package 'models' khác (vd: thư mục models/
        # chứa best.pt của chính app này).
        yolo_mod = importlib.import_module("models.yolo")
        mod_file = Path(getattr(yolo_mod, "__file__", "")).resolve()
        expected = (fw / "models" / "yolo.py").resolve()
        if mod_file != expected:
            raise ImportError(
                f"Import nhầm module YOLOv5!\n  Đang dùng: {mod_file}\n  Kỳ vọng: {expected}"
            )
        _YOLOV5_READY = True


def _load_with_repo_code(weights_path: Path):
    """Load best.pt bằng chính attempt_load() của repository gốc.

    Xử lý tương thích:
      * torch >= 2.6 mặc định weights_only=True -> patch torch.load tạm thời.
      * attempt_download() của YOLOv5 xoá dấu nháy "'" khỏi đường dẫn
        (breaks các thư mục kiểu "D:\\BinAI'") -> tạm vô hiệu hoá vì model
        đã nằm local, không cần check/download.
      * Fallback cuối: copy best.pt sang thư mục temp (không ký tự đặc biệt)
        rồi load từ đó — vẫn là NGUYÊN file best.pt của repository gốc.
    """
    from models.yolo import attempt_load  # framework đã nằm trong sys.path
    import models.experimental as experimental_module

    orig_load = torch.load
    try:
        import inspect
        supports_weights_only = "weights_only" in inspect.signature(orig_load).parameters
    except (TypeError, ValueError):
        supports_weights_only = False

    def patched_load(*args, **kwargs):
        kwargs.setdefault("map_location", "cpu")
        if supports_weights_only:
            kwargs.setdefault("weights_only", False)
        return orig_load(*args, **kwargs)

    orig_attempt_download = getattr(experimental_module, "attempt_download", None)

    def _local_file_download(file, *args, **kwargs):
        return str(file)  # model local — không download, không sửa đường dẫn

    torch.load = patched_load
    if orig_attempt_download is not None:
        experimental_module.attempt_download = _local_file_download
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return attempt_load(str(weights_path))
            except (FileNotFoundError, OSError):
                # Fallback: load từ đường dẫn temp không chứa ký tự đặc biệt
                import shutil
                import tempfile
                safe_copy = Path(tempfile.gettempdir()) / "plastic_waste_best.pt"
                shutil.copy2(weights_path, safe_copy)
                return attempt_load(str(safe_copy))
            finally:
                pass
    finally:
        torch.load = orig_load
        if orig_attempt_download is not None:
            experimental_module.attempt_download = orig_attempt_download


def _read_names_from_data_yaml(data_yaml: Path) -> Optional[List[str]]:
    try:
        import yaml

        with open(data_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        names = data.get("names")
        if isinstance(names, dict):
            return [names[i] for i in sorted(names.keys())]
        if isinstance(names, (list, tuple)):
            return list(names)
    except Exception:
        pass
    return None


class WasteDetector:
    """Bao bọc model YOLOv5s (best.pt) để chạy inference trên ảnh/khung hình BGR."""

    def __init__(
        self,
        weights_path=config.MODEL_PATH,
        framework_dir=config.YOLOV5_DIR,
        imgsz: int = config.DEFAULT_IMGSZ,
        conf_thres: float = config.DEFAULT_CONF_THRES,
        iou_thres: float = config.DEFAULT_IOU_THRES,
        device: str = "auto",
    ) -> None:
        self.weights_path = Path(weights_path)
        self.framework_dir = Path(framework_dir)
        self.imgsz = int(imgsz)
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)

        # --- 1. Kiểm tra file model ---------------------------------------------
        if not self.weights_path.exists():
            raise FileNotFoundError(
                "Không tìm thấy models/best.pt\n"
                f"  Đường dẫn: {self.weights_path}\n"
                "Vui lòng kiểm tra file model.\n"
                "Chạy:  python setup_model.py  để tải best.pt từ repository gốc."
            )

        # --- 2. Nạp framework YOLOv5 từ repository gốc ---------------------------
        _ensure_yolov5_framework(self.framework_dir)

        # --- 3. Chọn device (KHÔNG bắt buộc CUDA) --------------------------------
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # --- 4. Load model đúng theo code của repository --------------------------
        try:
            model = _load_with_repo_code(self.weights_path)
        except Exception as exc:
            raise RuntimeError(
                f"Load model best.pt THẤT BẠT: {exc}\n"
                "Kiểm tra: file best.pt có bị lỗi? Phiên bản torch có tương thích?\n"
                "Xem mục Troubleshooting trong README.md"
            ) from exc

        model = model.to(self.device).float().eval()
        self.model = model

        # --- 5. Lấy class names THẬT nhúng trong checkpoint -----------------------
        raw_names = getattr(model, "names", None)
        if raw_names is None and hasattr(model, "module"):
            raw_names = getattr(model.module, "names", None)
        if isinstance(raw_names, dict):
            self.class_names = [str(raw_names[i]) for i in sorted(raw_names.keys())]
        elif isinstance(raw_names, (list, tuple)):
            self.class_names = [str(n) for n in raw_names]
        else:
            self.class_names = (
                _read_names_from_data_yaml(config.DATA_YAML)
                or list(config.FALLBACK_CLASS_NAMES)
            )

        # --- 6. Warm-up 1 lần để khung hình đầu không bị trễ ----------------------
        try:
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            self.detect(dummy)
        except Exception:
            pass  # warm-up không được phép làm crash app

    # ---------------------------------------------------------------------------
    # API công khai
    # ---------------------------------------------------------------------------
    def set_thresholds(self, conf_thres: float, iou_thres: float) -> None:
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)

    def set_imgsz(self, imgsz: int) -> None:
        self.imgsz = int(imgsz)

    def detect(self, frame_bgr) -> Tuple[np.ndarray, List[Dict], float]:
        """Chạy inference trên 1 khung hình BGR.

        Trả về: (ảnh_đã_vẽ_box BGR, danh_sách_detection, thời_gian_inference_giây)
        Mỗi detection: {"class_name": str, "confidence": float, "bbox": [x1,y1,x2,y2]}
        """
        if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
            return frame_bgr, [], 0.0

        img, ratio, (pad_x, pad_y) = self._letterbox(frame_bgr, self.imgsz)

        # BGR -> RGB, HWC -> CHW, scale [0,1]
        x = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))
        tensor = torch.from_numpy(x).float().div_(255.0).unsqueeze(0).to(self.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            pred = self.model(tensor, augment=False)[0]  # model trả (pred, list_layer)
        dt = time.perf_counter() - t0

        from utils.general import non_max_suppression  # framework trong sys.path

        det = non_max_suppression(
            pred, self.conf_thres, self.iou_thres, max_det=config.MAX_DET
        )[0]

        detections: List[Dict] = []
        if det is not None and len(det) > 0:
            # Scale ngược về toạ độ ảnh gốc (đảo letterbox)
            det[:, [0, 2]] -= pad_x
            det[:, [1, 3]] -= pad_y
            det[:, :4] /= ratio
            h0, w0 = frame_bgr.shape[:2]
            det[:, [0, 2]] = det[:, [0, 2]].clamp(0, w0 - 1)
            det[:, [1, 3]] = det[:, [1, 3]].clamp(0, h0 - 1)

            for *box, conf, cls_id in det.tolist():
                x1, y1, x2, y2 = [int(round(v)) for v in box]
                idx = int(cls_id)
                name = self.class_names[idx] if idx < len(self.class_names) else str(idx)
                detections.append(
                    {
                        "class_name": name,
                        "confidence": float(conf),
                        "bbox": [x1, y1, x2, y2],
                    }
                )

        annotated = self._draw(frame_bgr.copy(), detections)
        return annotated, detections, dt

    # ---------------------------------------------------------------------------
    # Helper nội bộ
    # ---------------------------------------------------------------------------
    @staticmethod
    def _letterbox(im, new_shape, color=(114, 114, 114)):
        """Letterbox chuẩn YOLOv5: giữ tỉ lệ khung hình, pad màu 114."""
        h0, w0 = im.shape[:2]
        ratio = min(new_shape / h0, new_shape / w0)
        nh, nw = max(1, round(h0 * ratio)), max(1, round(w0 * ratio))
        resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
        dh, dw = (new_shape - nh) / 2, (new_shape - nw) / 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        out = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )
        return out, ratio, (left, top)

    @staticmethod
    def _draw(image_bgr, detections):
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            name = det["class_name"]
            color = config.CLASS_COLORS.get(name, config.DEFAULT_COLOR)

            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)
            label = f"{name} {conf * 100:.1f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            ty = y1 - 8 if y1 - 10 > th else y1 + th + 8
            cv2.rectangle(image_bgr, (x1, ty - th - 4), (x1 + tw + 6, ty + 4), color, -1)
            cv2.putText(
                image_bgr, label, (x1 + 3, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA,
            )
        return image_bgr


