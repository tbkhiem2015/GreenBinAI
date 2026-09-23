"""
test_model.py — Kiểm tra model best.pt TRƯỚC KHI dùng GUI.

Quy trình:
    1. Load best.pt (một lần duy nhất)
    2. Load 1 ảnh test (mặc định lấy ảnh mẫu trong assets/, tự tải nếu thiếu)
    3. Chạy inference thật bằng framework YOLOv5 của repository gốc
    4. In detection ra terminal
    5. Lưu ảnh kết quả vào outputs/

Cách chạy:
    python test_model.py                       # dùng ảnh mẫu tự tải
    python test_model.py --image duong/dan/anh.jpg
    python test_model.py --imgsz 320 --conf 0.4
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

import config
from detector import WasteDetector


def _force_utf8_stdout() -> None:
    """Đảm bảo in được tiếng Việt trên Windows (mặc định console là cp1252)."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


_force_utf8_stdout()


def download(url: str, dest: Path) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
            f.write(r.read())
        return True
    except Exception as exc:
        print(f"[WARN] Không tải được {url}: {exc}")
        return False


def get_test_image(path_arg: str | None) -> Path | None:
    """Ưu tiên ảnh do người dùng chỉ định, sau đó tìm trong assets/, cuối cùng tải mẫu."""
    if path_arg:
        p = Path(path_arg)
        return p if p.exists() else None

    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        for p in sorted(config.ASSETS_DIR.glob(pattern)):
            return p

    print("[INFO] Chưa có ảnh test trong assets/ — đang tải ảnh mẫu (chai nhựa)...")
    config.ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    for url in config.SAMPLE_IMAGE_URLS:
        dest = config.ASSETS_DIR / Path(url).name
        if download(url, dest) and dest.stat().st_size > 10_000:
            print(f"[OK] Đã tải ảnh mẫu: {dest}")
            return dest
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Kiểm tra model best.pt")
    parser.add_argument("--weights", default=str(config.MODEL_PATH), help="Đường dẫn best.pt")
    parser.add_argument("--image", default=None, help="Ảnh test (mặc định: ảnh trong assets/)")
    parser.add_argument("--imgsz", type=int, default=config.DEFAULT_IMGSZ, help="Input size (320/480/640)")
    parser.add_argument("--conf", type=float, default=config.DEFAULT_CONF_THRES, help="Ngưỡng confidence")
    parser.add_argument("--iou", type=float, default=config.DEFAULT_IOU_THRES, help="Ngưỡng IoU cho NMS")
    parser.add_argument("--no-save", action="store_true", help="Không lưu ảnh kết quả")
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        print(f"[LỖI] Không tìm thấy models/best.pt\n  Đường dẫn: {weights}")
        print("Chạy: python setup_model.py")
        return 1

    image_path = get_test_image(args.image)
    if image_path is None:
        print("[LỖI] Không có ảnh test. Đặt 1 ảnh vào assets/ hoặc dùng: --image <duong_dan>")
        return 1

    print("=" * 60)
    print("KỂM TRA MODEL best.pt")
    print("=" * 60)
    print(f"Model : {weights}")
    print(f"Ảnh   : {image_path}")

    try:
        detector = WasteDetector(
            weights_path=weights,
            imgsz=args.imgsz,
            conf_thres=args.conf,
            iou_thres=args.iou,
        )
    except Exception as exc:
        print(f"[LỖI] Load model thất bại:\n{exc}")
        return 1

    print(f"Device: {detector.device}")
    print(f"Class names (từ checkpoint): {detector.class_names}")

    # Đọc ảnh bằng imdecode (an toàn với đường dẫn Unicode trên Windows)
    data = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        print(f"[LỖI] Không đọc được ảnh: {image_path} (file không phải ảnh hợp lệ?)")
        return 1

    # Thu nhỏ ảnh quá lớn để test nhanh hơn (demo prototype)
    h, w = image.shape[:2]
    max_side = max(h, w)
    if max_side > config.MAX_IMAGE_SIDE:
        scale = config.MAX_IMAGE_SIDE / max_side
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    annotated, detections, dt = detector.detect(image)

    print("-" * 60)
    print("Detected objects:")
    if detections:
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            print(
                f"  {det['class_name']:<15} {det['confidence'] * 100:6.1f}%"
                f"   bbox=[{x1},{y1},{x2},{y2}]"
            )
    else:
        print("  No object detected")
    print("-" * 60)
    print(f"Tổng số object : {len(detections)}")
    print(f"Inference time : {dt * 1000:.1f} ms  (~{1.0 / max(dt, 1e-6):.1f} FPS)")

    if not args.no_save:
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = config.OUTPUT_DIR / "test_result.jpg"
        cv2.imwrite(str(out_path), annotated)
        print(f"Ảnh kết quả    : {out_path}")

    print("=" * 60)
    print("Pipeline inference hoạt động OK. Chạy GUI bằng: python app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
