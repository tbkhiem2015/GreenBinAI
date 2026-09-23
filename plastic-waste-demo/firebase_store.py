"""
firebase_store.py — Đẩy kết quả detection lên Firebase Realtime Database.

Dữ liệu mỗi bản ghi (đọc từ GUI: app.py, hoặc script tuỳ ý):

    {
      "timestamp": "2026-09-20 09:37:15",
      "epoch": 1758339435.123,
      "source": "image" | "webcam",
      "is_plastic": true,                  # plastic / ko plastic
      "class_name": "plastic bottle",      # null nếu không phát hiện gì
      "confidence": 0.58,
      "confidence_percent": 58.0,
      "num_objects": 1,
      "all_detections": [ {"class_name":..., "confidence":..., "bbox":[...]}, ... ]
    }

Kích hoạt: điền FIREBASE_DB_URL (và FIREBASE_DB_SECRET nếu rules cần auth)
trong config.py rồi đặt FIREBASE_ENABLED = True.
Không cần thêm dependency — dùng requests (đã có sẵn).
"""

from __future__ import annotations

import datetime
import time
from typing import Dict, List, Optional, Tuple

import requests

import config


def is_plastic(class_name: Optional[str]) -> bool:
    """Class này có phải rác nhựa không? (theo PLASTIC_CLASSES trong config.py)"""
    return class_name in config.PLASTIC_CLASSES


def build_record(source: str, detections: List[Dict], device: str = "") -> Dict:
    """Tạo 1 bản ghi kết quả từ danh sách detection (của ảnh hoặc 1 khung webcam).

    detection đầu tiên trong danh sách là vật điểm số cao nhất -> dùng làm
    class/confidence chính của bản ghi.
    """
    top = detections[0] if detections else None
    now = datetime.datetime.now()
    record = {
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "epoch": round(time.time(), 3),
        "source": source,  # "image" | "webcam"
        "is_plastic": bool(top) and is_plastic(top["class_name"]),
        "class_name": top["class_name"] if top else None,
        "confidence": round(top["confidence"], 4) if top else 0.0,
        "confidence_percent": round(top["confidence"] * 100, 1) if top else 0.0,
        "num_objects": len(detections),
        "all_detections": [
            {
                "class_name": d["class_name"],
                "confidence": round(d["confidence"], 4),
                "bbox": d["bbox"],
            }
            for d in detections
        ],
    }
    if device:
        record["device"] = device
    return record


def push_detection(record: Dict) -> Tuple[bool, str]:
    """Đẩy 1 bản ghi lên Firebase Realtime Database (POST -> key tự sinh).

    Trả về: (thành_công, mô_tả)
    """
    if not config.FIREBASE_ENABLED or not config.FIREBASE_DB_URL:
        return False, "Firebase chưa cấu hình (FIREBASE_ENABLED / FIREBASE_DB_URL trong config.py)"
    try:
        url = f"{config.FIREBASE_DB_URL.rstrip('/')}/{config.FIREBASE_NODE}.json"
        params = {"auth": config.FIREBASE_DB_SECRET} if config.FIREBASE_DB_SECRET else None
        resp = requests.post(url, json=record, params=params, timeout=10)
        if resp.status_code == 200:
            return True, f"Đã lưu ({record.get('class_name')}) "
        return False, f"HTTP {resp.status_code}: {resp.text[:120]}"
    except Exception as exc:
        return False, f"Lỗi kết nối: {exc}"
