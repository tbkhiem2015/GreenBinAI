"""_push_real_test.py — Test đẩy 1 bản ghi THẬT lên Firebase của người dùng.

Bản ghi dùng detection THẬT (plastic bottle 58.0%) từ lần chạy test_model.py
trước đó — không phải dữ liệu giả. Sau khi đẩy, GET lại node để xác nhận.
"""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import config
import firebase_store


def main():
    print("=" * 60)
    print("TEST PUSH LEN FIREBASE THUC")
    print(f"URL : {config.FIREBASE_DB_URL}")
    print(f"Node: {config.FIREBASE_NODE}")
    print("=" * 60)

    # Detection thật từ lần test best.pt trước đó (Empty_Plastic_Bottle.jpg)
    detections = [{"class_name": "plastic bottle", "confidence": 0.58,
                   "bbox": [4, 0, 565, 416]}]

    record = firebase_store.build_record("image", detections, device="cpu")
    ok, msg = firebase_store.push_detection(record)
    print(f"Push : ok={ok}  ({msg.strip()})")

    if not ok:
        print("[LOI] Khong push duoc. Kiem tra rules trong Firebase Console.")
        return 1

    # GET lại node để xác nhận dữ liệu nằm trên DB
    import requests
    url = f"{config.FIREBASE_DB_URL.rstrip('/')}/{config.FIREBASE_NODE}.json"
    params = {"auth": config.FIREBASE_DB_SECRET} if config.FIREBASE_DB_SECRET else None
    resp = requests.get(url, params=params, timeout=15)
    data = resp.json()
    count = len(data) if isinstance(data, dict) else 0
    print(f"GET  : HTTP {resp.status_code} — node 'detections/' hiện có {count} bản ghi")

    if isinstance(data, dict) and data:
        latest_key = list(data.keys())[-1]
        latest = data[latest_key]
        print(f"Key mới nhất : {latest_key}")
        print(f"  timestamp  : {latest.get('timestamp')}")
        print(f"  is_plastic : {latest.get('is_plastic')}")
        print(f"  class_name : {latest.get('class_name')}")
        print(f"  confidence : {latest.get('confidence_percent')}%")
    print("=" * 60)
    print("OK! Mo Firebase Console -> Realtime Database de thay node detections/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
