"""_test_firebase.py — Verify module firebase_store bằng server mock cục bộ.

Test phần TRUYỀN DỮ LIỆU + logic phân loại plastic (KHÔNG dùng detection giả
làm kết quả model — detection ở đây chỉ là dữ liệu test cho tầng upload).
Chạy: python _test_firebase.py
"""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import config
import firebase_store

RECEIVED = []


class MockRTDBHandler(BaseHTTPRequestHandler):
    """Giả lập endpoint POST của Realtime Database trên localhost."""

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        RECEIVED.append((self.path, json.loads(self.rfile.read(length))))
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"name":"-mockKey123"}')

    def log_message(self, *args):
        pass


def main():
    server = HTTPServer(("127.0.0.1", 8791), MockRTDBHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    config.FIREBASE_ENABLED = True
    config.FIREBASE_DB_URL = "http://127.0.0.1:8791"
    config.FIREBASE_NODE = "detections"

    print("=" * 60)
    print("TEST firebase_store.py (server mock cục bộ)")
    print("=" * 60)

    # --- Case 1: 1 chai nhựa (detection thật từ lần test trước) ---------------
    dets_bottle = [{"class_name": "plastic bottle", "confidence": 0.58,
                    "bbox": [4, 0, 565, 416]}]
    rec1 = firebase_store.build_record("image", dets_bottle, device="cpu")
    ok1, msg1 = firebase_store.push_detection(rec1)
    print(f"[1] chai nhựa    -> push ok={ok1}  is_plastic={rec1['is_plastic']}  "
          f"class={rec1['class_name']}  conf={rec1['confidence_percent']}%")

    # --- Case 2: dây cáp -> KHÔNG phải plastic --------------------------------
    dets_cable = [{"class_name": "cable", "confidence": 0.81, "bbox": [10, 10, 200, 300]}]
    rec2 = firebase_store.build_record("webcam", dets_cable, device="cpu")
    ok2, _ = firebase_store.push_detection(rec2)
    print(f"[2] cable        -> push ok={ok2}  is_plastic={rec2['is_plastic']}  "
          f"class={rec2['class_name']}")

    # --- Case 3: không phát hiện gì -------------------------------------------
    rec3 = firebase_store.build_record("image", [], device="cpu")
    ok3, _ = firebase_store.push_detection(rec3)
    print(f"[3] trống        -> push ok={ok3}  is_plastic={rec3['is_plastic']}  "
          f"class={rec3['class_name']}  num_objects={rec3['num_objects']}")

    server.shutdown()

    # --- Kiểm tra dữ liệu server nhận được -------------------------------------
    print("-" * 60)
    assert len(RECEIVED) == 3, f"Server nhận {len(RECEIVED)}/3 bản ghi!"
    for path, body in RECEIVED:
        assert path == "/detections.json", f"Sai path: {path}"
        for field in ("timestamp", "epoch", "source", "is_plastic",
                      "class_name", "confidence", "confidence_percent", "num_objects"):
            assert field in body, f"Thiếu trường {field}"
    assert RECEIVED[0][1]["is_plastic"] is True
    assert RECEIVED[1][1]["is_plastic"] is False
    assert RECEIVED[2][1]["is_plastic"] is False and RECEIVED[2][1]["class_name"] is None

    print("Bản ghi server nhận được (case 1):")
    print(json.dumps(RECEIVED[0][1], indent=2, ensure_ascii=False))
    print("=" * 60)
    print("FIREBASE STORE TEST: OK — mọi trường đúng, phân loại plastic đúng")
    return 0


if __name__ == "__main__":
    sys.exit(main())
