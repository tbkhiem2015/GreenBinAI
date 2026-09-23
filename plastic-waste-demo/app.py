"""
app.py — GUI demo kiểm thử model nhận diện rác thải
(best.pt từ has-bi/Plastic-Waste-Detection-YOLOv5s-Raspberry-Pi4)

Chế độ:
    Mode 1: Nhận diện ảnh       -> nút "Chọn ảnh"
    Mode 2: Nhận diện webcam    -> nút "Start Camera" / "Stop Camera"

Lưu ý kỹ thuật:
    * Model load MỘT LẦN khi app khởi động (thread nền, GUI không bị treo).
    * Webcam + inference chạy trong thread riêng; GUI nhận kết quả qua queue.
    * ImageTk.PhotoImage được tạo trong main thread (yêu cầu của Tkinter).
    * 100% không phụ thuộc Windows-only API — chạy được cả trên Raspberry Pi 4.
"""

from __future__ import annotations

import datetime
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

import config
import firebase_store
from detector import WasteDetector

# ---------------------------------------------------------------------------
# Giao diện dark theme
# ---------------------------------------------------------------------------
BG = "#0b1220"
PANEL = "#111a2e"
CARD = "#182745"
ACCENT = "#22c55e"
ACCENT_BLUE = "#38bdf8"
DANGER = "#ef4444"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"

CANVAS_W, CANVAS_H = 880, 520


class PlasticWasteApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("♻️ Plastic Waste Detection — YOLOv5s AI Demo")
        self.root.configure(bg=BG)
        self.root.geometry("1240x780")
        self.root.minsize(1080, 700)

        # --- Trạng thái ứng dụng -------------------------------------------------
        self.detector = None
        self.ui_queue = queue.Queue()
        self.camera_queue = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()
        self.cam_thread = None
        self.cam_running = False
        self.image_busy = False
        self.fps = 0.0
        self.last_annotated = None
        self._photo = None  # giữ tham chiếu ảnh Tk (tránh bị GC)
        self._last_status_text = ""

        # Trạng thái đẩy Firebase (chế độ webcam — throttle theo thay đổi kết quả)
        self._last_fb_signature = None
        self._last_fb_time = 0.0

        self._build_style()
        self._build_header()
        self._build_main_area()
        self._build_controls()
        self._build_statusbar()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(30, self._poll_queue)

        # Load model best.pt trong thread nền — GUI mở ngay, không đơ
        # (chốt tham số NGAY TRONG main thread, không đọc biến Tk từ thread phụ)
        imgsz_start = int(self.var_imgsz.get())
        conf_start = float(self.var_conf.get())
        self._set_status("Đang tải model best.pt ... (lần đầu có thể mất vài giây)")
        self._set_buttons(ready=False)
        threading.Thread(
            target=self._load_model_worker, args=(imgsz_start, conf_start), daemon=True
        ).start()

    # =========================================================================
    # XÂY DỰNG GIAO DIỆN
    # =========================================================================
    def _build_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(
            "Results.Treeview", background=PANEL, fieldbackground=PANEL,
            foreground=TEXT, rowheight=28, borderwidth=0, font=("Segoe UI", 10),
        )
        style.configure(
            "Results.Treeview.Heading", background=CARD, foreground=MUTED,
            font=("Segoe UI", 10, "bold"), borderwidth=0,
        )
        style.map("Results.Treeview", background=[("selected", CARD)])

    def _build_header(self) -> None:
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=20, pady=(16, 8))
        tk.Label(header, text="♻️ PLASTIC WASTE DETECTION",
                 bg=BG, fg=ACCENT, font=("Segoe UI", 22, "bold")).pack()
        tk.Label(
            header,
            text="YOLOv5s AI Demo  •  model: best.pt (has-bi/Plastic-Waste-Detection-YOLOv5s-Raspberry-Pi4)",
            bg=BG, fg=MUTED, font=("Segoe UI", 10),
        ).pack()

    def _build_main_area(self) -> None:
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True, padx=20, pady=8)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)

        # ---- Khung hiển thị ảnh / camera ------------------------------------
        canvas_wrap = tk.Frame(main, bg=CARD, highlightthickness=1,
                               highlightbackground="#24365e")
        canvas_wrap.grid(row=0, column=0, sticky="nsew")
        self.canvas = tk.Canvas(canvas_wrap, width=CANVAS_W, height=CANVAS_H,
                                bg="#050a14", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=8)
        self._draw_placeholder("Chưa có ảnh / camera\nHãy chọn ảnh hoặc Start Camera")

        # ---- Panel kết quả ----------------------------------------------------
        panel = tk.Frame(main, bg=PANEL)
        panel.grid(row=0, column=1, sticky="ns", padx=(12, 0))
        panel.configure(width=300)
        panel.pack_propagate(False)

        tk.Label(panel, text="Detection Results", bg=PANEL, fg=TEXT,
                 font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=12, pady=(12, 4))

        self.tree = ttk.Treeview(panel, columns=("class_name", "confidence"),
                                 show="headings", style="Results.Treeview", height=12)
        self.tree.heading("class_name", text="Class")
        self.tree.heading("confidence", text="Confidence")
        self.tree.column("class_name", width=150, anchor="w")
        self.tree.column("confidence", width=100, anchor="e")
        self.tree.pack(fill="x", padx=12, pady=6)

        self.lbl_objects = tk.Label(panel, text="Objects detected: 0", bg=PANEL, fg=TEXT,
                                    font=("Segoe UI", 11, "bold"), anchor="w")
        self.lbl_objects.pack(fill="x", padx=14, pady=(8, 0))

        self.lbl_fps = tk.Label(panel, text="FPS: —", bg=PANEL, fg=ACCENT_BLUE,
                                font=("Segoe UI", 11, "bold"), anchor="w")
        self.lbl_fps.pack(fill="x", padx=14, pady=(2, 6))

        self.lbl_classes = tk.Label(panel, text="Classes: (đang tải...)", bg=PANEL, fg=MUTED,
                                    font=("Segoe UI", 9), anchor="w", wraplength=270,
                                    justify="left")
        self.lbl_classes.pack(fill="x", padx=14, pady=(6, 12))

    def _build_controls(self) -> None:
        controls = tk.Frame(self.root, bg=PANEL)
        controls.pack(fill="x", padx=20, pady=(8, 4))

        btn_row = tk.Frame(controls, bg=PANEL)
        btn_row.pack(fill="x", padx=12, pady=(10, 2))

        self.btn_image = tk.Button(
            btn_row, text="📁  Chọn ảnh", command=self.choose_image,
            bg=ACCENT, fg="#04140a", activebackground="#16a34a", activeforeground="#04140a",
            font=("Segoe UI", 11, "bold"), relief="flat", cursor="hand2", padx=18, pady=6,
        )
        self.btn_image.pack(side="left", padx=(0, 10))

        self.btn_start = tk.Button(
            btn_row, text="📷  Start Camera", command=self.start_camera,
            bg=ACCENT_BLUE, fg="#06121f", activebackground="#0284c7", activeforeground="#06121f",
            font=("Segoe UI", 11, "bold"), relief="flat", cursor="hand2", padx=18, pady=6,
        )
        self.btn_start.pack(side="left", padx=(0, 10))

        self.btn_stop = tk.Button(
            btn_row, text="⏹  Stop Camera", command=self.stop_camera,
            bg=DANGER, fg="#fff", activebackground="#b91c1c", activeforeground="#fff",
            font=("Segoe UI", 11, "bold"), relief="flat", cursor="hand2", padx=18, pady=6,
            state="disabled",
        )
        self.btn_stop.pack(side="left", padx=(0, 10))

        self.btn_snapshot = tk.Button(
            btn_row, text="📸  Lưu khung hình", command=self.save_snapshot,
            bg=CARD, fg=TEXT, activebackground="#24365e", activeforeground=TEXT,
            font=("Segoe UI", 11), relief="flat", cursor="hand2", padx=14, pady=6,
            state="disabled",
        )
        self.btn_snapshot.pack(side="left")

        # ---- Hàng tham số ------------------------------------------------------
        params = tk.Frame(controls, bg=PANEL)
        params.pack(fill="x", padx=12, pady=(2, 10))

        tk.Label(params, text="Confidence:", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 10)).pack(side="left")
        self.var_conf = tk.DoubleVar(value=config.DEFAULT_CONF_THRES)
        self.scale_conf = tk.Scale(
            params, from_=0.05, to=0.95, resolution=0.05, orient="horizontal",
            variable=self.var_conf, bg=PANEL, fg=TEXT, troughcolor=CARD,
            highlightthickness=0, length=160, command=self._on_conf_change,
            font=("Segoe UI", 9),
        )
        self.scale_conf.pack(side="left", padx=(4, 18))

        tk.Label(params, text="Input size:", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 10)).pack(side="left")
        self.var_imgsz = tk.StringVar(value=str(config.DEFAULT_IMGSZ))
        self.cmb_imgsz = ttk.Combobox(
            params, textvariable=self.var_imgsz, values=["320", "480", "640"],
            width=5, state="readonly", font=("Segoe UI", 10),
        )
        self.cmb_imgsz.pack(side="left", padx=(4, 18))
        self.cmb_imgsz.bind("<<ComboboxSelected>>", self._on_imgsz_change)

        tk.Label(params, text="Infer mỗi N khung (Pi):", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 10)).pack(side="left")
        self.var_interval = tk.IntVar(value=1)
        self.scale_interval = tk.Scale(
            params, from_=1, to=5, resolution=1, orient="horizontal",
            variable=self.var_interval, bg=PANEL, fg=TEXT, troughcolor=CARD,
            highlightthickness=0, length=120, font=("Segoe UI", 9),
        )
        self.scale_interval.pack(side="left", padx=(4, 0))

    def _build_statusbar(self) -> None:
        bar = tk.Frame(self.root, bg=CARD)
        bar.pack(fill="x", side="bottom")
        self.lbl_status = tk.Label(bar, text="Khởi động...", bg=CARD, fg=MUTED,
                                   font=("Segoe UI", 9), anchor="w")
        self.lbl_status.pack(fill="x", padx=14, pady=4)

    # =========================================================================
    # LOAD MODEL (một lần duy nhất, thread nền)
    # =========================================================================
    def _load_model_worker(self, imgsz_start: int, conf_start: float) -> None:
        try:
            det = WasteDetector(
                imgsz=imgsz_start,
                conf_thres=conf_start,
                iou_thres=config.DEFAULT_IOU_THRES,
            )
            self.detector = det
            self.ui_queue.put(("model_ready", str(det.device), det.class_names))
        except Exception as exc:
            self.ui_queue.put(("model_error", str(exc)))

    # =========================================================================
    # MODE 1 — NHẬN DIỆN ẢNH
    # =========================================================================
    def choose_image(self) -> None:
        if self.detector is None:
            messagebox.showwarning("Chưa sẵn sàng", "Model đang tải, vui lòng đợi...")
            return
        if self.cam_running:
            messagebox.showinfo("Camera đang chạy", "Hãy bấm Stop Camera trước khi chọn ảnh.")
            return
        path = filedialog.askopenfilename(
            title="Chọn ảnh để nhận diện",
            filetypes=[("Ảnh", "*.jpg *.jpeg *.png *.webp"), ("Tất cả", "*.*")],
        )
        if not path:
            return
        self.image_busy = True
        self._set_buttons(ready=True, busy=True)
        self._set_status(f"Đang nhận diện: {path}")
        threading.Thread(target=self._image_worker, args=(path,), daemon=True).start()

    def _image_worker(self, path: str) -> None:
        try:
            # Đọc ảnh bằng imdecode — hỗ trợ đường dẫn Unicode trên Windows
            data = np.fromfile(path, dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if image is None:
                self.ui_queue.put(("image_error", "Ảnh không hợp lệ hoặc không đọc được.\n"
                                              "Hãy chọn file .jpg/.jpeg/.png/.webp"))
                return

            h, w = image.shape[:2]
            max_side = max(h, w)
            if max_side > config.MAX_IMAGE_SIDE:
                scale = config.MAX_IMAGE_SIDE / max_side
                image = cv2.resize(image, (int(w * scale), int(h * scale)),
                                   interpolation=cv2.INTER_AREA)

            annotated, detections, dt = self.detector.detect(image)

            # Lưu kết quả vào outputs/ (imencode để an toàn đường dẫn Unicode)
            config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = config.OUTPUT_DIR / f"annotated_{stamp}.jpg"
            ok, buf = cv2.imencode(".jpg", annotated)
            if ok:
                buf.tofile(str(out_path))

            # Đẩy kết quả lên Firebase (Thời gian, plastic/ko, class, confidence)
            if config.FIREBASE_ENABLED:
                record = firebase_store.build_record(
                    "image", detections, device=str(self.detector.device)
                )
                fb_ok, fb_msg = firebase_store.push_detection(record)
                self.ui_queue.put(("firebase_status", fb_ok, fb_msg))

            self.ui_queue.put(("image_done", annotated, detections, dt, str(out_path)))
        except Exception as exc:
            self.ui_queue.put(("image_error", f"Lỗi khi nhận diện ảnh:\n{exc}"))

    # =========================================================================
    # MODE 2 — NHẬN DIỆN WEBCAM (thread riêng, GUI không bị treo)
    # =========================================================================
    def start_camera(self) -> None:
        if self.detector is None:
            messagebox.showwarning("Chưa sẵn sàng", "Model đang tải, vui lòng đợi...")
            return
        if self.cam_running:
            return
        self.stop_event.clear()
        self.cam_running = True
        self._set_buttons(ready=True, cam=True)
        self._set_status("Đang mở webcam ...")
        self.cam_thread = threading.Thread(
            target=self._camera_worker, args=(config.WEBCAM_INDEX,), daemon=True
        )
        self.cam_thread.start()

    def stop_camera(self) -> None:
        if not self.cam_running:
            return
        self.stop_event.set()
        self._set_status("Đang dừng camera ...")

    def _camera_worker(self, cam_index: int) -> None:
        cap = None
        try:
            cap = cv2.VideoCapture(cam_index)  # backend mặc định (Windows/Pi OK)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.WEBCAM_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.WEBCAM_HEIGHT)
            if not cap.isOpened():
                self.ui_queue.put((
                    "camera_error",
                    "Không thể mở webcam.\nHãy kiểm tra camera của bạn "
                    "(hoặc đổi WEBCAM_INDEX trong config.py).",
                ))
                return

            self.ui_queue.put(("camera_started",))
            interval = max(1, int(self.var_interval.get()))
            frame_idx = 0
            fps_ema = 0.0
            last_detections = []

            while not self.stop_event.is_set():
                t_loop = time.perf_counter()
                ok, frame = cap.read()
                if not ok or frame is None:
                    self.ui_queue.put(("camera_error", "Mất kết nối webcam."))
                    break

                if frame_idx % interval == 0:
                    # Chỉ chạy inference đúng tần suất (tiết kiệm CPU trên Pi 4)
                    annotated, detections, _ = self.detector.detect(frame)
                    last_detections = detections
                    self._firebase_push_webcam(detections)
                else:
                    annotated = WasteDetector._draw(frame.copy(), last_detections)
                    detections = last_detections

                dt_loop = time.perf_counter() - t_loop
                inst_fps = 1.0 / max(dt_loop, 1e-6)
                fps_ema = inst_fps if fps_ema == 0 else 0.9 * fps_ema + 0.1 * inst_fps

                # Đưa frame vào queue (drop-oldest) — không ghi đĩa, không tràn RAM
                try:
                    self.camera_queue.put_nowait((annotated, detections, fps_ema))
                except queue.Full:
                    try:
                        self.camera_queue.get_nowait()
                    except queue.Empty:
                        pass
                frame_idx += 1
        except Exception as exc:
            self.ui_queue.put(("camera_error", f"Lỗi webcam:\n{exc}"))
        finally:
            if cap is not None:
                cap.release()
            self.ui_queue.put(("camera_stopped",))

    def _firebase_push_webcam(self, detections) -> None:
        """Đẩy kết quả webcam lên Firebase — CHỈ khi kết quả thay đổi và
        cách lần đẩy trước ít nhất WEBCAM_FIREBASE_MIN_INTERVAL_SEC giây
        (tránh spam hàng trăm bản ghi/phút)."""
        if not config.FIREBASE_ENABLED:
            return
        try:
            top = detections[0] if detections else None
            signature = (top["class_name"], round(top["confidence"], 2)) if top else None
            now_t = time.time()
            if (
                signature != self._last_fb_signature
                and now_t - self._last_fb_time >= config.WEBCAM_FIREBASE_MIN_INTERVAL_SEC
            ):
                record = firebase_store.build_record(
                    "webcam", detections, device=str(self.detector.device)
                )
                fb_ok, _fb_msg = firebase_store.push_detection(record)
                if fb_ok:
                    self._last_fb_signature = signature
                    self._last_fb_time = now_t
        except Exception:
            pass  # lỗi Firebase không được làm crash luồng camera

    # =========================================================================
    # VÒNG LẶP GUI — nhận kết quả từ thread nền (thread-safe)
    # =========================================================================
    def _poll_queue(self) -> None:
        # --- Message điều khiển -------------------------------------------------
        try:
            while True:
                msg = self.ui_queue.get_nowait()
                kind = msg[0]
                if kind == "model_ready":
                    self._on_model_ready(msg[1], msg[2])
                elif kind == "model_error":
                    self._on_model_error(msg[1])
                elif kind == "image_done":
                    annotated, detections, dt, out_path = msg[1], msg[2], msg[3], msg[4]
                    self._show_bgr(annotated)
                    self._update_results(detections, fps=None, dt=dt)
                    self.last_annotated = annotated
                    self.image_busy = False
                    self._set_buttons(ready=True)
                    self._set_status(f"Hoàn tất — đã lưu: {out_path}")
                elif kind == "image_error":
                    self.image_busy = False
                    self._set_buttons(ready=True)
                    messagebox.showerror("Lỗi nhận diện ảnh", msg[1])
                    self._set_status("Lỗi khi nhận diện ảnh.")
                elif kind == "firebase_status":
                    tag = "🔥 Firebase: đã lưu" if msg[1] else f"⚠ Firebase: {msg[2]}"
                    self._set_status(f"{self._last_status_text} • {tag}")
                elif kind == "camera_started":
                    self.btn_stop.config(state="normal")
                    self.btn_start.config(state="disabled")
                    self.btn_image.config(state="disabled")
                    self.btn_snapshot.config(state="normal")
                    self._set_status("Camera đang chạy — inference realtime bằng best.pt")
                elif kind == "camera_error":
                    messagebox.showerror("Lỗi webcam", msg[1])
                    self._finalize_camera_stop(error=True)
                elif kind == "camera_stopped":
                    self._finalize_camera_stop()
        except queue.Empty:
            pass

        # --- Frame webcam mới nhất ----------------------------------------------
        latest = None
        try:
            while True:
                latest = self.camera_queue.get_nowait()
        except queue.Empty:
            pass
        if latest is not None and self.cam_running:
            annotated, detections, fps = latest
            self._show_bgr(annotated)
            self.last_annotated = annotated
            self.fps = fps
            self._update_results(detections, fps=fps)

        self.root.after(30, self._poll_queue)

    def _on_model_ready(self, device: str, class_names) -> None:
        self.lbl_classes.config(text="Classes: " + ", ".join(class_names))
        self._set_buttons(ready=True)
        self._set_status(f"Model sẵn sàng  •  device: {device}  •  "
                         f"{len(class_names)} classes  •  best.pt đã load 1 lần")

    def _on_model_error(self, message: str) -> None:
        self._set_buttons(ready=False)
        self._set_status("LỖI model — xem chi tiết trong hộp thoại")
        messagebox.showerror("Không thể tải model best.pt", message)

    # =========================================================================
    # CÁC HÀM HIỂN THỊ / TIỆN ÍCH
    # =========================================================================
    def _show_bgr(self, frame_bgr) -> None:
        """Vẽ 1 khung hình BGR lên canvas (tạo ImageTk trong MAIN thread)."""
        if frame_bgr is None:
            return
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = min(CANVAS_W / w, CANVAS_H / h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        pil = Image.fromarray(rgb).resize((new_w, new_h), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        x = (CANVAS_W - new_w) // 2
        y = (CANVAS_H - new_h) // 2
        self.canvas.create_image(x, y, image=self._photo, anchor="nw")

    def _draw_placeholder(self, text: str) -> None:
        self.canvas.delete("all")
        cx, cy = CANVAS_W // 2, CANVAS_H // 2
        self.canvas.create_text(cx, cy - 20, text="♻️", font=("Segoe UI Emoji", 64),
                                fill="#1e3a5f")
        self.canvas.create_text(cx, cy + 70, text=text, font=("Segoe UI", 13),
                                fill=MUTED, justify="center")

    def _update_results(self, detections, fps=None, dt=None) -> None:
        self.tree.delete(*self.tree.get_children())
        for det in detections:
            self.tree.insert("", "end", values=(
                det["class_name"], f"{det['confidence'] * 100:.1f}%"))
        self.lbl_objects.config(text=f"Objects detected: {len(detections)}")
        if fps is not None:
            self.lbl_fps.config(text=f"FPS: {fps:.1f}")
        elif dt is not None:
            self.lbl_fps.config(text=f"FPS: —  (inference {dt * 1000:.0f} ms)")

    def _set_buttons(self, ready: bool, cam: bool = False, busy: bool = False) -> None:
        state_on = "normal" if (ready and not busy) else "disabled"
        self.btn_image.config(state=state_on)
        self.btn_start.config(state=state_on)
        if not cam:
            self.btn_stop.config(state="disabled")
            self.btn_snapshot.config(state="disabled")

    def _finalize_camera_stop(self, error: bool = False) -> None:
        self.cam_running = False
        self.stop_event.clear()
        self._set_buttons(ready=self.detector is not None)
        self.btn_stop.config(state="disabled")
        self.btn_snapshot.config(state="disabled")
        self.btn_start.config(state="normal")
        self.btn_image.config(state="normal")
        if error:
            self._set_status("Camera đã dừng do lỗi. Kiểm tra webcam rồi thử lại.")
        else:
            self._set_status("Camera đã dừng. Chọn ảnh hoặc Start Camera để tiếp tục.")

    def save_snapshot(self) -> None:
        if self.last_annotated is None:
            messagebox.showinfo("Chưa có gì để lưu", "Chưa có khung hình nào được nhận diện.")
            return
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = config.OUTPUT_DIR / f"webcam_{stamp}.jpg"
        ok, buf = cv2.imencode(".jpg", self.last_annotated)
        if ok:
            buf.tofile(str(out_path))
            self._set_status(f"Đã lưu khung hình: {out_path}")

    def _set_status(self, text: str) -> None:
        self._last_status_text = text
        self.lbl_status.config(text=text)

    def _on_conf_change(self, _value=None) -> None:
        if self.detector is not None:
            self.detector.set_thresholds(self.var_conf.get(), config.DEFAULT_IOU_THRES)

    def _on_imgsz_change(self, _event=None) -> None:
        if self.detector is not None:
            self.detector.set_imgsz(int(self.var_imgsz.get()))
            self._set_status(f"Input size mới: {self.var_imgsz.get()} (áp dụng từ lần inference kế)")

    def _on_close(self) -> None:
        try:
            if self.cam_running:
                self.stop_event.set()
                if self.cam_thread is not None:
                    self.cam_thread.join(timeout=2.0)
        finally:
            self.root.destroy()


def main() -> None:
    root = tk.Tk()
    PlasticWasteApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()






