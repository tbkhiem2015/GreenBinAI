"""Smoke test GUI: mở cửa sổ, chờ model load nền, in trạng thái rồi tự đóng."""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import tkinter as tk

import app

root = tk.Tk()
gui = app.PlasticWasteApp(root)
result = {}


def check():
    result["det"] = gui.detector is not None
    result["status"] = gui.lbl_status.cget("text")
    result["classes"] = gui.lbl_classes.cget("text")
    result["img_btn"] = str(gui.btn_image.cget("state"))
    root.destroy()


root.after(45000, check)
root.mainloop()

print("detector_ready:", result.get("det"))
print("status:", result.get("status"))
print("classes:", result.get("classes"))
print("btn_image_state:", result.get("img_btn"))
print("GUI SMOKE TEST:", "OK" if result.get("det") else "MODEL NOT READY YET")
