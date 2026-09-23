@echo off
rem ============================================================
rem  run_gui.bat — Mở GUI Plastic Waste Detection (double-click)
rem  Tự dùng venv nếu có, không có thì dùng Python hệ thống.
rem ============================================================
cd /d "%~dp0"

if exist "venv\Scripts\python.exe" (
    echo [INFO] Dung virtual environment: venv\
    "venv\Scripts\python.exe" app.py
) else (
    echo [INFO] Khong thay venv — dung Python he thong
    python app.py
)

if errorlevel 1 (
    echo.
    echo [LOI] App ket thuc voi loi. Doc thong bao tren de xu ly.
)
pause
