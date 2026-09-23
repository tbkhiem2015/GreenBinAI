@echo off
rem ============================================================
rem  run_test.bat — Test inference model bằng CLI (không mở GUI)
rem  In detection ra terminal + lưu ảnh vào outputs\
rem ============================================================
cd /d "%~dp0"

if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" test_model.py %*
) else (
    python test_model.py %*
)
pause
