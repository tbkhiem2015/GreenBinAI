#!/bin/bash
# run_pi.sh — Khởi động GUI Plastic Waste Detection trên Raspberry Pi 4.
# Dùng làm lệnh Exec cho autostart (~/.config/autostart) — xem README.md
# mục "Tự khởi động cùng Raspberry Pi 4 (autostart)".
cd "$(dirname "$0")"

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

export OPENBLAS_CORETYPE=ARMV8
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

mkdir -p outputs
LOG_FILE="outputs/run.log"

# Tự khởi động lại nếu app crash (exit code khác 0); dừng hẳn nếu người dùng
# đóng cửa sổ bình thường (exit code 0) — tránh vòng lặp khi đóng app chủ động.
while true; do
    python app.py >> "$LOG_FILE" 2>&1
    code=$?
    [ "$code" -eq 0 ] && break
    echo "$(date '+%Y-%m-%d %H:%M:%S'): app.py thoát với mã lỗi $code, khởi động lại sau 3s..." >> "$LOG_FILE"
    sleep 3
done
