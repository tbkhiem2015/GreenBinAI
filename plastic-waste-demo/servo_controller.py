"""servo_controller.py — Điều khiển servo SG92R báo hiệu kết quả phân loại.

Đấu dây: servo SG92R -> GPIO17 (BCM numbering) = PIN vật lý 11 trên
Raspberry Pi 4 (xem SERVO_GPIO_PIN trong config.py).

Hành vi:
    * Kết quả là "plastic"      -> quay PHẢI (+SERVO_ANGLE_OFFSET độ),
      giữ SERVO_HOLD_SECONDS giây, rồi về lại vị trí trung tâm.
    * Kết quả KHÔNG phải plastic -> quay TRÁI (-SERVO_ANGLE_OFFSET độ),
      tương tự rồi về lại vị trí trung tâm.

Mỗi lần trigger chạy trên 1 thread nền riêng (không chặn GUI/luồng nhận
diện). Nếu servo đang quay dở, lần trigger mới bị bỏ qua thay vì xếp hàng.
Trên máy không phải Raspberry Pi (không có RPi.GPIO) module tự vô hiệu hoá.
"""

from __future__ import annotations

import threading
import time

import config

try:
    import RPi.GPIO as GPIO
    _GPIO_AVAILABLE = True
except (ImportError, RuntimeError):
    GPIO = None
    _GPIO_AVAILABLE = False

_busy_lock = threading.Lock()
_pwm = None
_initialized = False


def _angle_to_duty(angle: float) -> float:
    """Góc servo (0-180°) -> duty cycle (%) cho PWM 50Hz kiểu SG92R/SG90."""
    angle = max(0.0, min(180.0, angle))
    return 2.5 + (angle / 180.0) * 10.0


def init() -> bool:
    """Khởi tạo GPIO + PWM, đưa servo về vị trí trung tâm. Gọi 1 lần lúc app khởi động."""
    global _pwm, _initialized
    if not config.SERVO_ENABLED or not _GPIO_AVAILABLE:
        return False
    if _initialized:
        return True
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(config.SERVO_GPIO_PIN, GPIO.OUT)
    _pwm = GPIO.PWM(config.SERVO_GPIO_PIN, config.SERVO_PWM_FREQ_HZ)
    _pwm.start(_angle_to_duty(config.SERVO_ANGLE_CENTER))
    _initialized = True
    return True


def _move_to(angle: float) -> None:
    _pwm.ChangeDutyCycle(_angle_to_duty(angle))


def _sweep(target_angle: float) -> None:
    if not _busy_lock.acquire(blocking=False):
        return  # servo đang quay dở -> bỏ qua lần trigger này
    try:
        _move_to(target_angle)
        time.sleep(config.SERVO_HOLD_SECONDS)
        _move_to(config.SERVO_ANGLE_CENTER)
    finally:
        _busy_lock.release()


def _trigger_async(target_angle: float) -> None:
    if not config.SERVO_ENABLED or not _GPIO_AVAILABLE or not _initialized:
        return
    threading.Thread(target=_sweep, args=(target_angle,), daemon=True).start()


def trigger_plastic() -> None:
    """Báo hiệu rác NHỰA — quay phải, giữ, rồi về vị trí cũ."""
    _trigger_async(config.SERVO_ANGLE_CENTER + config.SERVO_ANGLE_OFFSET)


def trigger_non_plastic() -> None:
    """Báo hiệu rác KHÔNG PHẢI nhựa — quay trái, giữ, rồi về vị trí cũ."""
    _trigger_async(config.SERVO_ANGLE_CENTER - config.SERVO_ANGLE_OFFSET)


def cleanup() -> None:
    """Dừng PWM + giải phóng GPIO. Gọi khi đóng app."""
    global _pwm, _initialized
    if not _GPIO_AVAILABLE or not _initialized:
        return
    if _pwm is not None:
        _pwm.stop()
        _pwm = None
    GPIO.cleanup(config.SERVO_GPIO_PIN)
    _initialized = False
