"""
setup_model.py — Tải tài nguyên TỪ CHÍNH repository gốc:
    https://github.com/has-bi/Plastic-Waste-Detection-YOLOv5s-Raspberry-Pi4

Script này KHÔNG train model, KHÔNG tạo model giả. Nó chỉ:
    1. Tải (hoặc dùng bản clone có sẵn) repository gốc
    2. Copy  best.pt            ->  models/best.pt
    3. Copy  models/ + utils/   ->  yolov5/   (framework YOLOv5 để load best.pt)
    4. Copy  data.yaml          ->  yolov5/data.yaml
    5. Tải 1-2 ảnh test mẫu vào assets/

Cách chạy:
    python setup_model.py                            # tải ZIP từ GitHub
    python setup_model.py --repo <duong_dan_clone>   # dùng bản clone có sẵn
    python setup_model.py --skip-samples             # không tải ảnh mẫu
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import config

ZIP_TOP_DIR = "Plastic-Waste-Detection-YOLOv5s-Raspberry-Pi4-main"


def _force_utf8_stdout() -> None:
    """Đảm bảo in được tiếng Việt trên Windows (mặc định console là cp1252)."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


_force_utf8_stdout()

# Thành phần cần thiết để LOAD & CHẠY INFERENCE best.pt (theo cách của repo gốc)
SINGLE_FILES = {
    "best.pt": config.BASE_DIR / "models" / "best.pt",
    "data.yaml": config.YOLOV5_DIR / "data.yaml",
    "classes.txt": config.YOLOV5_DIR / "classes.txt",
}
WANTED_DIRS = {
    "models": config.YOLOV5_DIR / "models",   # framework YOLOv5 (attempt_load...)
    "utils": config.YOLOV5_DIR / "utils",     # non_max_suppression, torch_utils...
}


def _print(msg: str) -> None:
    print(msg, flush=True)


def copy_from_local_repo(repo_dir: Path) -> None:
    """Copy best.pt + framework YOLOv5 từ một bản clone repository gốc có sẵn."""
    repo_dir = repo_dir.resolve()
    if not (repo_dir / "best.pt").exists():
        raise FileNotFoundError(f"Không thấy best.pt trong: {repo_dir}")

    for src_name, dest in SINGLE_FILES.items():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo_dir / src_name, dest)
        _print(f"[OK] Đã copy {src_name} -> {dest}")

    for dir_name, dest_dir in WANTED_DIRS.items():
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(
            repo_dir / dir_name, dest_dir,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        _print(f"[OK] Đã copy {dir_name}/ -> {dest_dir}")


def download_repo_zip(dest_zip: Path) -> None:
    _print(f"[INFO] Đang tải ZIP repository ({config.REPO_URL}) ...")
    req = urllib.request.Request(
        config.REPO_ZIP_URL, headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req, timeout=300) as r, open(dest_zip, "wb") as f:
        shutil.copyfileobj(r, f)
    size_mb = dest_zip.stat().st_size / 1024 / 1024
    _print(f"[OK] Đã tải ZIP ({size_mb:.1f} MB): {dest_zip}")


def extract_from_zip(zip_path: Path) -> None:
    """Chỉ extract best.pt, data.yaml, models/, utils/ (bỏ __pycache__/*.pyc)."""
    prefix = ZIP_TOP_DIR + "/"
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if not name.startswith(prefix):
                continue
            rel = name[len(prefix):]
            if not rel or rel.endswith("/"):
                continue
            if "__pycache__" in rel or rel.endswith(".pyc"):
                continue

            if rel in SINGLE_FILES:
                dest = SINGLE_FILES[rel]
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(name) as src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out)
                _print(f"[OK] Extract {rel} -> {dest}")
            else:
                for dir_name, dest_dir in WANTED_DIRS.items():
                    if rel.startswith(dir_name + "/"):
                        dest = dest_dir / rel[len(dir_name) + 1:]
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        with zf.open(name) as src, open(dest, "wb") as out:
                            shutil.copyfileobj(src, out)
                        break


def download_samples() -> None:
    config.ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    for url in config.SAMPLE_IMAGE_URLS:
        dest = config.ASSETS_DIR / Path(url).name
        if dest.exists() and dest.stat().st_size > 10_000:
            _print(f"[OK] Ảnh mẫu đã có: {dest}")
            continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
                shutil.copyfileobj(r, f)
            _print(f"[OK] Đã tải ảnh mẫu: {dest}")
        except Exception as exc:
            _print(f"[WARN] Không tải được ảnh mẫu {url}: {exc}")


def verify() -> bool:
    _print("-" * 60)
    ok = True
    if config.MODEL_PATH.exists():
        size_mb = config.MODEL_PATH.stat().st_size / 1024 / 1024
        _print(f"[OK] models/best.pt ({size_mb:.1f} MB)")
        if size_mb < 1.0:
            _print("[LỖI] best.pt quá nhỏ — có thể tải lỗi. Xoá file và chạy lại.")
            ok = False
    else:
        _print(f"[LỖI] Thiếu {config.MODEL_PATH}")
        ok = False

    for rel in ("yolov5/models/yolo.py", "yolov5/models/experimental.py",
                "yolov5/utils/general.py", "yolov5/utils/torch_utils.py",
                "yolov5/data.yaml"):
        p = config.BASE_DIR / rel
        if p.exists():
            _print(f"[OK] {rel}")
        else:
            _print(f"[LỖI] Thiếu {rel}")
            ok = False

    _print("-" * 60)
    if ok:
        _print("Hoàn tất! Giờ chạy:")
        _print("    python test_model.py     # kiểm tra inference trước")
        _print("    python app.py            # mở GUI demo")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Tải best.pt + framework YOLOv5 từ repo gốc")
    parser.add_argument("--repo", default=None,
                        help="Đường dẫn bản clone repository gốc (bỏ qua việc tải ZIP)")
    parser.add_argument("--skip-samples", action="store_true", help="Không tải ảnh mẫu")
    args = parser.parse_args()

    try:
        if args.repo:
            copy_from_local_repo(Path(args.repo))
        else:
            with tempfile.TemporaryDirectory() as tmp:
                zip_path = Path(tmp) / "repo.zip"
                download_repo_zip(zip_path)
                extract_from_zip(zip_path)
        if not args.skip_samples:
            download_samples()
        return 0 if verify() else 1
    except Exception as exc:
        _print(f"[LỖI] {exc}")
        _print("Fallback thủ công:")
        _print(f"  1. Clone:  git clone {config.REPO_URL}.git")
        _print("  2. Copy best.pt -> models/best.pt")
        _print("  3. Copy models/ và utils/ từ repo -> yolov5/")
        return 1


if __name__ == "__main__":
    sys.exit(main())

