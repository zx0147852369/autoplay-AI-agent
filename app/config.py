import os
import secrets
from datetime import timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
MEDIA_DIR = DATA_DIR / "media"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{(DATA_DIR / 'app.db').as_posix()}")

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))

# เวลาที่แสดงบนหน้าเว็บ (ค่าเริ่มต้น: เวลาประเทศไทย UTC+7)
DISPLAY_TZ = timezone(timedelta(hours=int(os.getenv("DISPLAY_TZ_OFFSET", "7"))))


def _load_secret_key() -> str:
    key = os.getenv("SECRET_KEY", "").strip()
    if key:
        return key
    # ไม่ได้ตั้ง SECRET_KEY ใน .env -> สร้างครั้งเดียวแล้วเก็บไว้ใน data/secret.key
    path = DATA_DIR / "secret.key"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    key = secrets.token_urlsafe(48)
    path.write_text(key, encoding="utf-8")
    return key


SECRET_KEY = _load_secret_key()
