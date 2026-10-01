"""นับการเรียก AI และคำนวณโควตาที่เหลือ

Gemini API ไม่บอกโควตาที่เหลือมากับคำตอบ ระบบจึงนับเองจากทุกครั้งที่เรียก แล้วเทียบกับเพดานที่ตั้งไว้
(ตัวเลขจริงดูได้ที่ https://aistudio.google.com/rate-limit) · โควตารายวันของ Google รีเซ็ตเที่ยงคืนเวลาแปซิฟิก
"""

import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from sqlalchemy import func, select

from .database import AiUsage, SessionLocal, get_settings, utcnow

PACIFIC = ZoneInfo("America/Los_Angeles")

# ค่าเริ่มต้นจากข้อมูลโควตาฟรี (ก.ย. 2026) แก้ได้ในหน้าตั้งค่าให้ตรงกับ AI Studio · 0 = ไม่จำกัด
DEFAULT_LIMITS = {
    "gemini-3.8-flash": {"rpm": 5, "rpd": 20},
    "gemini-3.5-flash-lite": {"rpm": 15, "rpd": 500},
    "gemini-2.5-flash": {"rpm": 10, "rpd": 250},
}


def limits(settings: dict[str, str]) -> dict[str, dict[str, int]]:
    result = {m: dict(v) for m, v in DEFAULT_LIMITS.items()}
    try:
        saved = json.loads(settings.get("gemini_limits") or "{}")
    except json.JSONDecodeError:
        saved = {}
    for model, values in saved.items():
        if model in result and isinstance(values, dict):
            for key in ("rpm", "rpd"):
                try:
                    result[model][key] = max(0, int(values.get(key, result[model][key])))
                except (TypeError, ValueError):
                    pass
    return result


def day_start_utc(now: datetime | None = None) -> datetime:
    """เที่ยงคืนเวลาแปซิฟิกของวันนี้ (เวลาที่ Google รีเซ็ตโควตารายวัน) เป็น UTC แบบ naive"""
    now = (now or utcnow()).replace(tzinfo=timezone.utc).astimezone(PACIFIC)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return start.astimezone(timezone.utc).replace(tzinfo=None)


def next_reset_utc(now: datetime | None = None) -> datetime:
    now = (now or utcnow()).replace(tzinfo=timezone.utc).astimezone(PACIFIC)
    tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return tomorrow.astimezone(timezone.utc).replace(tzinfo=None)


def record(model: str, ok: bool, code: str = "ok", input_tokens: int = 0, output_tokens: int = 0) -> None:
    with SessionLocal() as db:
        row = AiUsage(model=model, ok=ok, code=code, input_tokens=input_tokens or 0, output_tokens=output_tokens or 0)
        db.add(row)
        db.flush()
        if row.id % 500 == 0:  # ล้างประวัติเก่ากว่า 30 วันเป็นครั้งคราว
            db.query(AiUsage).filter(AiUsage.at < utcnow() - timedelta(days=30)).delete()
        db.commit()


def _counts(db, model: str, since: datetime, ok_only: bool) -> int:
    query = select(func.count(AiUsage.id)).where(AiUsage.model == model, AiUsage.at >= since)
    if ok_only:
        query = query.where(AiUsage.ok.is_(True))
    return db.scalar(query) or 0


def available(model: str) -> tuple[bool, str]:
    """ยังเรียกรุ่นนี้ได้ไหม (ตามที่ระบบนับเอง) คืนค่า (ได้/ไม่ได้, เหตุผล)"""
    with SessionLocal() as db:
        lim = limits(get_settings(db)).get(model)
        if not lim:
            return True, ""
        now = utcnow()
        if lim["rpd"] and _counts(db, model, day_start_utc(now), ok_only=True) >= lim["rpd"]:
            return False, f"ใช้ครบ {lim['rpd']} ครั้งของวันนี้แล้ว"
        if lim["rpm"] and _counts(db, model, now - timedelta(seconds=60), ok_only=False) >= lim["rpm"]:
            return False, f"ครบ {lim['rpm']} ครั้งต่อนาทีแล้ว"
        last_429 = db.scalar(select(func.max(AiUsage.at)).where(AiUsage.model == model, AiUsage.code == "429"))
        if last_429 and now - last_429 < timedelta(seconds=60):
            return False, "Google แจ้งโควตาเต็มเมื่อไม่ถึง 1 นาทีที่แล้ว"
    return True, ""


def snapshot(settings: dict[str, str]) -> dict:
    """ข้อมูลสำหรับหลอดโควตาในหน้าเว็บ"""
    now = utcnow()
    start = day_start_utc(now)
    reset = next_reset_utc(now)
    lims = limits(settings)
    selected = settings.get("ai_model", "")
    rows = []
    with SessionLocal() as db:
        for model, lim in lims.items():
            used = _counts(db, model, start, ok_only=True)
            failed = db.scalar(select(func.count(AiUsage.id)).where(
                AiUsage.model == model, AiUsage.at >= start, AiUsage.ok.is_(False))) or 0
            per_min = _counts(db, model, now - timedelta(seconds=60), ok_only=False)
            last_429 = db.scalar(select(func.max(AiUsage.at)).where(
                AiUsage.model == model, AiUsage.code == "429", AiUsage.at >= start))
            pct = min(100, round(used / lim["rpd"] * 100)) if lim["rpd"] else 0
            rows.append({
                "model": model, "used": used, "failed": failed, "limit": lim["rpd"],
                "left": max(0, lim["rpd"] - used) if lim["rpd"] else None, "pct": pct,
                "level": "full" if lim["rpd"] and used >= lim["rpd"] else "high" if pct >= 80 else "mid" if pct >= 50 else "ok",
                "per_min": per_min, "rpm": lim["rpm"], "last_429": last_429, "selected": model == selected,
            })
        tokens = db.execute(select(func.coalesce(func.sum(AiUsage.input_tokens), 0),
                                   func.coalesce(func.sum(AiUsage.output_tokens), 0)).where(AiUsage.at >= start)).one()
    left = (reset - now).total_seconds()
    current = next((r for r in rows if r["selected"]), None)
    total_limit = sum(r["limit"] for r in rows if r["limit"])
    total_used = sum(r["used"] for r in rows)
    return {
        "rows": rows, "current": current, "is_gemini": selected.startswith("gemini"),
        "total_used": total_used, "total_limit": total_limit,
        "total_left": max(0, total_limit - total_used) if total_limit else None,
        "total_pct": min(100, round(total_used / total_limit * 100)) if total_limit else 0,
        "input_tokens": tokens[0], "output_tokens": tokens[1],
        "reset_at": reset, "reset_in": f"{int(left // 3600)} ชม. {int(left % 3600 // 60)} นาที",
    }
