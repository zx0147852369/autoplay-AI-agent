"""เก็บ log ล่าสุดไว้ในหน่วยความจำ เพื่อให้ดูในหน้าเว็บได้ (บรรทัดเดียวกับที่ไปขึ้นใน Railway log)

เก็บแบบวงแหวน (ring buffer) ~800 บรรทัด · หายเมื่อรีสตาร์ต/ดีพลอยใหม่ (ประวัติเต็มดูได้ที่ Railway)
"""

import logging
from collections import deque
from datetime import datetime, timezone

MAX = 800
_RECORDS: deque = deque(maxlen=MAX)
_SEQ = 0
_LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}
_fmt = logging.Formatter()


class BufferHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        global _SEQ
        try:
            msg = record.getMessage()
            if record.exc_info:
                msg += "\n" + _fmt.formatException(record.exc_info)
        except Exception:  # noqa: BLE001 - handler ห้าม throw
            return
        _SEQ += 1
        _RECORDS.append({
            "id": _SEQ,
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "msg": msg,
        })


def install(level: int = logging.INFO) -> None:
    handler = BufferHandler()
    handler.setLevel(level)
    logging.getLogger().addHandler(handler)  # root: รับ log ของแอป (app.*) ที่ propagate ขึ้นมา
    # uvicorn ไม่ส่งต่อไป root -> ติด handler ให้เองเพื่อเก็บ log ตอนเริ่ม/ตอน error (ข้าม access log เพราะจะรก)
    for name in ("uvicorn", "uvicorn.error"):
        logging.getLogger(name).addHandler(handler)


def records(after: int = 0, level: str = "", limit: int = 500) -> dict:
    thr = _LEVELS.get(level, 0)
    items = [r for r in _RECORDS if r["id"] > after and (not thr or _LEVELS.get(r["level"], 0) >= thr)]
    return {"records": items[-limit:], "last": _SEQ, "total": len(_RECORDS)}
