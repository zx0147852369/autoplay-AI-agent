"""รับข้อความใหม่ -> รอลูกค้าพิมพ์ครบ -> ให้ AI วิเคราะห์ -> สร้างร่างคำตอบ (รออนุมัติ) + ticket"""

import asyncio
import json
import logging
import re
import time
from datetime import timedelta
from difflib import SequenceMatcher

from sqlalchemy import func, select

from . import ai_service, dev_bridge
from .telegram_service import STICKER_TEXT
from .database import (
    Chat, Message, Reply, SessionLocal, Ticket, TicketAttachment, TicketEvent, get_settings, int_setting, utcnow,
)
from .site_checker import check_site

log = logging.getLogger(__name__)

OPEN_STATUSES = ("open", "in_progress")

SWEEP_INTERVAL = 60          # ตรวจหาข้อความค้างทุก 60 วินาที
RETRY_AFTER_ERROR = 5 * 60   # ถ้า AI ผิดพลาด รอ 5 นาทีก่อนลองใหม่ (กันโควตาฟรีหมดเร็ว)

# ขอลิงก์เว็บไซต์จากลูกค้าเมื่อแจ้งปัญหาแต่ไม่ได้ให้ลิงก์
ASK_LINK = "รบกวนขอลิงก์เว็บไซต์ที่พบปัญหาด้วยนะคะ"
ASK_LINK_FULL = "รับเรื่องแล้วค่ะ " + ASK_LINK
LINK_REQUEST_RE = re.compile(r"ลิงก์|ลิงค์|ลิ้งค์|ลิ้งก์|link|url", re.IGNORECASE)
URL_RE = re.compile(r"https?://[^\s<>\"')]+", re.IGNORECASE)
# ลิงก์รูปภาพ / โซเชียล ไม่ใช่เว็บไซต์ของลูกค้า
NOT_SITE_HOSTS = ("imgur.com", "pic.in.th", "ibb.co", "postimg", "prnt.sc", "gyazo", "t.me", "telegram.",
                  "line.me", "lin.ee", "facebook.com", "fb.com", "youtube.com", "youtu.be", "google.com/drive",
                  "drive.google", "tiktok.com", "instagram.com")

_timers: dict[int, asyncio.Task] = {}
_last_attempt: dict[int, float] = {}
_locks: dict[int, asyncio.Lock] = {}
last_error: dict[int, str] = {}  # chat_id -> ข้อผิดพลาดล่าสุด (แสดงบนหน้าเว็บ)


def schedule(chat_id: int) -> None:
    """ลูกค้ามักส่งหลายข้อความติดกัน จึงรอจนเงียบไป N วินาทีแล้วค่อยวิเคราะห์ทีเดียว"""
    with SessionLocal() as db:
        delay = int_setting(get_settings(db), "debounce_seconds", 0, 600)
    if (task := _timers.get(chat_id)) and not task.done():
        task.cancel()
    _timers[chat_id] = asyncio.get_running_loop().create_task(_delayed(chat_id, delay))


async def _delayed(chat_id: int, delay: int) -> None:
    try:
        await asyncio.sleep(delay)
    except asyncio.CancelledError:
        return
    try:
        await analyze(chat_id)
    except Exception:  # noqa: BLE001 - งานเบื้องหลัง ห้ามล้มทั้งระบบ
        log.exception("analyze chat %s failed", chat_id)


def is_waiting(chat_id: int) -> bool:
    task = _timers.get(chat_id)
    return bool(task and not task.done())


async def sweep_once() -> None:
    """หาแชทที่มีข้อความลูกค้ายังไม่ได้วิเคราะห์ แล้ววิเคราะห์ให้อัตโนมัติ"""
    with SessionLocal() as db:
        delay = int_setting(get_settings(db), "debounce_seconds", 0, 600)
        monitored = set(db.scalars(select(Chat.id).where(Chat.monitored)))
        rows = db.execute(
            select(Message.chat_id, func.max(Message.date))
            .where(Message.analyzed.is_(False), Message.is_outgoing.is_(False))
            .group_by(Message.chat_id)
        ).all()
    now = utcnow()
    for chat_id, last in rows:
        if chat_id not in monitored or is_waiting(chat_id) or (now - last).total_seconds() < delay:
            continue
        if chat_id in last_error and time.monotonic() - _last_attempt.get(chat_id, 0) < RETRY_AFTER_ERROR:
            continue
        try:
            await analyze(chat_id)
        except Exception:  # noqa: BLE001 - งานเบื้องหลัง ห้ามล้มทั้งระบบ
            log.exception("auto analyze chat %s failed", chat_id)


async def sweeper() -> None:
    while True:
        await asyncio.sleep(SWEEP_INTERVAL)
        try:
            await sweep_once()
        except Exception:  # noqa: BLE001
            log.exception("sweeper failed")


async def analyze(chat_id: int) -> str:
    """วิเคราะห์ข้อความที่ยังไม่ได้วิเคราะห์ของแชทนี้ คืนค่าข้อความสรุปผล"""
    _last_attempt[chat_id] = time.monotonic()
    lock = _locks.setdefault(chat_id, asyncio.Lock())
    async with lock:
        with SessionLocal() as db:
            settings = get_settings(db)
            chat = db.get(Chat, chat_id)
            new_messages = list(db.scalars(
                select(Message).where(Message.chat_id == chat_id, Message.analyzed.is_(False))
                .order_by(Message.date, Message.id)
            ))
            if not new_messages:
                return "ไม่มีข้อความใหม่"
            limit = max(len(new_messages), int_setting(settings, "context_messages", 5, 100))
            history = list(db.scalars(
                select(Message).where(Message.chat_id == chat_id)
                .order_by(Message.date.desc(), Message.id.desc()).limit(limit)
            ))[::-1]
            open_tickets = list(db.scalars(
                select(Ticket).where(Ticket.chat_id == chat_id, Ticket.status.in_(OPEN_STATUSES))
            ))

        if str(chat_id) == settings.get("dev_group_id"):
            _mark_analyzed(new_messages)  # กลุ่มโปรแกรมเมอร์ไม่ใช่แชทลูกค้า
            return "กลุ่มโปรแกรมเมอร์"
        if settings.get("auto_draft") != "1" and settings.get("auto_ticket") != "1":
            _mark_analyzed(new_messages)
            return "ปิดการทำงานอัตโนมัติไว้"
        customer_new = [m for m in new_messages if not m.is_outgoing]
        if all(not m.media_path and (m.text or "").strip() in ("", STICKER_TEXT) for m in customer_new):
            _mark_analyzed(new_messages)  # มีแต่สติกเกอร์ / ข้อความว่าง ไม่ต้องเรียก AI
            return "มีแต่สติกเกอร์ ไม่ต้องวิเคราะห์"
        rejected = recent_rejected(chat_id)

        try:
            result = await ai_service.analyze_chat(
                settings, chat.title if chat else str(chat_id), history, new_messages, open_tickets,
                rejected=[r for r, _ in rejected],
            )
        except ai_service.AIError as e:
            last_error[chat_id] = str(e)
            log.warning("AI error on chat %s: %s", chat_id, e)
            return f"ผิดพลาด: {e}"
        last_error.pop(chat_id, None)

        ticket_id = None
        if result.is_issue and settings.get("auto_ticket") == "1":
            ticket_id = await _save_ticket(chat_id, result, new_messages, settings)
        ask_link = bool(ticket_id) and settings.get("ask_link") == "1" and needs_link_request(ticket_id)
        if ask_link:
            if result.needs_reply and result.reply_text.strip():
                if not LINK_REQUEST_RE.search(result.reply_text):
                    result.reply_text = result.reply_text.rstrip() + "\n" + ASK_LINK
            else:
                result.needs_reply, result.reply_text = True, ASK_LINK_FULL
            result.note_for_admin = (result.note_for_admin + " · ระบบขอลิงก์เว็บไซต์จากลูกค้า").strip(" ·")
        similar = next((text for text, _ in rejected if is_similar(result.reply_text, text)), None)
        if result.needs_reply and similar:
            log.info("ไม่สร้างร่างซ้ำกับที่แอดมินปฏิเสธไปแล้วในแชท %s", chat_id)
            result.needs_reply = False
        if result.needs_reply and result.reply_text.strip() and (settings.get("auto_draft") == "1" or ask_link):
            _save_draft(chat_id, result, new_messages, ticket_id)
        _mark_analyzed(new_messages)

        parts = []
        if ticket_id:
            parts.append(f"ticket #{ticket_id}")
        if result.needs_reply:
            parts.append("สร้างร่างคำตอบแล้ว")
        return ", ".join(parts) or "ไม่ต้องตอบ / ไม่ใช่การแจ้งปัญหา"


REJECT_MEMORY_HOURS = 24
SIMILAR_RATIO = 0.6


def recent_rejected(chat_id: int) -> list[tuple[str, str]]:
    """ร่างที่แอดมินปฏิเสธในแชทนี้ภายใน 24 ชม. -> [(ข้อความ, เหตุผล)]"""
    since = utcnow() - timedelta(hours=REJECT_MEMORY_HOURS)
    with SessionLocal() as db:
        rows = db.execute(select(Reply.final_text, Reply.reject_reason).where(
            Reply.chat_id == chat_id, Reply.status == "rejected", Reply.decided_at >= since
        ).order_by(Reply.decided_at.desc()).limit(5)).all()
    return [(text or "", reason or "") for text, reason in rows]


def is_similar(a: str, b: str) -> bool:
    a, b = re.sub(r"\s+", " ", a or "").strip(), re.sub(r"\s+", " ", b or "").strip()
    return bool(a and b) and SequenceMatcher(None, a, b).ratio() >= SIMILAR_RATIO


def find_site_url(messages: list[Message]) -> str:
    """ลิงก์เว็บไซต์แรกที่ลูกค้าพิมพ์มา (ข้ามลิงก์รูปภาพ/โซเชียล)"""
    for m in messages:
        for url in URL_RE.findall(m.text or ""):
            url = url.rstrip(".,;")
            if not any(host in url.lower() for host in NOT_SITE_HOSTS):
                return url
    return ""


def needs_link_request(ticket_id: int) -> bool:
    """ticket ยังไม่มีลิงก์เว็บไซต์ และยังไม่เคยขอลิงก์จากลูกค้า"""
    with SessionLocal() as db:
        ticket = db.get(Ticket, ticket_id)
        if not ticket or ticket.website_url:
            return False
        # ร่าง AI ที่ยังไม่ส่งจะถูกแทนที่ด้วยร่างใหม่ จึงนับเฉพาะที่ส่งแล้ว หรือข้อความประเภทอื่นที่ยังรออยู่
        asked = db.scalars(select(Reply.final_text).where(
            Reply.ticket_id == ticket_id,
            Reply.status.in_(("sending", "sent")) | (Reply.status.in_(("pending", "failed")) & (Reply.kind != "ai"))))
        return not any(LINK_REQUEST_RE.search(text or "") for text in asked)


def _mark_analyzed(messages: list[Message]) -> None:
    with SessionLocal() as db:
        for m in messages:
            db.get(Message, m.id).analyzed = True
        db.commit()


def _save_draft(chat_id: int, result: ai_service.Analysis, new_messages: list[Message], ticket_id) -> None:
    valid_ids = {m.tg_message_id for m in new_messages if not m.is_outgoing}
    reply_to = result.reply_to_message_id if result.reply_to_message_id in valid_ids else None
    if reply_to is None and valid_ids:
        reply_to = max(valid_ids)
    with SessionLocal() as db:
        # ร่างเก่าที่ยังไม่อนุมัติของแชทนี้ล้าสมัยแล้ว เพราะมีข้อความใหม่เข้ามา
        for old in db.scalars(select(Reply).where(
                Reply.chat_id == chat_id, Reply.status == "pending", Reply.kind == "ai")):
            old.status = "superseded"
        db.add(Reply(
            chat_id=chat_id,
            reply_to_tg_id=reply_to,
            ai_text=result.reply_text.strip(),
            final_text=result.reply_text.strip(),
            note=result.note_for_admin,
            ticket_id=ticket_id,
        ))
        db.commit()


async def _save_ticket(chat_id: int, result: ai_service.Analysis, new_messages: list[Message], settings) -> int:
    customer_msgs = [m for m in new_messages if not m.is_outgoing]
    with SessionLocal() as db:
        ticket = None
        created = False
        if result.existing_ticket_id:
            ticket = db.get(Ticket, result.existing_ticket_id)
            if not ticket or ticket.chat_id != chat_id or ticket.status not in OPEN_STATUSES:
                ticket = None
        if ticket is None:
            ticket = Ticket(
                chat_id=chat_id,
                category=result.issue_category,
                title=result.issue_title or ai_service.CATEGORIES.get(result.issue_category, "ปัญหา"),
                summary=result.issue_summary,
                severity=result.severity,
                customer_name=result.customer_name or (customer_msgs[0].sender_name if customer_msgs else ""),
                website_url=result.website_url.strip() or find_site_url(customer_msgs),
            )
            db.add(ticket)
            db.flush()
            created = True
            db.add(TicketEvent(ticket_id=ticket.id, kind="ai_summary", author="AI", body=result.issue_summary))
        else:
            db.add(TicketEvent(ticket_id=ticket.id, kind="ai_summary", author="AI",
                               body="ข้อมูลเพิ่มเติม: " + result.issue_summary))
            if not ticket.website_url:
                ticket.website_url = result.website_url.strip() or find_site_url(customer_msgs)
            if ai_service.SEVERITIES.index(result.severity) > ai_service.SEVERITIES.index(ticket.severity):
                ticket.severity = result.severity

        for m in customer_msgs:
            db.add(TicketEvent(ticket_id=ticket.id, kind="customer_message", author=m.sender_name,
                               body=m.text or "", media_path=m.media_path or "", created_at=m.date))
            if m.media_path:
                db.add(TicketAttachment(ticket_id=ticket.id, media_path=m.media_path, caption=m.text))
        db.commit()
        ticket_id, url = ticket.id, ticket.website_url

    if url and settings.get("site_check") == "1":
        try:
            await run_site_check(ticket_id)
        except Exception:  # noqa: BLE001
            log.exception("site check failed for ticket %s", ticket_id)
    # แจ้งกลุ่มโปรแกรมเมอร์: ticket ใหม่ -> โพสต์รายละเอียด, ticket เดิม -> ส่งข้อมูลเพิ่มจากลูกค้า
    try:
        if created:
            await dev_bridge.post_ticket(ticket_id)
        else:
            await dev_bridge.post_customer_update(ticket_id, customer_msgs)
    except Exception:  # noqa: BLE001
        log.exception("notify dev group failed for ticket %s", ticket_id)
    return ticket_id


async def run_site_check(ticket_id: int) -> dict | None:
    with SessionLocal() as db:
        ticket = db.get(Ticket, ticket_id)
        url = ticket.website_url if ticket else ""
    if not url:
        return None
    result = await check_site(url)
    with SessionLocal() as db:
        ticket = db.get(Ticket, ticket_id)
        ticket.site_check = json.dumps(result, ensure_ascii=False)
        status = "เข้าไม่ได้" if not result["ok"] else (
            f"ถูกพาไปโดเมนอื่น ({result.get('final_host')})" if result.get("other_host") else "เข้าได้")
        detail = f"HTTP {result['status_code']}" if result["status_code"] else result["error"]
        db.add(TicketEvent(ticket_id=ticket_id, kind="site_check", author="ระบบ",
                           body=f"ตรวจ {result['url']}: {status} ({detail}, {result['elapsed_ms']} ms)"))
        db.commit()
    return result
