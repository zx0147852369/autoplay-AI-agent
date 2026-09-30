"""รับข้อความใหม่ -> รอลูกค้าพิมพ์ครบ -> ให้ AI วิเคราะห์ -> สร้างร่างคำตอบ (รออนุมัติ) + ticket"""

import asyncio
import json
import logging

from sqlalchemy import select

from . import ai_service
from .database import Chat, Message, Reply, SessionLocal, Ticket, TicketAttachment, TicketEvent, get_settings
from .site_checker import check_site

log = logging.getLogger(__name__)

OPEN_STATUSES = ("open", "in_progress")

_timers: dict[int, asyncio.Task] = {}
_locks: dict[int, asyncio.Lock] = {}
last_error: dict[int, str] = {}  # chat_id -> ข้อผิดพลาดล่าสุด (แสดงบนหน้าเว็บ)


def schedule(chat_id: int) -> None:
    """ลูกค้ามักส่งหลายข้อความติดกัน จึงรอจนเงียบไป N วินาทีแล้วค่อยวิเคราะห์ทีเดียว"""
    with SessionLocal() as db:
        delay = max(0, int(get_settings(db).get("debounce_seconds") or 15))
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


async def analyze(chat_id: int) -> str:
    """วิเคราะห์ข้อความที่ยังไม่ได้วิเคราะห์ของแชทนี้ คืนค่าข้อความสรุปผล"""
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
            limit = max(len(new_messages), int(settings.get("context_messages") or 20))
            history = list(db.scalars(
                select(Message).where(Message.chat_id == chat_id)
                .order_by(Message.date.desc(), Message.id.desc()).limit(limit)
            ))[::-1]
            open_tickets = list(db.scalars(
                select(Ticket).where(Ticket.chat_id == chat_id, Ticket.status.in_(OPEN_STATUSES))
            ))

        if settings.get("auto_draft") != "1" and settings.get("auto_ticket") != "1":
            _mark_analyzed(new_messages)
            return "ปิดการทำงานอัตโนมัติไว้"

        try:
            result = await ai_service.analyze_chat(
                settings, chat.title if chat else str(chat_id), history, new_messages, open_tickets
            )
        except ai_service.AIError as e:
            last_error[chat_id] = str(e)
            log.warning("AI error on chat %s: %s", chat_id, e)
            return f"ผิดพลาด: {e}"
        last_error.pop(chat_id, None)

        ticket_id = None
        if result.is_issue and settings.get("auto_ticket") == "1":
            ticket_id = await _save_ticket(chat_id, result, new_messages, settings)
        if result.needs_reply and result.reply_text.strip() and settings.get("auto_draft") == "1":
            _save_draft(chat_id, result, new_messages, ticket_id)
        _mark_analyzed(new_messages)

        parts = []
        if ticket_id:
            parts.append(f"ticket #{ticket_id}")
        if result.needs_reply:
            parts.append("สร้างร่างคำตอบแล้ว")
        return ", ".join(parts) or "ไม่ต้องตอบ / ไม่ใช่การแจ้งปัญหา"


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
        for old in db.scalars(select(Reply).where(Reply.chat_id == chat_id, Reply.status == "pending")):
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
                website_url=result.website_url.strip(),
            )
            db.add(ticket)
            db.flush()
            db.add(TicketEvent(ticket_id=ticket.id, kind="ai_summary", author="AI", body=result.issue_summary))
        else:
            db.add(TicketEvent(ticket_id=ticket.id, kind="ai_summary", author="AI",
                               body="ข้อมูลเพิ่มเติม: " + result.issue_summary))
            if result.website_url and not ticket.website_url:
                ticket.website_url = result.website_url.strip()
            if ai_service.SEVERITIES.index(result.severity) > ai_service.SEVERITIES.index(ticket.severity):
                ticket.severity = result.severity

        for m in customer_msgs:
            body = m.text or ("(รูปภาพ)" if m.media_path else "")
            db.add(TicketEvent(ticket_id=ticket.id, kind="customer_message", author=m.sender_name,
                               body=body, created_at=m.date))
            if m.media_path:
                db.add(TicketAttachment(ticket_id=ticket.id, media_path=m.media_path, caption=m.text))
        db.commit()
        ticket_id, url = ticket.id, ticket.website_url

    if url and settings.get("site_check") == "1":
        await run_site_check(ticket_id)
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
        status = "เข้าได้" if result["ok"] else "เข้าไม่ได้"
        detail = f"HTTP {result['status_code']}" if result["status_code"] else result["error"]
        db.add(TicketEvent(ticket_id=ticket_id, kind="site_check", author="ระบบ",
                           body=f"ตรวจ {result['url']}: {status} ({detail}, {result['elapsed_ms']} ms)"))
        db.commit()
    return result
