"""เชื่อมกับกลุ่มโปรแกรมเมอร์ (เช่น Autopay Support)

- ticket ใหม่ -> โพสต์รายละเอียดและรูปเข้ากลุ่ม
- ลูกค้าส่งข้อมูลเพิ่ม -> ตอบใต้โพสต์ ticket เดิม
- โปรแกรมเมอร์ตอบ (reply โพสต์ ticket หรือพิมพ์ #เลข) -> อัปเดตสถานะ และร่างข้อความถึงลูกค้า (รออนุมัติ)
"""

import asyncio
import json
import logging
import re

from sqlalchemy import select

from . import ai_service
from .config import MEDIA_DIR, PUBLIC_URL
from .database import (
    Chat,
    Message,
    SessionLocal,
    Ticket,
    TicketEvent,
    TicketLink,
    get_settings,
)
from .notices import draft_update, sync_resolved_notice
from .telegram_service import telegram

log = logging.getLogger(__name__)

SEVERITY_TH = {"low": "ต่ำ", "medium": "ปานกลาง", "high": "สูง", "critical": "วิกฤต"}
STATUS_TH = {"open": "เปิดใหม่", "in_progress": "กำลังแก้ไข", "resolved": "แก้ไขแล้ว", "closed": "ปิด"}
MAX_PHOTOS = 6

_tasks: set[asyncio.Task] = set()


def configure() -> None:
    """โหลดการตั้งค่ากลุ่มโปรแกรมเมอร์ให้ telegram_service"""
    with SessionLocal() as db:
        settings = get_settings(db)
    telegram.set_dev(settings.get("dev_group_id"), settings.get("dev_usernames", ""))
    telegram.on_dev_message = _schedule


def _schedule(info: dict) -> None:
    task = asyncio.get_running_loop().create_task(_safe_handle(info))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)


async def _safe_handle(info: dict) -> None:
    try:
        await handle_dev_message(info)
    except Exception:  # noqa: BLE001 - งานเบื้องหลัง ห้ามล้มทั้งระบบ
        log.exception("handle dev message failed")


def _group_id(settings: dict[str, str]) -> int | None:
    try:
        return int(settings.get("dev_group_id") or 0) or None
    except ValueError:
        return None


def _save_links(ticket_id: int, chat_id: int, message_ids: list[int]) -> None:
    with SessionLocal() as db:
        for mid in message_ids:
            db.add(TicketLink(ticket_id=ticket_id, chat_id=chat_id, tg_message_id=mid))
        db.commit()


def format_ticket(ticket: Ticket, chat_title: str) -> str:
    lines = [
        f"Ticket #{ticket.id} · {ai_service.CATEGORIES.get(ticket.category, ticket.category)}",
        f"ความรุนแรง: {SEVERITY_TH.get(ticket.severity, ticket.severity)}",
        f"ลูกค้า: {ticket.customer_name or '-'} · แชท: {chat_title}",
    ]
    if ticket.website_url:
        site = ""
        if ticket.site_check:
            sc = json.loads(ticket.site_check)
            site = " (เข้าได้)" if sc.get("ok") else f" (เข้าไม่ได้: {sc.get('error') or sc.get('status_code')})"
        lines.append(f"เว็บไซต์: {ticket.website_url}{site}")
    lines += ["", f"ปัญหา: {ticket.title}", ticket.summary]
    if PUBLIC_URL:
        lines += ["", f"ดูรายละเอียด: {PUBLIC_URL}/tickets/{ticket.id}"]
    lines += ["", "ตอบกลับ (reply) ข้อความนี้เพื่ออัปเดต เช่น \"กำลังแก้\" \"แก้เสร็จแล้ว\" หรือขอข้อมูลลูกค้าเพิ่ม"]
    return "\n".join(lines)


async def post_ticket(ticket_id: int, force: bool = False) -> str:
    """โพสต์ ticket เข้ากลุ่มโปรแกรมเมอร์ คืนค่าข้อความสรุปผล"""
    with SessionLocal() as db:
        settings = get_settings(db)
        ticket = db.get(Ticket, ticket_id)
        if not ticket:
            return "ไม่พบ ticket"
        posted = db.scalar(select(TicketLink.id).where(TicketLink.ticket_id == ticket_id).limit(1))
        chat = db.get(Chat, ticket.chat_id)
        photos = [str(MEDIA_DIR / a.media_path) for a in ticket.attachments][:MAX_PHOTOS]
        text = format_ticket(ticket, chat.title if chat else str(ticket.chat_id))
    group_id = _group_id(settings)
    if not group_id:
        return "ยังไม่ได้เลือกกลุ่มโปรแกรมเมอร์ในหน้าตั้งค่า"
    if not force and (settings.get("dev_forward") != "1" or posted):
        return ""
    if not telegram.connected:
        return "ยังไม่ได้เชื่อมต่อ Telegram"
    root = await telegram.send_text(group_id, text)
    ids = [root]
    photos = [p for p in photos if (MEDIA_DIR / p).exists()]
    if photos:
        ids += await telegram.send_files(group_id, photos, reply_to=root)
    _save_links(ticket_id, group_id, ids)
    with SessionLocal() as db:
        db.add(TicketEvent(ticket_id=ticket_id, kind="dev", author="ระบบ", body="ส่งรายละเอียดเข้ากลุ่มโปรแกรมเมอร์แล้ว"))
        db.commit()
    return "ส่งเข้ากลุ่มโปรแกรมเมอร์แล้ว"


async def post_customer_update(ticket_id: int, messages: list[Message]) -> None:
    """ลูกค้าส่งข้อมูลเพิ่มเรื่อง ticket เดิม -> ตอบใต้โพสต์ ticket ในกลุ่ม"""
    if not messages or not telegram.connected:
        return
    with SessionLocal() as db:
        settings = get_settings(db)
        root = db.scalar(select(TicketLink.tg_message_id).where(TicketLink.ticket_id == ticket_id)
                         .order_by(TicketLink.id).limit(1))
    group_id = _group_id(settings)
    if not group_id or not root:
        return
    lines = [f"ข้อมูลเพิ่มเติมจากลูกค้า · Ticket #{ticket_id}"]
    lines += [f"- {m.sender_name or 'ลูกค้า'}: {m.text or '(รูปภาพ)'}" for m in messages]
    ids = [await telegram.send_text(group_id, "\n".join(lines), reply_to=root)]
    photos = [str(MEDIA_DIR / m.media_path) for m in messages if m.media_path][:MAX_PHOTOS]
    if photos:
        ids += await telegram.send_files(group_id, photos, reply_to=ids[0])
    _save_links(ticket_id, group_id, ids)


# ---------------------------------------------------------------- ข้อความจากโปรแกรมเมอร์
def find_ticket(db, info: dict) -> Ticket | None:
    if info.get("reply_to"):
        ticket_id = db.scalar(select(TicketLink.ticket_id).where(
            TicketLink.chat_id == info["chat_id"], TicketLink.tg_message_id == info["reply_to"]).limit(1))
        if ticket_id:
            return db.get(Ticket, ticket_id)
    match = re.search(r"#\s?(\d+)", info.get("text") or "")
    if match:
        return db.get(Ticket, int(match.group(1)))
    return None


FALLBACK_MESSAGES = {
    "in_progress": "ทีมงานได้รับเรื่องและกำลังดำเนินการแก้ไขให้ค่ะ รบกวนรอสักครู่นะคะ",
    "resolved": "",  # ใช้ข้อความแจ้งแก้ไขเสร็จจากหน้าตั้งค่า
    # ไม่ใส่ข้อความโปรแกรมเมอร์ตรงๆ (อาจมีศัพท์ภายใน) แอดมินแก้ให้ตรงกับที่ต้องการก่อนอนุมัติ ดูได้จากหมายเหตุ
    "need_info": "รบกวนขอข้อมูลเพิ่มเติมเพื่อให้ทีมงานตรวจสอบได้ค่ะ เช่น ยูสเซอร์ที่ใช้งาน และภาพหน้าจอที่พบปัญหา",
}


def guess_intent(text: str) -> str:
    """ใช้เมื่อ AI ใช้งานไม่ได้ (เช่น โควตาเต็ม): จัดประเภทจากคำสำคัญ"""
    t = text.lower()
    if re.search(r"เสร็จ|เรียบร้อย|แก้แล้ว|แก้ให้แล้ว|ใช้ได้แล้ว|fixed|done|ลองใหม่", t):
        return "resolved"
    if re.search(r"ขอ\s*(ยูส|user|สลิป|รูป|ข้อมูล|เบอร์|ไอดี|id|ลิงก์|link)|ส่ง.*มา", t):
        return "need_info"
    if re.search(r"รับ|กำลัง|ดูให้|ตรวจสอบ|เช็ค|เช็ก|check|ok|โอเค", t):
        return "in_progress"
    return "comment"


async def handle_dev_message(info: dict) -> str:
    """ประมวลผลข้อความโปรแกรมเมอร์ คืนค่า intent ที่ใช้ (หรือค่าว่างถ้าไม่เกี่ยวกับ ticket)"""
    with SessionLocal() as db:
        settings = get_settings(db)
        if settings.get("dev_watch") != "1":
            return ""
        ticket = find_ticket(db, info)
        if not ticket:
            log.info("dev message %s not linked to any ticket", info.get("message_id"))
            return ""
        ticket_id = ticket.id
        db.add(TicketEvent(ticket_id=ticket_id, kind="dev", author=f"@{info['username']}", body=info["text"]))
        db.commit()
        db.refresh(ticket)
        db.expunge(ticket)

    try:
        result = await ai_service.classify_dev_message(settings, ticket, info["text"])
    except ai_service.AIError as e:
        intent = guess_intent(info["text"])
        result = ai_service.DevIntent(intent, FALLBACK_MESSAGES.get(intent, ""),
                                      f"จัดประเภทจากคำสำคัญ (AI ใช้งานไม่ได้: {e})")

    who = f"@{info['username']}"
    note = f"จากโปรแกรมเมอร์ {who}: {info['text']}"
    ack = []
    with SessionLocal() as db:
        ticket = db.get(Ticket, ticket_id)
        old = ticket.status
        if result.intent == "in_progress" and ticket.status == "open":
            ticket.status = "in_progress"
        elif result.intent == "resolved" and ticket.status in ("open", "in_progress"):
            ticket.status = "resolved"
        if ticket.status != old:
            db.add(TicketEvent(ticket_id=ticket_id, kind="status", author=who,
                               body=f"สถานะ: {STATUS_TH[old]} → {STATUS_TH[ticket.status]}"))
            ack.append(f"สถานะ → {STATUS_TH[ticket.status]}")
        db.commit()

        drafted = False
        if result.intent == "resolved" and ticket.status != old:
            drafted = sync_resolved_notice(db, ticket, who, text=result.customer_message or None) == "created"
        elif result.intent == "need_info" and result.customer_message:
            draft_update(db, ticket, result.customer_message, note)
            drafted = True
        elif result.intent == "in_progress" and ticket.status != old and result.customer_message:
            draft_update(db, ticket, result.customer_message, note)
            drafted = True
        if drafted:
            ack.append("ร่างข้อความถึงลูกค้าแล้ว รอแอดมินอนุมัติ")

    if ack and telegram.connected:
        try:
            mid = await telegram.send_text(info["chat_id"], f"Ticket #{ticket_id}: " + " · ".join(ack),
                                           reply_to=info["message_id"])
            _save_links(ticket_id, info["chat_id"], [mid])
        except Exception:  # noqa: BLE001
            log.exception("send ack to dev group failed")
    return result.intent
