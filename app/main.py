import asyncio
import json
import logging
import sys
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import timezone
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from starlette.middleware.sessions import SessionMiddleware

from . import ai_service, analyzer
from .config import ADMIN_PASSWORD, ADMIN_USERNAME, DISPLAY_TZ, MEDIA_DIR, SECRET_KEY
from .database import (
    DEFAULT_SETTINGS,
    Chat,
    Message,
    Reply,
    SessionLocal,
    Setting,
    TelegramAccount,
    Ticket,
    TicketEvent,
    User,
    get_settings,
    init_db,
    int_setting,
    utcnow,
)
from .security import decrypt, hash_password, verify_password
from .telegram_service import TelegramLoginError, telegram

# คอนโซล Windows ไม่ใช่ UTF-8 โดยค่าเริ่มต้น ทำให้ log ภาษาไทยพัง
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

APP_DIR = Path(__file__).resolve().parent
ROLES = {"admin": "ผู้ดูแลระบบ", "agent": "แอดมินตอบแชท", "programmer": "โปรแกรมเมอร์"}
TICKET_STATUSES = {"open": "เปิดใหม่", "in_progress": "กำลังแก้ไข", "resolved": "แก้ไขแล้ว", "closed": "ปิด"}
SEVERITY_LABELS = {"low": "ต่ำ", "medium": "ปานกลาง", "high": "สูง", "critical": "วิกฤต"}
REPLY_STATUSES = {"pending": "รออนุมัติ", "sent": "ส่งแล้ว", "rejected": "ปฏิเสธ",
                  "failed": "ส่งไม่สำเร็จ", "superseded": "ถูกแทนที่", "sending": "กำลังส่ง"}


def ensure_admin() -> None:
    """ADMIN_USERNAME / ADMIN_PASSWORD ใน env คือบัญชีผู้ดูแลหลัก: สร้างถ้ายังไม่มี และรีเซ็ตรหัสให้ตรงกับ env
    ทุกครั้งที่เปิดโปรแกรม (ใช้กู้บัญชีได้ด้วยการเปลี่ยน ADMIN_PASSWORD แล้วรีสตาร์ท)"""
    if not ADMIN_PASSWORD:
        log.warning("ยังไม่ได้ตั้งค่า ADMIN_PASSWORD: ตั้งค่าใน .env / Variables แล้วรีสตาร์ท")
        return
    with SessionLocal() as db:
        user = db.scalar(select(User).where(User.username == ADMIN_USERNAME))
        if user is None:
            db.add(User(username=ADMIN_USERNAME, password_hash=hash_password(ADMIN_PASSWORD), role="admin"))
            log.info("สร้างผู้ใช้ผู้ดูแลระบบ '%s' แล้ว", ADMIN_USERNAME)
        elif not verify_password(ADMIN_PASSWORD, user.password_hash) or user.role != "admin":
            user.password_hash, user.role = hash_password(ADMIN_PASSWORD), "admin"
            log.info("อัปเดตรหัสผ่านผู้ดูแลระบบ '%s' ตาม ADMIN_PASSWORD แล้ว", ADMIN_USERNAME)
        db.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    ensure_admin()
    telegram.on_message = analyzer.schedule
    startup = asyncio.create_task(telegram.start_from_db())
    yield
    startup.cancel()
    await telegram.stop()


app = FastAPI(title="Telegram AI Support", lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY, same_site="lax", max_age=60 * 60 * 12)
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=APP_DIR / "templates")


def _localtime(dt):
    if not dt:
        return "-"
    return dt.replace(tzinfo=timezone.utc).astimezone(DISPLAY_TZ).strftime("%d/%m/%Y %H:%M")


templates.env.filters["localtime"] = _localtime
templates.env.filters["fromjson"] = lambda s: json.loads(s) if s else None
templates.env.globals.update(
    CATEGORIES=ai_service.CATEGORIES, TICKET_STATUSES=TICKET_STATUSES, SEVERITY_LABELS=SEVERITY_LABELS,
    REPLY_STATUSES=REPLY_STATUSES, ROLES=ROLES, ENV_ADMIN=ADMIN_USERNAME,
)


# ---------------------------------------------------------------- auth helpers
class NeedLogin(Exception):
    pass


class Forbidden(Exception):
    pass


@app.exception_handler(NeedLogin)
async def _need_login(request: Request, exc: NeedLogin):
    return RedirectResponse("/login", status_code=303)


@app.exception_handler(Forbidden)
async def _forbidden(request: Request, exc: Forbidden):
    flash(request, "คุณไม่มีสิทธิ์เข้าถึงหน้านี้", "error")
    return RedirectResponse("/tickets", status_code=303)


def current_user(request: Request, *roles: str) -> User:
    user_id = request.session.get("user_id")
    if not user_id:
        raise NeedLogin()
    with SessionLocal() as db:
        user = db.get(User, user_id)
    if not user:
        request.session.clear()
        raise NeedLogin()
    if roles and user.role not in roles and user.role != "admin":
        raise Forbidden()
    return user


def flash(request: Request, message: str, kind: str = "ok") -> None:
    # เก็บไว้แค่ 3 ข้อความล่าสุด กันคุกกี้ session ใหญ่เกินจนเบราว์เซอร์ทิ้ง
    request.session["flash"] = (request.session.get("flash", []) + [{"message": message, "kind": kind}])[-3:]


def render(request: Request, name: str, user: User | None, **context):
    context.update(user=user, flashes=request.session.pop("flash", []), path=request.url.path)
    return templates.TemplateResponse(request, name, context)


def back(url: str) -> RedirectResponse:
    return RedirectResponse(url, status_code=303)


# ---------------------------------------------------------------- login
@app.get("/login")
async def login_page(request: Request):
    return render(request, "login.html", None, need_setup=not ADMIN_PASSWORD)


# กันการเดารหัสผ่าน: ผิดเกิน LOGIN_MAX_FAILS ครั้งใน LOGIN_WINDOW วินาที ต่อ IP -> ต้องรอ
LOGIN_MAX_FAILS = 8
LOGIN_WINDOW = 10 * 60
_login_fails: dict[str, deque] = defaultdict(deque)


@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    ip = request.client.host if request.client else "?"
    fails, now = _login_fails[ip], time.monotonic()
    while fails and now - fails[0] > LOGIN_WINDOW:
        fails.popleft()
    if len(fails) >= LOGIN_MAX_FAILS:
        wait = int(LOGIN_WINDOW - (now - fails[0])) // 60 + 1
        flash(request, f"ใส่รหัสผ่านผิดบ่อยเกินไป กรุณาลองใหม่ในอีก {wait} นาที", "error")
        return back("/login")
    with SessionLocal() as db:
        user = db.scalar(select(User).where(User.username == username.strip()))
    if not user or not verify_password(password, user.password_hash):
        fails.append(now)
        flash(request, "ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง", "error")
        return back("/login")
    fails.clear()
    request.session.clear()
    request.session["user_id"] = user.id
    return back("/" if user.role != "programmer" else "/tickets")


@app.post("/logout")
async def logout(request: Request):
    request.session.clear()
    return back("/login")


# ---------------------------------------------------------------- dashboard
@app.get("/")
async def dashboard(request: Request):
    user = current_user(request, "agent")
    with SessionLocal() as db:
        account = db.get(TelegramAccount, 1)
        pending = db.scalar(select(func.count(Reply.id)).where(Reply.status == "pending"))
        open_by_cat = dict(db.execute(
            select(Ticket.category, func.count(Ticket.id))
            .where(Ticket.status.in_(analyzer.OPEN_STATUSES)).group_by(Ticket.category)
        ).all())
        recent = list(db.scalars(select(Ticket).order_by(Ticket.created_at.desc()).limit(8)))
        monitored = db.scalar(select(func.count(Chat.id)).where(Chat.monitored))
        chat_titles = {c.id: c.title for c in db.scalars(select(Chat))}
    return render(request, "dashboard.html", user, account=account, pending=pending, open_by_cat=open_by_cat,
                  recent=recent, monitored=monitored, chat_titles=chat_titles,
                  ai_errors=analyzer.last_error, connected=telegram.connected)


@app.get("/api/badge")
async def badge(request: Request):
    current_user(request)
    with SessionLocal() as db:
        pending = db.scalar(select(func.count(Reply.id)).where(Reply.status == "pending"))
        open_tickets = db.scalar(select(func.count(Ticket.id)).where(Ticket.status.in_(analyzer.OPEN_STATUSES)))
    return JSONResponse({"pending": pending, "open_tickets": open_tickets})


# ---------------------------------------------------------------- telegram account
@app.get("/telegram")
async def telegram_page(request: Request):
    user = current_user(request, "admin")
    with SessionLocal() as db:
        account = db.get(TelegramAccount, 1)
    return render(request, "telegram.html", user, account=account, connected=telegram.connected,
                  has_api_hash=bool(decrypt(account.api_hash_enc)))


@app.post("/telegram/send-code")
async def telegram_send_code(request: Request, api_id: str = Form(...), api_hash: str = Form(""),
                             phone: str = Form(...)):
    current_user(request, "admin")
    if not api_hash.strip():  # ใช้ค่าเดิมที่บันทึกไว้
        with SessionLocal() as db:
            api_hash = decrypt(db.get(TelegramAccount, 1).api_hash_enc)
    try:
        await telegram.send_code(int(api_id), api_hash.strip(), phone.strip().replace(" ", ""))
        flash(request, "ส่งรหัสยืนยันไปที่แอป Telegram ของคุณแล้ว")
    except ValueError:
        flash(request, "API ID ต้องเป็นตัวเลข", "error")
    except TelegramLoginError as e:
        flash(request, str(e), "error")
    except Exception as e:  # noqa: BLE001
        log.exception("send code failed")
        flash(request, f"เชื่อมต่อ Telegram ไม่ได้: {e}", "error")
    return back("/telegram")


@app.post("/telegram/verify-code")
async def telegram_verify_code(request: Request, code: str = Form(...)):
    current_user(request, "admin")
    try:
        result = await telegram.verify_code(code)
        if result == "password_needed":
            flash(request, "บัญชีนี้เปิดการยืนยันสองขั้นตอน กรุณาใส่รหัสผ่าน Telegram")
        else:
            flash(request, "เชื่อมต่อ Telegram สำเร็จ")
    except TelegramLoginError as e:
        flash(request, str(e), "error")
    return back("/telegram")


@app.post("/telegram/verify-password")
async def telegram_verify_password(request: Request, password: str = Form(...)):
    current_user(request, "admin")
    try:
        await telegram.verify_password(password)  # ใช้ครั้งเดียว ไม่บันทึกรหัสผ่านลงระบบ
        flash(request, "เชื่อมต่อ Telegram สำเร็จ")
    except TelegramLoginError as e:
        flash(request, str(e), "error")
    return back("/telegram")


@app.post("/telegram/logout")
async def telegram_logout(request: Request):
    current_user(request, "admin")
    await telegram.logout()
    flash(request, "ออกจากระบบ Telegram แล้ว")
    return back("/telegram")


# ---------------------------------------------------------------- chats
@app.get("/chats")
async def chats_page(request: Request):
    user = current_user(request, "agent")
    with SessionLocal() as db:
        chats = list(db.scalars(select(Chat).order_by(Chat.monitored.desc(), Chat.title)))
        counts = dict(db.execute(select(Message.chat_id, func.count(Message.id)).group_by(Message.chat_id)).all())
    return render(request, "chats.html", user, chats=chats, counts=counts, connected=telegram.connected,
                  ai_errors=analyzer.last_error)


@app.post("/chats/sync")
async def chats_sync(request: Request):
    current_user(request, "admin")
    dialogs = await telegram.list_dialogs()
    with SessionLocal() as db:
        for d in dialogs:
            chat = db.get(Chat, d["id"])
            if chat is None:
                db.add(Chat(id=d["id"], title=d["title"], kind=d["kind"]))
            else:
                chat.title, chat.kind = d["title"], d["kind"]
        db.commit()
    flash(request, f"ดึงรายชื่อแชท {len(dialogs)} รายการ" if dialogs else "ยังไม่ได้เชื่อมต่อ Telegram",
          "ok" if dialogs else "error")
    return back("/chats")


@app.post("/chats/{chat_id}/toggle")
async def chats_toggle(request: Request, chat_id: int):
    current_user(request, "admin")
    with SessionLocal() as db:
        chat = db.get(Chat, chat_id)
        if chat:
            chat.monitored = not chat.monitored
            db.commit()
    telegram.reload_monitored()
    return back("/chats")


@app.post("/chats/{chat_id}/analyze")
async def chats_analyze(request: Request, chat_id: int):
    current_user(request, "agent")
    result = await analyzer.analyze(chat_id)
    flash(request, f"วิเคราะห์แล้ว: {result}", "error" if result.startswith("ผิดพลาด") else "ok")
    return back("/chats")


@app.get("/chats/{chat_id}")
async def chat_detail(request: Request, chat_id: int):
    user = current_user(request, "agent")
    with SessionLocal() as db:
        chat = db.get(Chat, chat_id)
        messages = list(db.scalars(
            select(Message).where(Message.chat_id == chat_id).order_by(Message.date.desc()).limit(200)
        ))[::-1]
    if not chat:
        return back("/chats")
    return render(request, "chat_detail.html", user, chat=chat, messages=messages)


# ---------------------------------------------------------------- replies (approval queue)
@app.get("/replies")
async def replies_page(request: Request, status: str = "pending"):
    user = current_user(request, "agent")
    with SessionLocal() as db:
        query = select(Reply).order_by(Reply.created_at.desc()).limit(100)
        if status != "all":
            query = query.where(Reply.status == status)
        replies = list(db.scalars(query))
        chat_titles = {c.id: c.title for c in db.scalars(select(Chat))}
        context = {}
        for r in replies:
            if r.status == "pending":
                context[r.id] = list(db.scalars(
                    select(Message).where(Message.chat_id == r.chat_id)
                    .order_by(Message.date.desc()).limit(8)
                ))[::-1]
    return render(request, "replies.html", user, replies=replies, status=status, chat_titles=chat_titles,
                  context=context)


def _load_pending(reply_id: int) -> Reply | None:
    with SessionLocal() as db:
        reply = db.get(Reply, reply_id)
    return reply if reply and reply.status in ("pending", "failed") else None


@app.post("/replies/{reply_id}/approve")
async def replies_approve(request: Request, reply_id: int, text: str = Form(...)):
    user = current_user(request, "agent")
    if not _load_pending(reply_id):
        flash(request, "ข้อความนี้ถูกดำเนินการไปแล้ว", "error")
        return back("/replies")
    text = text.strip()
    if not text:
        flash(request, "ข้อความว่าง ส่งไม่ได้", "error")
        return back("/replies")
    with SessionLocal() as db:
        reply = db.get(Reply, reply_id)
        reply.status, reply.final_text = "sending", text
        reply.decided_by, reply.decided_at = user.username, utcnow()
        db.commit()
        chat_id, reply_to = reply.chat_id, reply.reply_to_tg_id
    try:
        await telegram.send_reply(chat_id, text, reply_to)
        status, error = "sent", ""
        flash(request, "อนุมัติและส่งข้อความแล้ว")
    except TelegramLoginError as e:
        status, error = "failed", str(e)
        flash(request, f"ส่งข้อความไม่สำเร็จ: {e}", "error")
    except Exception as e:  # noqa: BLE001
        log.exception("send reply failed")
        status, error = "failed", str(e)
        flash(request, f"ส่งข้อความไม่สำเร็จ: {e}", "error")
    with SessionLocal() as db:
        reply = db.get(Reply, reply_id)
        reply.status, reply.error = status, error
        db.commit()
    return back("/replies")


@app.post("/replies/{reply_id}/reject")
async def replies_reject(request: Request, reply_id: int, reason: str = Form("")):
    user = current_user(request, "agent")
    with SessionLocal() as db:
        reply = db.get(Reply, reply_id)
        if reply and reply.status in ("pending", "failed"):
            reply.status, reply.reject_reason = "rejected", reason.strip()
            reply.decided_by, reply.decided_at = user.username, utcnow()
            db.commit()
            flash(request, "ปฏิเสธการส่งข้อความแล้ว (ไม่ได้ส่งถึงลูกค้า)")
    return back("/replies")


@app.post("/replies/{reply_id}/regenerate")
async def replies_regenerate(request: Request, reply_id: int, instruction: str = Form(...),
                             text: str = Form("")):
    current_user(request, "agent")
    reply = _load_pending(reply_id)
    if not reply:
        return back("/replies")
    with SessionLocal() as db:
        settings = get_settings(db)
        chat = db.get(Chat, reply.chat_id)
        history = list(db.scalars(
            select(Message).where(Message.chat_id == reply.chat_id).order_by(Message.date.desc()).limit(20)
        ))[::-1]
    try:
        new_text = await ai_service.rewrite_reply(
            settings, chat.title if chat else "", history, text or reply.final_text, instruction
        )
    except ai_service.AIError as e:
        flash(request, str(e), "error")
        return back("/replies")
    with SessionLocal() as db:
        db.get(Reply, reply_id).final_text = new_text
        db.commit()
    flash(request, "AI เขียนคำตอบใหม่แล้ว ตรวจสอบก่อนกดอนุมัติ")
    return back("/replies")


# ---------------------------------------------------------------- tickets
@app.get("/tickets")
async def tickets_page(request: Request, status: str = "active", category: str = ""):
    user = current_user(request, "programmer")
    with SessionLocal() as db:
        query = select(Ticket).order_by(Ticket.created_at.desc()).limit(300)
        if status == "active":
            query = query.where(Ticket.status.in_(analyzer.OPEN_STATUSES))
        elif status != "all":
            query = query.where(Ticket.status == status)
        if category:
            query = query.where(Ticket.category == category)
        tickets = list(db.scalars(query))
        chat_titles = {c.id: c.title for c in db.scalars(select(Chat))}
        users = {u.id: u.username for u in db.scalars(select(User))}
    return render(request, "tickets.html", user, tickets=tickets, status=status, category=category,
                  chat_titles=chat_titles, users=users)


@app.get("/tickets/{ticket_id}")
async def ticket_detail(request: Request, ticket_id: int):
    user = current_user(request, "programmer")
    with SessionLocal() as db:
        ticket = db.get(Ticket, ticket_id)
        if not ticket:
            return back("/tickets")
        events, attachments = list(ticket.events), list(ticket.attachments)
        chat = db.get(Chat, ticket.chat_id)
        users = list(db.scalars(select(User).order_by(User.username)))
        replies = list(db.scalars(select(Reply).where(Reply.ticket_id == ticket_id).order_by(Reply.created_at)))
    return render(request, "ticket_detail.html", user, ticket=ticket, events=events, attachments=attachments,
                  chat=chat, users=users, replies=replies)


@app.post("/tickets/{ticket_id}/update")
async def ticket_update(request: Request, ticket_id: int, status: str = Form(...), severity: str = Form(...),
                        assignee_id: str = Form(""), website_url: str = Form(""), title: str = Form(...)):
    user = current_user(request, "programmer")
    with SessionLocal() as db:
        ticket = db.get(Ticket, ticket_id)
        if ticket:
            changes = []
            if status in TICKET_STATUSES and status != ticket.status:
                changes.append(f"สถานะ: {TICKET_STATUSES[ticket.status]} → {TICKET_STATUSES[status]}")
                ticket.status = status
            if severity in SEVERITY_LABELS and severity != ticket.severity:
                changes.append(f"ความรุนแรง: {SEVERITY_LABELS[ticket.severity]} → {SEVERITY_LABELS[severity]}")
                ticket.severity = severity
            assignee = db.get(User, int(assignee_id)) if assignee_id.isdigit() else None
            new_assignee = assignee.id if assignee else None
            if new_assignee != ticket.assignee_id:
                changes.append(f"ผู้รับผิดชอบ: {assignee.username if assignee else '-'}")
                ticket.assignee_id = new_assignee
            ticket.website_url, ticket.title = website_url.strip(), title.strip() or ticket.title
            if changes:
                db.add(TicketEvent(ticket_id=ticket_id, kind="status", author=user.username, body="\n".join(changes)))
            db.commit()
            flash(request, "บันทึก ticket แล้ว")
    return back(f"/tickets/{ticket_id}")


@app.post("/tickets/{ticket_id}/note")
async def ticket_note(request: Request, ticket_id: int, body: str = Form(...)):
    user = current_user(request, "programmer")
    if body.strip():
        with SessionLocal() as db:
            if db.get(Ticket, ticket_id):
                db.add(TicketEvent(ticket_id=ticket_id, kind="note", author=user.username, body=body.strip()))
                db.get(Ticket, ticket_id).updated_at = utcnow()
                db.commit()
    return back(f"/tickets/{ticket_id}")


@app.post("/tickets/{ticket_id}/check-site")
async def ticket_check_site(request: Request, ticket_id: int):
    current_user(request, "programmer")
    result = await analyzer.run_site_check(ticket_id)
    if result is None:
        flash(request, "ticket นี้ไม่มีลิงก์เว็บไซต์", "error")
    else:
        flash(request, "เว็บไซต์เข้าได้ปกติ" if result["ok"] else f"เว็บไซต์เข้าไม่ได้: {result['error']}",
              "ok" if result["ok"] else "error")
    return back(f"/tickets/{ticket_id}")


@app.get("/media/{name}")
async def media(request: Request, name: str):
    current_user(request)
    path = (MEDIA_DIR / name).resolve()
    if path.parent != MEDIA_DIR.resolve() or not path.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path)


# ---------------------------------------------------------------- settings & users
@app.get("/settings")
async def settings_page(request: Request):
    user = current_user(request, "admin")
    with SessionLocal() as db:
        settings = get_settings(db)
        users = list(db.scalars(select(User).order_by(User.username)))
    return render(request, "settings.html", user, settings=settings, users=users)


@app.post("/settings")
async def settings_save(request: Request):
    current_user(request, "admin")
    form = await request.form()
    with SessionLocal() as db:
        for key in DEFAULT_SETTINGS:
            if key in ("auto_draft", "auto_ticket", "site_check"):
                value = "1" if form.get(key) else "0"
            elif key in ("debounce_seconds", "context_messages"):
                low, high = (0, 600) if key == "debounce_seconds" else (5, 100)
                value = str(int_setting({key: str(form.get(key, ""))}, key, low, high))
            elif key in form:
                value = str(form[key]).strip()
            else:
                continue
            row = db.get(Setting, key) or Setting(key=key)
            row.value = value
            db.merge(row)
        db.commit()
    flash(request, "บันทึกการตั้งค่าแล้ว")
    return back("/settings")


@app.post("/users")
async def users_create(request: Request, username: str = Form(...), password: str = Form(...),
                       role: str = Form(...)):
    current_user(request, "admin")
    username = username.strip()
    if role not in ROLES or not username or len(password) < 8:
        flash(request, "ข้อมูลไม่ครบ (รหัสผ่านอย่างน้อย 8 ตัวอักษร)", "error")
        return back("/settings")
    with SessionLocal() as db:
        if db.scalar(select(User).where(User.username == username)):
            flash(request, "มีชื่อผู้ใช้นี้แล้ว", "error")
            return back("/settings")
        db.add(User(username=username, password_hash=hash_password(password), role=role))
        db.commit()
    flash(request, f"เพิ่มผู้ใช้ {username} แล้ว")
    return back("/settings")


@app.post("/users/{user_id}/delete")
async def users_delete(request: Request, user_id: int):
    me = current_user(request, "admin")
    if user_id == me.id:
        flash(request, "ลบบัญชีตัวเองไม่ได้", "error")
        return back("/settings")
    with SessionLocal() as db:
        user = db.get(User, user_id)
        if user:
            for t in db.scalars(select(Ticket).where(Ticket.assignee_id == user_id)):
                t.assignee_id = None
            db.delete(user)
            db.commit()
    return back("/settings")


@app.get("/account")
async def account_page(request: Request):
    return render(request, "account.html", current_user(request))


@app.post("/account/password")
async def change_password(request: Request, current: str = Form(...), new: str = Form(...)):
    me = current_user(request)
    if not verify_password(current, me.password_hash) or len(new) < 8:
        flash(request, "รหัสผ่านเดิมไม่ถูกต้อง หรือรหัสใหม่สั้นกว่า 8 ตัวอักษร", "error")
    else:
        with SessionLocal() as db:
            db.get(User, me.id).password_hash = hash_password(new)
            db.commit()
        flash(request, "เปลี่ยนรหัสผ่านแล้ว")
    return back("/account")
