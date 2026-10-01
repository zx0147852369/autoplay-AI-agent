import asyncio
import json
import logging
import os
import sys
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from starlette.middleware.sessions import SessionMiddleware

from . import ai_service, analyzer, dev_bridge, quota
from .config import ADMIN_PASSWORD, ADMIN_USERNAME, DATA_DIR, DISPLAY_TZ, EPHEMERAL_STORAGE, MEDIA_DIR, SECRET_KEY
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
    TicketLink,
    User,
    get_settings,
    init_db,
    int_setting,
    utcnow,
)
from .notices import sync_resolved_notice
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
    log.info("เก็บข้อมูลที่ %s", DATA_DIR)
    if EPHEMERAL_STORAGE:
        log.warning("ข้อมูลไม่ได้อยู่ใน Railway Volume: ตั้งค่าและข้อมูลทั้งหมดจะหายเมื่อ deploy ใหม่")
    telegram.on_message = analyzer.schedule
    dev_bridge.configure()
    startup = asyncio.create_task(telegram.start_from_db())
    sweeper = asyncio.create_task(analyzer.sweeper())
    lag_watch = asyncio.create_task(_watch_loop_lag())
    yield
    startup.cancel()
    sweeper.cancel()
    lag_watch.cancel()
    await telegram.stop()


app = FastAPI(title="Telegram AI Support", lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY, same_site="lax", max_age=60 * 60 * 12)

# ---------------------------------------------------------------- วัดความเร็ว (ใช้หาสาเหตุเว็บช้า)
_loop_lag = {"max": 0.0, "since": time.monotonic()}


async def _watch_loop_lag() -> None:
    """ทุก 0.5 วินาที วัดว่า event loop ค้างนานเท่าไร (ถ้ามีงานแบบ sync บล็อก ทั้งเว็บจะช้าตาม)"""
    while True:
        start = time.monotonic()
        await asyncio.sleep(0.5)
        lag = time.monotonic() - start - 0.5
        if time.monotonic() - _loop_lag["since"] > 60:
            _loop_lag.update(max=0.0, since=time.monotonic())
        _loop_lag["max"] = max(_loop_lag["max"], lag)
        if lag > 1:
            log.warning("event loop ค้าง %.1f วินาที", lag)


@app.middleware("http")
async def _timing(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - started) * 1000
    # app = เวลาที่ระบบใช้จริง, lag = event loop ค้างสูงสุดใน 1 นาทีล่าสุด (ดูได้ใน DevTools > Network > Timing)
    response.headers["Server-Timing"] = f"app;dur={ms:.0f}, lag;dur={_loop_lag['max'] * 1000:.0f}"
    if ms > 1000:
        log.warning("ช้า %s %s %.0f ms", request.method, request.url.path, ms)
    return response
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=APP_DIR / "templates")


def _localtime(dt):
    if not dt:
        return "-"
    return dt.replace(tzinfo=timezone.utc).astimezone(DISPLAY_TZ).strftime("%d/%m/%Y %H:%M")


THAI_DAYS = ["จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์"]
THAI_MONTHS = ["มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน", "กรกฎาคม",
               "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม"]


def thai_today() -> str:
    d = utcnow().replace(tzinfo=timezone.utc).astimezone(DISPLAY_TZ)
    return f"วัน{THAI_DAYS[d.weekday()]}ที่ {d.day} {THAI_MONTHS[d.month - 1]} {d.year + 543}"


def _iso_localtime(value: str) -> str:
    try:
        return _localtime(datetime.fromisoformat(value))
    except (TypeError, ValueError):
        return "-"


templates.env.filters["localtime"] = _localtime
templates.env.filters["localtime_hm"] = lambda dt: dt.replace(tzinfo=timezone.utc).astimezone(DISPLAY_TZ).strftime("%H:%M") if dt else "-"
templates.env.filters["isolocal"] = _iso_localtime
templates.env.filters["fromjson"] = lambda s: json.loads(s) if s else None
templates.env.globals.update(
    CATEGORIES=ai_service.CATEGORIES, TICKET_STATUSES=TICKET_STATUSES, SEVERITY_LABELS=SEVERITY_LABELS,
    REPLY_STATUSES=REPLY_STATUSES, ROLES=ROLES, ENV_ADMIN=ADMIN_USERNAME, EPHEMERAL_STORAGE=EPHEMERAL_STORAGE,
    GEMINI_MODELS=ai_service.GEMINI_MODELS, CLAUDE_MODELS=ai_service.CLAUDE_MODELS,
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
    # ทุกบทบาทเข้าหน้า Tickets ได้ แต่กันวนซ้ำไว้เผื่อหน้า Tickets เองไม่มีสิทธิ์
    target = "/account" if request.url.path.startswith("/tickets") else "/tickets"
    return RedirectResponse(target, status_code=303)


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
    if user and "quota_info" not in context:
        try:
            with SessionLocal() as db:
                context["quota_info"] = quota.snapshot(get_settings(db))
        except Exception:  # noqa: BLE001 - หลอดโควตาพังต้องไม่ทำให้ทั้งหน้าพัง
            log.exception("quota snapshot failed")
    return templates.TemplateResponse(request, name, context)


def back(url: str) -> RedirectResponse:
    return RedirectResponse(url, status_code=303)


# ---------------------------------------------------------------- login
@app.get("/login")
async def login_page(request: Request):
    # หน้า login แสดงข้อผิดพลาดในฟอร์มเลย แทนการเด้งเป็น toast
    flashes = request.session.pop("flash", [])
    return render(request, "login.html", None, need_setup=not ADMIN_PASSWORD,
                  login_errors=list(dict.fromkeys(f["message"] for f in flashes if f["kind"] == "error")))


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
def local_day_start_utc():
    """เวลาเริ่มต้นของ "วันนี้" ตามเวลาไทย แปลงเป็น UTC (naive) สำหรับค้นในฐานข้อมูล"""
    now_local = utcnow().replace(tzinfo=timezone.utc).astimezone(DISPLAY_TZ)
    start = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    return start.astimezone(timezone.utc).replace(tzinfo=None)


def model_key_ready(model: str) -> bool:
    if ai_service.is_gemini(model):
        return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    return bool(os.getenv("ANTHROPIC_API_KEY"))


@app.get("/")
async def dashboard(request: Request):
    user = current_user(request, "agent")
    day = local_day_start_utc()
    with SessionLocal() as db:
        settings = get_settings(db)
        account = db.get(TelegramAccount, 1)
        pending = db.scalar(select(func.count(Reply.id)).where(Reply.status == "pending"))
        open_by_cat = dict(db.execute(
            select(Ticket.category, func.count(Ticket.id))
            .where(Ticket.status.in_(analyzer.OPEN_STATUSES)).group_by(Ticket.category)
        ).all())
        by_status = dict(db.execute(select(Ticket.status, func.count(Ticket.id)).group_by(Ticket.status)).all())
        recent = list(db.scalars(select(Ticket).order_by(Ticket.created_at.desc()).limit(6)))
        pending_list = list(db.scalars(
            select(Reply).where(Reply.status == "pending").order_by(Reply.created_at.desc()).limit(5)))
        monitored = db.scalar(select(func.count(Chat.id)).where(Chat.monitored))
        chat_titles = {c.id: c.title for c in db.scalars(select(Chat))}
        dev_title = chat_titles.get(telegram.dev_group_id, "") if telegram.dev_group_id else ""
        today = {
            "messages": db.scalar(select(func.count(Message.id))
                                  .where(Message.date >= day, Message.is_outgoing.is_(False))),
            "tickets": db.scalar(select(func.count(Ticket.id)).where(Ticket.created_at >= day)),
            "resolved": db.scalar(select(func.count(Ticket.id))
                                  .where(Ticket.updated_at >= day, Ticket.status.in_(("resolved", "closed")))),
            "sent": db.scalar(select(func.count(Reply.id))
                              .where(Reply.status == "sent", Reply.decided_at >= day)),
        }
    model = settings.get("ai_model", "")
    model_label = {**ai_service.GEMINI_MODELS, **ai_service.CLAUDE_MODELS}.get(model, model)
    setup = [
        ("เชื่อมต่อบัญชี Telegram", telegram.connected, "/telegram"),
        ("เลือกแชทลูกค้าที่จะติดตาม", bool(monitored), "/chats"),
        ("ใส่ API key ของโมเดล AI", model_key_ready(model), "/settings"),
        ("กรอกข้อมูลธุรกิจและคำตอบมาตรฐาน",
         settings.get("knowledge_base") != DEFAULT_SETTINGS["knowledge_base"], "/settings"),
    ]
    system = [
        ("Telegram", telegram.connected, account.me_name or "ยังไม่ได้เชื่อมต่อ"),
        ("โมเดล AI", model_key_ready(model), model_label + ("" if model_key_ready(model) else " · ยังไม่มี API key")
         + (f" · ตอนนี้ใช้รุ่นสำรอง {ai_service.last_model_used} (รุ่นหลักล่มชั่วคราว)"
            if ai_service.last_model_used and ai_service.last_model_used != model else "")),
        ("การเก็บข้อมูล", not EPHEMERAL_STORAGE, "ถาวร (Volume)" if not EPHEMERAL_STORAGE else "ชั่วคราว หายเมื่อ deploy"),
        ("กลุ่มโปรแกรมเมอร์", bool(dev_title) and not dev_bridge.last_error,
         dev_bridge.last_error or dev_title or "ยังไม่ได้เลือกกลุ่ม (ตั้งค่า → กลุ่มแจ้งปัญหาโปรแกรมเมอร์)"),
        ("ร่างคำตอบอัตโนมัติ", settings.get("auto_draft") == "1", "เปิด" if settings.get("auto_draft") == "1" else "ปิด"),
        ("เปิด ticket อัตโนมัติ", settings.get("auto_ticket") == "1", "เปิด" if settings.get("auto_ticket") == "1" else "ปิด"),
        ("ตรวจเว็บไซต์อัตโนมัติ", settings.get("site_check") == "1", "เปิด" if settings.get("site_check") == "1" else "ปิด"),
    ]
    return render(request, "dashboard.html", user, account=account, pending=pending, open_by_cat=open_by_cat,
                  by_status=by_status, recent=recent, pending_list=pending_list, monitored=monitored,
                  chat_titles=chat_titles, ai_errors=analyzer.last_error, connected=telegram.connected,
                  today=thai_today(), today_counts=today, setup=setup, system=system,
                  quota_info=quota.snapshot(settings),
                  setup_done=sum(1 for item in setup if item[1]))


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
    day = local_day_start_utc()
    with SessionLocal() as db:
        chats = list(db.scalars(select(Chat).order_by(Chat.monitored.desc(), Chat.title)))
        counts = dict(db.execute(select(Message.chat_id, func.count(Message.id)).group_by(Message.chat_id)).all())
        today_counts = dict(db.execute(
            select(Message.chat_id, func.count(Message.id))
            .where(Message.date >= day, Message.is_outgoing.is_(False)).group_by(Message.chat_id)).all())
        last_at = dict(db.execute(select(Message.chat_id, func.max(Message.date)).group_by(Message.chat_id)).all())
        open_tickets = dict(db.execute(
            select(Ticket.chat_id, func.count(Ticket.id))
            .where(Ticket.status.in_(analyzer.OPEN_STATUSES)).group_by(Ticket.chat_id)).all())
        pending = dict(db.execute(
            select(Reply.chat_id, func.count(Reply.id)).where(Reply.status == "pending").group_by(Reply.chat_id)).all())
        unanalyzed = dict(db.execute(
            select(Message.chat_id, func.count(Message.id))
            .where(Message.analyzed.is_(False), Message.is_outgoing.is_(False)).group_by(Message.chat_id)).all())
    summary = {
        "total": len(chats),
        "monitored": sum(1 for c in chats if c.monitored),
        "today": sum(today_counts.values()),
        "open_tickets": sum(open_tickets.values()),
    }
    return render(request, "chats.html", user, chats=chats, counts=counts, today_counts=today_counts,
                  last_at=last_at, open_tickets=open_tickets, pending=pending, summary=summary, unanalyzed=unanalyzed,
                  dev_group_id=telegram.dev_group_id,
                  connected=telegram.connected, ai_errors=analyzer.last_error)


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
        dev_set = get_settings(db).get("dev_group_id")
        match = next((d for d in dialogs if d["kind"] != "user" and "autopay support" in d["title"].lower()), None)
        if not dev_set and match:
            db.merge(Setting(key="dev_group_id", value=str(match["id"])))
            db.commit()
            flash(request, f"ตั้งกลุ่ม \"{match['title']}\" เป็นกลุ่มแจ้งปัญหาโปรแกรมเมอร์แล้ว (เปลี่ยนได้ที่หน้าตั้งค่า)")
    dev_bridge.configure()
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
        enabled = bool(chat and chat.monitored)
        limit = int_setting(get_settings(db), "context_messages", 5, 100)
    telegram.reload_monitored()
    if enabled and telegram.connected:
        try:
            waiting = await telegram.backfill(chat_id, limit=limit)
        except Exception as e:  # noqa: BLE001
            log.exception("backfill failed")
            flash(request, f"เปิดติดตามแล้ว แต่ดึงข้อความย้อนหลังไม่สำเร็จ: {e}", "error")
            return back("/chats")
        if waiting:
            analyzer.schedule(chat_id)
            flash(request, f"เปิดติดตามแล้ว พบข้อความลูกค้าที่ยังไม่ได้ตอบ {waiting} ข้อความ กำลังวิเคราะห์ให้อัตโนมัติ")
        else:
            flash(request, "เปิดติดตามแล้ว ระบบจะวิเคราะห์ข้อความใหม่ให้อัตโนมัติ")
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
        status_counts = dict(db.execute(select(Reply.status, func.count(Reply.id)).group_by(Reply.status)).all())
        chat_titles = {c.id: c.title for c in db.scalars(select(Chat))}
        context = {}
        for r in replies:
            if r.status == "pending":
                context[r.id] = list(db.scalars(
                    select(Message).where(Message.chat_id == r.chat_id)
                    .order_by(Message.date.desc()).limit(8)
                ))[::-1]
    status_counts["all"] = sum(status_counts.values())
    return render(request, "replies.html", user, replies=replies, status=status, chat_titles=chat_titles,
                  context=context, status_counts=status_counts)


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
        if status == "sent" and reply.ticket_id and db.get(Ticket, reply.ticket_id):
            label = "แจ้งลูกค้าว่าแก้ไขเรียบร้อยแล้ว" if reply.kind == "resolved" else "ตอบลูกค้า"
            db.add(TicketEvent(ticket_id=reply.ticket_id, kind="status", author=user.username,
                               body=f"{label}: {text}"))
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
async def tickets_page(request: Request, status: str = "active", category: str = "", q: str = ""):
    user = current_user(request, "programmer", "agent")
    q = q.strip()
    with SessionLocal() as db:
        query = select(Ticket).order_by(Ticket.created_at.desc()).limit(300)
        if status == "active":
            query = query.where(Ticket.status.in_(analyzer.OPEN_STATUSES))
        elif status != "all":
            query = query.where(Ticket.status == status)
        if category:
            query = query.where(Ticket.category == category)
        if q:
            like = f"%{q}%"
            query = query.where(
                Ticket.title.ilike(like) | Ticket.customer_name.ilike(like) | Ticket.summary.ilike(like)
                | Ticket.website_url.ilike(like) | (Ticket.id == (int(q.lstrip("#")) if q.lstrip("#").isdigit() else -1))
            )
        tickets = list(db.scalars(query))
        status_counts = dict(db.execute(select(Ticket.status, func.count(Ticket.id)).group_by(Ticket.status)).all())
        chat_titles = {c.id: c.title for c in db.scalars(select(Chat))}
        users = {u.id: u.username for u in db.scalars(select(User))}
    status_counts["active"] = sum(status_counts.get(k, 0) for k in analyzer.OPEN_STATUSES)
    status_counts["all"] = sum(v for k, v in status_counts.items() if k in TICKET_STATUSES)
    return render(request, "tickets.html", user, tickets=tickets, status=status, category=category, q=q,
                  chat_titles=chat_titles, users=users, status_counts=status_counts)


@app.get("/tickets/{ticket_id}")
async def ticket_detail(request: Request, ticket_id: int):
    user = current_user(request, "programmer", "agent")
    with SessionLocal() as db:
        ticket = db.get(Ticket, ticket_id)
        if not ticket:
            return back("/tickets")
        events, attachments = list(ticket.events), list(ticket.attachments)
        chat = db.get(Chat, ticket.chat_id)
        users = list(db.scalars(select(User).order_by(User.username)))
        replies = list(db.scalars(select(Reply).where(Reply.ticket_id == ticket_id).order_by(Reply.created_at)))
        dev_posted = db.scalar(select(func.count(TicketLink.id)).where(TicketLink.ticket_id == ticket_id))
    return render(request, "ticket_detail.html", user, ticket=ticket, timeline=build_timeline(events, attachments),
                  attachments=attachments, chat=chat, users=users, replies=replies, dev_posted=dev_posted,
                  dev_group_set=bool(telegram.dev_group_id))


def build_timeline(events, attachments) -> list[dict]:
    """จัดประวัติ ticket: ข้อความลูกค้าที่ส่งติดกัน (คนเดียวกัน ห่างกันไม่เกิน 10 นาที) รวมเป็นก้อนเดียว พร้อมรูป"""
    # ประวัติเก่าที่ยังไม่มี media_path: จับคู่รูปแนบตามลำดับ (คำบรรยายรูป = ข้อความเดียวกัน)
    unused = list(attachments)
    groups: list[dict] = []
    for e in events:
        media = e.media_path
        if e.kind == "customer_message" and not media:
            caption = "" if e.body == "(รูปภาพ)" else e.body
            # รูปเปล่า: คำบรรยายว่าง / รูปพร้อมข้อความ: คำบรรยายตรงกับข้อความ
            match = next((a for a in unused if (a.caption or "") == caption), None) if (caption or e.body) else None
            if match:
                unused.remove(match)
                media = match.media_path
        elif media:
            unused = [a for a in unused if a.media_path != media]
        body = "" if e.body == "(รูปภาพ)" and media else e.body
        last = groups[-1] if groups else None
        if (last and e.kind == "customer_message" and last["kind"] == e.kind and last["author"] == e.author
                and (e.created_at - last["last_at"]).total_seconds() <= 600):
            last["items"].append({"body": body, "media": media})
            last["last_at"] = e.created_at
            continue
        groups.append({"kind": e.kind, "author": e.author, "created_at": e.created_at, "last_at": e.created_at,
                       "items": [{"body": body, "media": media}]})
    return groups


@app.post("/tickets/{ticket_id}/post-dev")
async def ticket_post_dev(request: Request, ticket_id: int):
    current_user(request, "agent")
    try:
        result = await dev_bridge.post_ticket(ticket_id, force=True)
    except Exception as e:  # noqa: BLE001
        log.exception("post ticket to dev group failed")
        result = f"ส่งไม่สำเร็จ: {e}"
    flash(request, result or "ส่งแล้ว", "ok" if result.startswith("ส่งเข้ากลุ่ม") else "error")
    return back(f"/tickets/{ticket_id}")


@app.post("/tickets/{ticket_id}/update")
async def ticket_update(request: Request, ticket_id: int, status: str = Form(...), severity: str = Form(...),
                        assignee_id: str = Form(""), website_url: str = Form(""), title: str = Form(...)):
    user = current_user(request, "programmer", "agent")
    with SessionLocal() as db:
        ticket = db.get(Ticket, ticket_id)
        if ticket:
            changes = []
            old_status = ticket.status
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
            notice = sync_resolved_notice(db, ticket, user.username) if ticket.status != old_status else ""
            if notice == "created":
                flash(request, "ร่างข้อความแจ้งลูกค้าว่าแก้ไขเรียบร้อยแล้ว รออนุมัติที่หน้า \"รออนุมัติ\"")
            elif notice == "cancelled":
                flash(request, "ยกเลิกข้อความแจ้งลูกค้าที่ยังไม่ได้ส่ง เพราะ ticket ยังไม่ได้แก้ไขเสร็จ")
    return back(f"/tickets/{ticket_id}")




def delete_tickets(db, ticket_ids: list[int]) -> int:
    """ลบ ticket พร้อมประวัติ รูปแนบ และลิงก์ในกลุ่มโปรแกรมเมอร์
    ข้อความถึงลูกค้าที่ผูกกับ ticket: ร่างแจ้งความคืบหน้าที่ยังไม่ส่งจะถูกยกเลิก ส่วนข้อความอื่นยังเก็บไว้เป็นประวัติ
    (ไฟล์รูปไม่ลบ เพราะยังใช้แสดงในประวัติแชท)"""
    deleted = 0
    for ticket_id in ticket_ids:
        ticket = db.get(Ticket, ticket_id)
        if not ticket:
            continue
        for r in db.scalars(select(Reply).where(Reply.ticket_id == ticket_id)):
            if r.kind != "ai" and r.status in ("pending", "failed"):
                r.status = "superseded"
            r.ticket_id = None
        for link in db.scalars(select(TicketLink).where(TicketLink.ticket_id == ticket_id)):
            db.delete(link)
        db.delete(ticket)
        deleted += 1
    db.commit()
    return deleted


@app.post("/tickets/{ticket_id}/delete")
async def ticket_delete(request: Request, ticket_id: int):
    current_user(request, "admin")
    with SessionLocal() as db:
        n = delete_tickets(db, [ticket_id])
    flash(request, f"ลบ ticket #{ticket_id} แล้ว" if n else "ไม่พบ ticket นี้", "ok" if n else "error")
    return back("/tickets")


@app.post("/tickets/delete")
async def tickets_bulk_delete(request: Request):
    current_user(request, "admin")
    form = await request.form()
    ids = [int(v) for v in form.getlist("ids") if str(v).isdigit()]
    if not ids:
        flash(request, "ยังไม่ได้เลือก ticket", "error")
        return back("/tickets")
    with SessionLocal() as db:
        n = delete_tickets(db, ids)
    flash(request, f"ลบ ticket แล้ว {n} รายการ")
    return back("/tickets")


@app.post("/tickets/{ticket_id}/note")
async def ticket_note(request: Request, ticket_id: int, body: str = Form(...)):
    user = current_user(request, "programmer", "agent")
    if body.strip():
        with SessionLocal() as db:
            if db.get(Ticket, ticket_id):
                db.add(TicketEvent(ticket_id=ticket_id, kind="note", author=user.username, body=body.strip()))
                db.get(Ticket, ticket_id).updated_at = utcnow()
                db.commit()
    return back(f"/tickets/{ticket_id}")


@app.post("/tickets/{ticket_id}/check-site")
async def ticket_check_site(request: Request, ticket_id: int):
    current_user(request, "programmer", "agent")
    result = await analyzer.run_site_check(ticket_id)
    if result is None:
        flash(request, "ticket นี้ไม่มีลิงก์เว็บไซต์", "error")
    else:
        if not result["ok"]:
            flash(request, f"เว็บไซต์เข้าไม่ได้: {result['error']}", "error")
        elif result.get("other_host"):
            flash(request, f"เว็บไซต์ถูกพาไปโดเมนอื่น ({result.get('final_host')}) โดเมนอาจหมดอายุ", "error")
        else:
            flash(request, "เว็บไซต์เข้าได้ปกติ")
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
    with SessionLocal() as db:
        groups = list(db.scalars(select(Chat).where(Chat.kind != "user").order_by(Chat.title)))
    suggested = next((g for g in groups if "autopay support" in g.title.lower()), None)
    return render(request, "settings.html", user, settings=settings, users=users, groups=groups, suggested=suggested,
                  limits=quota.limits(settings),
                  has_gemini_key=bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
                  has_claude_key=bool(os.getenv("ANTHROPIC_API_KEY")))


@app.post("/settings")
async def settings_save(request: Request):
    current_user(request, "admin")
    form = await request.form()
    with SessionLocal() as db:
        for key in DEFAULT_SETTINGS:
            if key in ("auto_draft", "auto_ticket", "site_check", "ask_link", "notify_resolved", "dev_forward", "dev_watch"):
                value = "1" if form.get(key) else "0"
            elif key == "gemini_limits":
                limits = {}
                for model in quota.DEFAULT_LIMITS:
                    entry = {}
                    for field in ("rpm", "rpd"):
                        raw = str(form.get(f"{field}__{model}", "")).strip()
                        if raw.isdigit():
                            entry[field] = int(raw)
                    if entry:
                        limits[model] = entry
                value = json.dumps(limits)
            elif key == "dev_group_id":
                value = str(form.get(key, "")).strip()
                if value and not value.lstrip("-").isdigit():
                    continue
            elif key == "ai_model":
                value = str(form.get(key, ""))
                if value not in ai_service.GEMINI_MODELS and value not in ai_service.CLAUDE_MODELS:
                    continue
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
    dev_bridge.configure()
    if telegram.dev_group_id and telegram.connected:
        dev_bridge.schedule_backlog()
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
