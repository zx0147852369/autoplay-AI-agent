from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    inspect,
    select,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from .config import DATABASE_URL

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def utcnow() -> datetime:
    """เก็บเวลาเป็น UTC แบบ naive ในฐานข้อมูล"""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True)
    password_hash: Mapped[str] = mapped_column(String(256))
    # admin = ทุกอย่าง, agent = อนุมัติข้อความ + ticket, programmer = ticket อย่างเดียว
    role: Mapped[str] = mapped_column(String(16), default="agent")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Setting(Base):
    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(Text, default="")


class TelegramAccount(Base):
    """บัญชี Telegram ของผู้ใช้ (มีได้บัญชีเดียวต่อระบบ, id = 1)"""

    __tablename__ = "telegram_account"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    api_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    api_hash_enc: Mapped[str] = mapped_column(Text, default="")
    phone: Mapped[str] = mapped_column(String(32), default="")
    session_enc: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(32), default="disconnected")
    me_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    me_name: Mapped[str] = mapped_column(String(128), default="")
    last_error: Mapped[str] = mapped_column(Text, default="")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class Chat(Base):
    __tablename__ = "chats"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)  # Telegram chat id (marked)
    title: Mapped[str] = mapped_column(String(256), default="")
    kind: Mapped[str] = mapped_column(String(16), default="group")  # group / channel / user
    monitored: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chat_id: Mapped[int] = mapped_column(BigInteger, index=True)
    tg_message_id: Mapped[int] = mapped_column(Integer)
    sender_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    sender_name: Mapped[str] = mapped_column(String(256), default="")
    is_outgoing: Mapped[bool] = mapped_column(Boolean, default=False)
    text: Mapped[str] = mapped_column(Text, default="")
    media_path: Mapped[str] = mapped_column(String(512), default="")  # ชื่อไฟล์ใน data/media
    date: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    analyzed: Mapped[bool] = mapped_column(Boolean, default=False)


class Ticket(Base):
    __tablename__ = "tickets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chat_id: Mapped[int] = mapped_column(BigInteger, index=True)
    category: Mapped[str] = mapped_column(String(32), default="other")
    title: Mapped[str] = mapped_column(String(256), default="")
    summary: Mapped[str] = mapped_column(Text, default="")
    severity: Mapped[str] = mapped_column(String(16), default="medium")
    status: Mapped[str] = mapped_column(String(16), default="open")
    customer_name: Mapped[str] = mapped_column(String(256), default="")
    website_url: Mapped[str] = mapped_column(String(1024), default="")
    site_check: Mapped[str] = mapped_column(Text, default="")  # JSON ผลตรวจเว็บไซต์ล่าสุด
    assignee_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    assignee: Mapped[User | None] = relationship()
    events: Mapped[list["TicketEvent"]] = relationship(
        back_populates="ticket", order_by="TicketEvent.created_at", cascade="all, delete-orphan"
    )
    attachments: Mapped[list["TicketAttachment"]] = relationship(
        back_populates="ticket", cascade="all, delete-orphan"
    )


class TicketEvent(Base):
    __tablename__ = "ticket_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticket_id: Mapped[int] = mapped_column(ForeignKey("tickets.id"), index=True)
    kind: Mapped[str] = mapped_column(String(24))  # customer_message / ai_summary / note / status
    author: Mapped[str] = mapped_column(String(128), default="")
    body: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    ticket: Mapped[Ticket] = relationship(back_populates="events")


class TicketAttachment(Base):
    __tablename__ = "ticket_attachments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticket_id: Mapped[int] = mapped_column(ForeignKey("tickets.id"), index=True)
    media_path: Mapped[str] = mapped_column(String(512))
    caption: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    ticket: Mapped[Ticket] = relationship(back_populates="attachments")


class Reply(Base):
    """ข้อความตอบกลับที่ AI ร่างไว้ รอแอดมินอนุมัติก่อนส่ง"""

    __tablename__ = "replies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chat_id: Mapped[int] = mapped_column(BigInteger, index=True)
    reply_to_tg_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ai_text: Mapped[str] = mapped_column(Text, default="")
    final_text: Mapped[str] = mapped_column(Text, default="")
    note: Mapped[str] = mapped_column(Text, default="")  # เหตุผล/บันทึกจาก AI ถึงแอดมิน
    # ai = ร่างคำตอบจาก AI, resolved = แจ้งลูกค้าว่าแก้ไขปัญหาเรียบร้อยแล้ว
    kind: Mapped[str] = mapped_column(String(16), default="ai", server_default="ai")
    # pending / sending / sent / rejected / failed / superseded
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    ticket_id: Mapped[int | None] = mapped_column(ForeignKey("tickets.id"), nullable=True)
    decided_by: Mapped[str] = mapped_column(String(64), default="")
    decided_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    reject_reason: Mapped[str] = mapped_column(Text, default="")
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


DEFAULT_SETTINGS = {
    "ai_model": "gemini-3.8-flash",
    "ai_effort": "medium",
    "business_context": (
        "เราเป็นทีมซัพพอร์ตของเว็บไซต์ให้บริการลูกค้า ลูกค้าจะทักเข้ามาในกลุ่ม Telegram "
        "เพื่อสอบถามหรือแจ้งปัญหา เช่น เว็บไซต์เข้าไม่ได้ เข้าสู่ระบบไม่ได้ ฝากเงินแล้วยอดไม่เข้าอัตโนมัติ"
    ),
    "knowledge_base": (
        "- ถ้าฝากเงินแล้วยอดไม่เข้า ให้ขอสลิปการโอน และยูสเซอร์ของลูกค้า\n"
        "- ถ้าเข้าสู่ระบบไม่ได้ ให้ขอยูสเซอร์ และภาพหน้าจอข้อความ error\n"
        "- ถ้าเว็บไซต์เข้าไม่ได้ ให้ขอลิงก์ที่ลูกค้าใช้ และภาพหน้าจอ"
    ),
    "reply_style": "สุภาพ เป็นกันเอง กระชับ ลงท้ายด้วย ค่ะ",
    "debounce_seconds": "15",
    "context_messages": "20",
    "auto_draft": "1",
    "auto_ticket": "1",
    "site_check": "1",
    "notify_resolved": "1",
    "resolved_message": (
        "สวัสดีค่ะ คุณ{customer} ปัญหา \"{title}\" ที่แจ้งไว้ ทีมงานได้แก้ไขเรียบร้อยแล้วค่ะ "
        "รบกวนลองใช้งานอีกครั้ง หากยังพบปัญหาแจ้งทีมงานได้เลยนะคะ ขอบคุณค่ะ"
    ),
}


def _add_missing_columns() -> None:
    """ฐานข้อมูลเก่า (อยู่ใน Volume) จะไม่มีคอลัมน์ที่เพิ่มทีหลัง -> เพิ่มให้อัตโนมัติ"""
    inspector = inspect(engine)
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if not inspector.has_table(table.name):
                continue
            existing = {c["name"] for c in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in existing:
                    continue
                ddl = f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {column.type.compile(engine.dialect)}'
                if column.server_default is not None:
                    ddl += f" DEFAULT '{column.server_default.arg}'"
                conn.execute(text(ddl))


def init_db() -> None:
    Base.metadata.create_all(engine)
    _add_missing_columns()
    with SessionLocal() as db:
        for key, value in DEFAULT_SETTINGS.items():
            if db.get(Setting, key) is None:
                db.add(Setting(key=key, value=value))
        if db.get(TelegramAccount, 1) is None:
            db.add(TelegramAccount(id=1))
        db.commit()


def int_setting(settings: dict[str, str], key: str, low: int, high: int) -> int:
    """อ่านค่าตัวเลขจากตั้งค่า ถ้าผิดรูปแบบใช้ค่าเริ่มต้น และบังคับให้อยู่ในช่วง"""
    try:
        value = int(settings.get(key) or DEFAULT_SETTINGS[key])
    except ValueError:
        value = int(DEFAULT_SETTINGS[key])
    return max(low, min(high, value))


def get_settings(db) -> dict[str, str]:
    values = dict(DEFAULT_SETTINGS)
    for row in db.scalars(select(Setting)):
        values[row.key] = row.value
    return values
