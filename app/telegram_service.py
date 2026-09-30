"""เชื่อมต่อบัญชี Telegram ของผู้ใช้ (userbot) ด้วย Telethon"""

import logging

from sqlalchemy import select
from telethon import TelegramClient, events
from telethon.errors import (
    FloodWaitError,
    PasswordHashInvalidError,
    PhoneCodeExpiredError,
    PhoneCodeInvalidError,
    PhoneNumberInvalidError,
    RPCError,
    SessionPasswordNeededError,
)
from telethon.sessions import StringSession

from .config import MEDIA_DIR
from .database import Chat, Message, SessionLocal, TelegramAccount, utcnow
from .security import decrypt, encrypt

log = logging.getLogger(__name__)

IMAGE_MIME_EXT = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp", "image/gif": ".gif"}


class TelegramLoginError(Exception):
    pass


class TelegramService:
    def __init__(self) -> None:
        self.client: TelegramClient | None = None
        self._pending_login: dict | None = None  # เก็บ client ระหว่างขั้นตอนขอรหัส -> ยืนยันรหัส
        self._monitored: set[int] = set()
        self.on_message = None  # callback(message_id) ถูกตั้งจาก analyzer

    # ------------------------------------------------------------------ state
    @property
    def connected(self) -> bool:
        return self.client is not None and self.client.is_connected()

    def reload_monitored(self) -> None:
        with SessionLocal() as db:
            self._monitored = set(db.scalars(select(Chat.id).where(Chat.monitored)))

    def _set_status(self, status: str, error: str = "", **fields) -> None:
        with SessionLocal() as db:
            acc = db.get(TelegramAccount, 1)
            acc.status = status
            acc.last_error = error
            for k, v in fields.items():
                setattr(acc, k, v)
            db.commit()

    # ------------------------------------------------------------------ startup
    async def start_from_db(self) -> None:
        """เชื่อมต่ออัตโนมัติเมื่อเปิดโปรแกรม ถ้าเคยล็อกอินไว้แล้ว"""
        with SessionLocal() as db:
            acc = db.get(TelegramAccount, 1)
            api_id, api_hash, session = acc.api_id, decrypt(acc.api_hash_enc), decrypt(acc.session_enc)
        if not (api_id and api_hash and session):
            return
        client = TelegramClient(StringSession(session), api_id, api_hash)
        try:
            await client.connect()
            if not await client.is_user_authorized():
                await client.disconnect()
                self._set_status("disconnected", "เซสชันหมดอายุ กรุณาเข้าสู่ระบบ Telegram ใหม่", session_enc="")
                return
            await self._activate(client)
        except (OSError, RPCError) as e:
            log.exception("connect telegram failed")
            self._set_status("error", f"เชื่อมต่อ Telegram ไม่ได้: {e}")

    async def _activate(self, client: TelegramClient) -> None:
        self.client = client
        me = await client.get_me()
        name = " ".join(filter(None, [me.first_name, me.last_name])) or me.username or str(me.id)
        self._set_status("connected", me_id=me.id, me_name=name,
                         session_enc=encrypt(client.session.save()))
        self.reload_monitored()
        client.add_event_handler(self._handle_new_message, events.NewMessage())
        # โหลดรายชื่อแชทไว้ในแคช เพื่อให้ส่งข้อความหา chat id ได้หลังรีสตาร์ท
        await client.get_dialogs()
        log.info("Telegram connected as %s", name)

    async def stop(self) -> None:
        if self.client:
            await self.client.disconnect()
        if self._pending_login:
            await self._pending_login["client"].disconnect()

    # ------------------------------------------------------------------ login
    async def send_code(self, api_id: int, api_hash: str, phone: str) -> None:
        if self._pending_login:
            await self._pending_login["client"].disconnect()
            self._pending_login = None
        client = TelegramClient(StringSession(), api_id, api_hash)
        await client.connect()
        try:
            sent = await client.send_code_request(phone)
        except PhoneNumberInvalidError as e:
            await client.disconnect()
            raise TelegramLoginError("เบอร์โทรศัพท์ไม่ถูกต้อง (ใส่รูปแบบ +66xxxxxxxxx)") from e
        except FloodWaitError as e:
            await client.disconnect()
            raise TelegramLoginError(f"ขอรหัสบ่อยเกินไป กรุณารอ {e.seconds} วินาที") from e
        except RPCError as e:
            await client.disconnect()
            raise TelegramLoginError(f"ขอรหัสไม่สำเร็จ: {e}") from e
        self._pending_login = {"client": client, "phone": phone, "hash": sent.phone_code_hash}
        self._set_status("code_sent", api_id=api_id, api_hash_enc=encrypt(api_hash), phone=phone)

    async def verify_code(self, code: str) -> str:
        """คืนค่า 'connected' หรือ 'password_needed' (บัญชีเปิดยืนยันสองขั้นตอน)"""
        pending = self._require_pending()
        try:
            await pending["client"].sign_in(pending["phone"], code.strip(), phone_code_hash=pending["hash"])
        except SessionPasswordNeededError:
            self._set_status("password_needed")
            return "password_needed"
        except PhoneCodeInvalidError as e:
            raise TelegramLoginError("รหัสยืนยันไม่ถูกต้อง") from e
        except PhoneCodeExpiredError as e:
            raise TelegramLoginError("รหัสยืนยันหมดอายุ กรุณาขอรหัสใหม่") from e
        await self._finish_login()
        return "connected"

    async def verify_password(self, password: str) -> None:
        pending = self._require_pending()
        try:
            await pending["client"].sign_in(password=password)
        except PasswordHashInvalidError as e:
            raise TelegramLoginError("รหัสผ่าน Telegram (2FA) ไม่ถูกต้อง") from e
        await self._finish_login()

    def _require_pending(self) -> dict:
        if not self._pending_login:
            raise TelegramLoginError("ไม่พบขั้นตอนการเข้าสู่ระบบ กรุณาขอรหัสใหม่")
        return self._pending_login

    async def _finish_login(self) -> None:
        client = self._pending_login["client"]
        self._pending_login = None
        if self.client:
            await self.client.disconnect()
        await self._activate(client)

    async def logout(self) -> None:
        if self.client:
            try:
                await self.client.log_out()
            except RPCError:
                await self.client.disconnect()
            self.client = None
        self._set_status("disconnected", session_enc="", me_id=None, me_name="")

    # ------------------------------------------------------------------ chats
    async def list_dialogs(self) -> list[dict]:
        if not self.connected:
            return []
        dialogs = []
        async for d in self.client.iter_dialogs(limit=300):
            kind = "group" if d.is_group else "channel" if d.is_channel else "user"
            dialogs.append({"id": d.id, "title": d.name or str(d.id), "kind": kind})
        return dialogs

    async def send_reply(self, chat_id: int, text: str, reply_to: int | None) -> None:
        if not self.connected:
            raise TelegramLoginError("ยังไม่ได้เชื่อมต่อ Telegram")
        try:
            entity = await self.client.get_input_entity(chat_id)
        except ValueError:
            await self.client.get_dialogs()
            entity = await self.client.get_input_entity(chat_id)
        await self.client.send_message(entity, text, reply_to=reply_to or None)

    # ------------------------------------------------------------------ events
    async def _handle_new_message(self, event: events.NewMessage.Event) -> None:
        if event.chat_id not in self._monitored:
            return
        msg = event.message
        try:
            sender = await event.get_sender()
        except RPCError:
            sender = None
        sender_name = ""
        if sender is not None:
            sender_name = (
                " ".join(filter(None, [getattr(sender, "first_name", None), getattr(sender, "last_name", None)]))
                or getattr(sender, "title", None) or getattr(sender, "username", None) or ""
            )
            if getattr(sender, "username", None):
                sender_name += f" (@{sender.username})"

        media_path = await self._download_image(event.chat_id, msg)
        with SessionLocal() as db:
            row = Message(
                chat_id=event.chat_id,
                tg_message_id=msg.id,
                sender_id=event.sender_id,
                sender_name=sender_name,
                is_outgoing=bool(msg.out),
                text=msg.message or "",
                media_path=media_path,
                date=msg.date.replace(tzinfo=None) if msg.date else utcnow(),
                analyzed=bool(msg.out),  # ข้อความของทีมงานเองไม่ต้องวิเคราะห์
            )
            db.add(row)
            db.commit()
        if not msg.out and self.on_message:
            self.on_message(event.chat_id)

    async def _download_image(self, chat_id: int, msg) -> str:
        ext = None
        if msg.photo:
            ext = ".jpg"
        elif msg.document and msg.file and msg.file.mime_type in IMAGE_MIME_EXT:
            ext = IMAGE_MIME_EXT[msg.file.mime_type]
        if not ext:
            return ""
        name = f"{abs(chat_id)}_{msg.id}{ext}"
        try:
            await msg.download_media(file=str(MEDIA_DIR / name))
        except (OSError, RPCError):
            log.exception("download media failed")
            return ""
        return name if (MEDIA_DIR / name).exists() else ""


telegram = TelegramService()

