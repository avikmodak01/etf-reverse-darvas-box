# bot_server.py
# Run with: uvicorn bot_server:app --host 0.0.0.0 --port $PORT
import os
import re
import json
from typing import Optional, Dict, Any, List

import requests
from fastapi import FastAPI, Request, HTTPException

app = FastAPI(title="Reverse Darvas Telegram Bot")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_ALLOWED_CHATS = os.getenv("TELEGRAM_ALLOWED_CHATS", "").strip()  # comma-separated chat IDs (recommended)
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "holdings")

def tg_api(method: str) -> str:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not set")
    return f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"

def is_allowed_chat(chat_id: int) -> bool:
    if not TELEGRAM_ALLOWED_CHATS:
        return True  # if not set, allow all (you can tighten this by setting it)
    allowed = {c.strip() for c in TELEGRAM_ALLOWED_CHATS.split(",") if c.strip()}
    return str(chat_id) in allowed

def send_message(chat_id: int, text: str) -> None:
    try:
        requests.post(tg_api("sendMessage"), json={
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True
        }, timeout=20)
    except Exception as e:
        print("[send_message] error:", e)

def upsert_holding(ticker: str, wac: float, qty: Optional[float]) -> tuple[bool, str]:
    if not (SUPABASE_URL and SUPABASE_KEY):
        return False, "Supabase env vars missing"
    try:
        payload = [{"Ticker": ticker, "WAC": float(wac), "Qty": float(qty) if qty is not None else 0}]
        r = requests.post(
            f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}",
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "resolution=merge-duplicates",
            },
            json=payload,
            timeout=30
        )
        if not r.ok:
            return False, f"HTTP {r.status_code}: {r.text}"
        return True, "Upserted"
    except Exception as e:
        return False, str(e)

SET_RE = re.compile(r"^/set\s+([A-Za-z0-9._-]+)\s+([0-9]*\.?[0-9]+)(?:\s+([0-9]*\.?[0-9]+))?\s*$", re.IGNORECASE)

HELP_TEXT = (
    "Use `/set <TICKER> <WAC> [QTY]`\n"
    "• Example: `/set ITBEES 348.25 100`\n"
    "This updates Supabase table: *holdings* (Ticker, WAC, Qty).\n"
)

@app.get("/")
def health():
    return {"ok": True, "service": "reverse-darvas-bot"}

@app.post("/telegram-webhook")
async def telegram_webhook(req: Request):
    if not TELEGRAM_BOT_TOKEN:
        raise HTTPException(status_code=500, detail="Bot token not configured")
    try:
        update = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # Support messages and channel posts
    msg = update.get("message") or update.get("channel_post")
    if not msg:
        return {"ok": True}

    chat = msg.get("chat") or {}
    chat_id = chat.get("id")
    text = msg.get("text", "") or ""

    if chat_id is None:
        return {"ok": True}

    # Access control
    if not is_allowed_chat(int(chat_id)):
        send_message(chat_id, "Unauthorized chat. Ask the admin to allow this chat ID.")
        return {"ok": True}

    # Parse command
    txt = text.strip()
    if txt.lower() in ("/start", "/help"):
        send_message(chat_id, HELP_TEXT)
        return {"ok": True}

    m = SET_RE.match(txt)
    if m:
        ticker = m.group(1).upper()
        try:
            wac = float(m.group(2))
        except ValueError:
            send_message(chat_id, "Invalid WAC. Usage: `/set TICKER WAC [QTY]`")
            return {"ok": True}
        qty = None
        if m.group(3) is not None:
            try:
                qty = float(m.group(3))
            except ValueError:
                send_message(chat_id, "Invalid QTY. Usage: `/set TICKER WAC [QTY]`")
                return {"ok": True}

        ok, msg = upsert_holding(ticker, wac, qty)
        if ok:
            qtxt = f"{qty:g}" if qty is not None else "0"
            send_message(chat_id, f"✅ Saved: *{ticker}* — WAC ₹{wac:.2f}, Qty {qtxt}")
        else:
            send_message(chat_id, f"❌ Failed: {msg}")
        return {"ok": True}

    # Fallback
    send_message(chat_id, "Unknown command.\n" + HELP_TEXT)
    return {"ok": True}
