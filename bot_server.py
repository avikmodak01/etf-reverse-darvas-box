# bot_server.py — Telegram bot with /nextweek, /set, /buy, /sell (whitelist for /nextweek)
import os, re
from datetime import datetime, date, timezone, timedelta
from typing import Optional, List, Dict, Tuple

import requests
from fastapi import FastAPI, Request, HTTPException
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=False)
app = FastAPI(title="Reverse Darvas Telegram Bot")

# ---- Env
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_ALLOWED_CHATS = os.getenv("TELEGRAM_ALLOWED_CHATS", "").strip()

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()
PRICES_TABLE = os.getenv("SUPABASE_PRICES_TABLE", "etf_prices")
HOLDINGS_TABLE = os.getenv("SUPABASE_HOLDINGS_TABLE", "holdings")
TRADES_TABLE = os.getenv("SUPABASE_TRADES_TABLE", "trades")

BUY_OFFSET_ABS = float(os.getenv("BUY_OFFSET_ABS", "0.10"))
TP_PCT = float(os.getenv("TP_PCT", "6.28")) / 100.0

HEADERS = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}

# ---- Telegram helpers
def tg_api(method: str) -> str:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not set")
    return f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"

def is_allowed_chat(chat_id: int) -> bool:
    if not TELEGRAM_ALLOWED_CHATS:
        return True
    allowed = {c.strip() for c in TELEGRAM_ALLOWED_CHATS.split(",") if c.strip()}
    return str(chat_id) in allowed

def send_message(chat_id: int, text: str):
    try:
        requests.post(tg_api("sendMessage"), json={
            "chat_id": chat_id, "text": text,
            "parse_mode": "Markdown", "disable_web_page_preview": True
        }, timeout=20)
    except Exception as e:
        print("[send_message] error:", e)

# ---- Date helpers (IST)
IST = timezone(timedelta(hours=5, minutes=30))
def parse_date(s: Optional[str]) -> datetime:
    if not s:
        return datetime.now(IST)
    for fmt in ("%Y-%m-%d", "%d-%m-%Y"):
        try:
            d = datetime.strptime(s, fmt)
            return datetime(d.year, d.month, d.day, tzinfo=IST)
        except ValueError:
            continue
    raise ValueError("Use YYYY-MM-DD or DD-MM-YYYY")

def previous_week_bounds(ref: date) -> Tuple[date, date]:
    back = (ref.weekday() - 4) % 7 or 7
    fri = ref - timedelta(days=back)
    mon = fri - timedelta(days=4)
    return mon, fri

# ---- Supabase helpers
def sb_whitelist_symbols() -> list[str]:
    r = requests.get(f"{SUPABASE_URL}/rest/v1/etf_symbols",
                     headers=HEADERS, params={"select":"symbol","order":"symbol.asc","limit":"10000"}, timeout=20)
    if not r.ok: return []
    return sorted({(row.get("symbol") or "").upper().strip() for row in r.json() if row.get("symbol")})

def sb_get_holdings_map() -> Dict[str, Tuple[float, float]]:
    r = requests.get(f"{SUPABASE_URL}/rest/v1/{HOLDINGS_TABLE}",
                     headers=HEADERS, params={"select":"ticker,wac,qty","limit":"10000"}, timeout=20)
    if not r.ok: return {}
    out = {}
    for row in r.json():
        t = str(row.get("ticker") or "").upper().strip()
        if t:
            out[t] = (float(row.get("wac")), float(row.get("qty") or 0))
    return out

def sb_upsert_holdings(ticker: str, wac: float, qty: float) -> Tuple[bool, str]:
    payload = [{"ticker": ticker, "wac": float(wac), "qty": float(qty)}]
    r = requests.post(f"{SUPABASE_URL}/rest/v1/{HOLDINGS_TABLE}",
                      headers={**HEADERS, "Content-Type":"application/json","Prefer":"resolution=merge-duplicates"},
                      json=payload, timeout=20)
    return (r.ok, r.text if not r.ok else "ok")

def sb_insert_trade(ticker: str, side: str, price: float, qty: float, traded_at: datetime, note: str = "") -> Tuple[bool, str]:
    payload = [{"ticker": ticker, "side": side, "price": float(price), "qty": float(qty),
                "traded_at": traded_at.isoformat(), "note": note}]
    r = requests.post(f"{SUPABASE_URL}/rest/v1/{TRADES_TABLE}",
                      headers={**HEADERS, "Content-Type":"application/json"}, json=payload, timeout=20)
    return (r.ok, r.text if not r.ok else "ok")

def sb_weekly_lows(symbols: List[str], mon: date, fri: date) -> Dict[str, float]:
    syms = ",".join(s.strip().upper() for s in symbols)
    params = [
        ("select","symbol,low"),
        ("symbol", f"in.({syms})"),
        ("trading_date", f"gte.{mon.isoformat()}"),
        ("trading_date", f"lte.{fri.isoformat()}"),
        ("limit","100000")
    ]
    r = requests.get(f"{SUPABASE_URL}/rest/v1/{PRICES_TABLE}", headers=HEADERS, params=params, timeout=30)
    r.raise_for_status()
    import pandas as pd
    df = pd.DataFrame(r.json())
    if df.empty: return {}
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    lows = df.groupby("symbol")["low"].min()
    return {k: float(v) for k, v in lows.to_dict().items()}

# ---- Command patterns
SET_RE   = re.compile(r"^/set\s+([A-Za-z0-9._-]+)\s+([0-9]*\.?[0-9]+)(?:\s+([0-9]*\.?[0-9]+))?\s*$", re.I)
BUY_RE   = re.compile(r"^/buy\s+([A-Za-z0-9._-]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)(?:\s+(\d{4}-\d{2}-\d{2}|\d{2}-\d{2}-\d{4}))?\s*$", re.I)
SELL_RE  = re.compile(r"^/sell\s+([A-Za-z0-9._-]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)(?:\s+(\d{4}-\d{2}-\d{4}|\d{2}-\d{2}-\d{4}))?\s*$", re.I)
NEXTWK_RE= re.compile(r"^/(nextweek|buylist)(?:\s+(\d{4}-\d{2}-\d{2}|\d{2}-\d{2}-\d{4}))?(?:\s+(.+))?$", re.I)

HELP = (
"*Commands*\n"
"• `/nextweek` — show next-week BUY levels from Supabase (uses `etf_symbols` whitelist)\n"
"   `/nextweek 2025-08-09 ITBEES,NIFTYBEES` to override date/symbols (still whitelisted)\n"
"• `/set TICKER WAC [QTY]` — set/update holdings WAC/Qty\n"
"• `/buy TICKER PRICE QTY [DATE]` — record buy; updates holdings WAC & Qty\n"
"• `/sell TICKER PRICE QTY [DATE]` — record sell; reduces Qty\n"
"_Dates: YYYY-MM-DD or DD-MM-YYYY. Amounts in ₹._"
)

# ---- Business logic
def handle_nextweek(args_date: Optional[str], args_syms: Optional[str]) -> str:
    ref = parse_date(args_date).date() if args_date else date.today()
    mon, fri = previous_week_bounds(ref)
    wl = set(sb_whitelist_symbols())
    if args_syms:
        symbols = [s.strip().upper() for s in args_syms.replace("\n",",").split(",") if s.strip()]
        symbols = [s for s in symbols if s in wl]
    else:
        symbols = sorted(wl)
    if not symbols:
        return "_No valid ETFs in `etf_symbols` to compute._"
    try:
        weekly_lows = sb_weekly_lows(symbols, mon, fri)
    except Exception as e:
        return f"_Failed to pull prices: {e}_"
    wac_map = sb_get_holdings_map()
    lines = [f"*Reverse Darvas — Weekly Triggers*",
             f"_Prev week: {mon.strftime('%d %b %Y')} → {fri.strftime('%d %b %Y')}_",
             ""]
    have = set()
    for s in sorted(weekly_lows.keys()):
        have.add(s)
        low = weekly_lows[s]
        buy = max(low - BUY_OFFSET_ABS, 0.0)
        if s in wac_map:
            wac = wac_map[s][0]
            tp = wac * (1 + TP_PCT)
            lines.append(f"• `{s}` — *BUY:* ₹{buy:.2f} | *TP:* ₹{tp:.2f} (WAC ₹{wac:.2f})")
        else:
            lines.append(f"• `{s}` — *BUY:* ₹{buy:.2f} | *TP:* N/A (no WAC)")
    missing = [s for s in symbols if s not in have]
    if missing:
        lines += ["", "_No prices last week for:_ " + ", ".join(missing)]
    return "\n".join(lines)

def handle_buy(ticker: str, price: float, qty: float, traded_at: datetime) -> str:
    t = ticker.upper()
    ok, msg = sb_insert_trade(t, "BUY", price, qty, traded_at)
    if not ok:
        return f"❌ trade insert failed: {msg}"
    holdings = sb_get_holdings_map()
    prev_wac, prev_qty = holdings.get(t, (0.0, 0.0))
    new_qty = prev_qty + qty
    new_wac = (prev_wac * prev_qty + price * qty) / new_qty if new_qty > 0 else 0.0
    ok2, msg2 = sb_upsert_holdings(t, new_wac, new_qty)
    if not ok2:
        return f"⚠️ trade saved but holdings update failed: {msg2}"
    return f"✅ BUY saved: *{t}* — {qty:g} @ ₹{price:.2f} on {traded_at.date().isoformat()}\nWAC ₹{new_wac:.4f}, Qty {new_qty:g}"

def handle_sell(ticker: str, price: float, qty: float, traded_at: datetime) -> str:
    t = ticker.upper()
    ok, msg = sb_insert_trade(t, "SELL", price, qty, traded_at)
    if not ok:
        return f"❌ trade insert failed: {msg}"
    holdings = sb_get_holdings_map()
    prev_wac, prev_qty = holdings.get(t, (0.0, 0.0))
    if prev_qty <= 0:
        new_qty = 0.0
        ok2, msg2 = sb_upsert_holdings(t, prev_wac, new_qty)
        note = " (no prior qty)"
    else:
        sell_qty = min(qty, prev_qty)
        new_qty = prev_qty - sell_qty
        ok2, msg2 = sb_upsert_holdings(t, prev_wac if new_qty > 0 else 0.0, new_qty)
        note = "" if sell_qty == qty else f" (capped to {sell_qty:g} from {qty:g})"
    if not ok2:
        return f"⚠️ trade saved but holdings update failed: {msg2}"
    return f"✅ SELL saved: *{t}* — {qty:g} @ ₹{price:.2f} on {traded_at.date().isoformat()}{note}\nWAC ₹{prev_wac:.4f}, Qty {new_qty:g}"

# ---- HTTP routes
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

    msg = update.get("message") or update.get("channel_post")
    if not msg: return {"ok": True}
    chat = msg.get("chat") or {}
    chat_id = chat.get("id")
    text = (msg.get("text") or "").strip()
    if chat_id is None: return {"ok": True}
    if not is_allowed_chat(int(chat_id)):
        send_message(chat_id, "Unauthorized chat. Ask admin to allow this chat ID.")
        return {"ok": True}

    if text.lower() in ("/start", "/help"):
        send_message(chat_id, HELP)
        return {"ok": True}

    m = NEXTWK_RE.match(text)
    if m:
        d = m.group(2)  # optional date
        syms = m.group(3)  # optional list
        out = handle_nextweek(d, syms)
        send_message(chat_id, out)
        return {"ok": True}

    m = SET_RE.match(text)
    if m:
        t = m.group(1).upper()
        wac = float(m.group(2)); qty = float(m.group(3) or 0)
        ok, msg2 = sb_upsert_holdings(t, wac, qty)
        send_message(chat_id, f"✅ Saved: *{t}* — WAC ₹{wac:.2f}, Qty {qty:g}" if ok else f"❌ {msg2}")
        return {"ok": True}

    m = BUY_RE.match(text)
    if m:
        t = m.group(1); price = float(m.group(2)); qty = float(m.group(3)); d = m.group(4)
        try:
            when = parse_date(d)
        except ValueError:
            send_message(chat_id, "Invalid date. Use YYYY-MM-DD or DD-MM-YYYY."); return {"ok": True}
        out = handle_buy(t, price, qty, when)
        send_message(chat_id, out); return {"ok": True}

    m = SELL_RE.match(text)
    if m:
        t = m.group(1); price = float(m.group(2)); qty = float(m.group(3)); d = m.group(4)
        try:
            when = parse_date(d)
        except ValueError:
            send_message(chat_id, "Invalid date. Use YYYY-MM-DD or DD-MM-YYYY."); return {"ok": True}
        out = handle_sell(t, price, qty, when)
        send_message(chat_id, out); return {"ok": True}

    send_message(chat_id, "Unknown command.\n" + HELP)
    return {"ok": True}
