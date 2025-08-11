# cron_job.py — Saturday Telegram from Supabase (uses etf_symbols whitelist)
import os
from datetime import date, timedelta
from typing import List, Dict, Tuple

import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=False)

BUY_OFFSET_ABS = float(os.getenv("BUY_OFFSET_ABS", "0.10"))
TP_PCT = float(os.getenv("TP_PCT", "6.28")) / 100.0

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()
SUPABASE_PRICES_TABLE = os.getenv("SUPABASE_PRICES_TABLE", "etf_prices")
SUPABASE_HOLDINGS_TABLE = os.getenv("SUPABASE_HOLDINGS_TABLE", "holdings")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"

HEADERS = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}

def previous_week_bounds(ref: date):
    back = (ref.weekday() - 4) % 7 or 7
    fri = ref - timedelta(days=back)
    mon = fri - timedelta(days=4)
    return mon, fri

def sb_whitelist_symbols() -> list[str]:
    r = requests.get(f"{SUPABASE_URL}/rest/v1/etf_symbols",
                     headers=HEADERS, params={"select":"symbol","order":"symbol.asc","limit":"10000"}, timeout=20)
    if not r.ok: return []
    return sorted({(row.get("symbol") or "").upper().strip() for row in r.json() if row.get("symbol")})

def sb_select_prices(symbols: List[str], mon: date, fri: date) -> pd.DataFrame:
    syms = ",".join(s.strip().upper() for s in symbols)
    params = [
        ("select", "symbol,low,trading_date"),
        ("symbol", f"in.({syms})"),
        ("trading_date", f"gte.{mon.isoformat()}"),
        ("trading_date", f"lte.{fri.isoformat()}"),
        ("limit", "100000"),
    ]
    url = f"{SUPABASE_URL}/rest/v1/{SUPABASE_PRICES_TABLE}"
    r = requests.get(url, headers=HEADERS, params=params, timeout=30)
    if not r.ok:
        raise SystemExit(f"Supabase prices fetch failed: {r.status_code} {r.text}")
    df = pd.DataFrame(r.json())
    return df

def sb_load_wac() -> Dict[str, float]:
    url = f"{SUPABASE_URL}/rest/v1/{SUPABASE_HOLDINGS_TABLE}"
    r = requests.get(url, headers=HEADERS, params={"select":"ticker,wac","limit":"10000"}, timeout=20)
    if not r.ok:
        return {}
    out = {}
    for row in r.json():
        t = str(row.get("ticker") or "").upper().strip()
        if t:
            out[t] = float(row.get("wac"))
    return out

def build_message() -> str:
    wl = sb_whitelist_symbols()
    if not wl:
        return "_No ETFs in etf_symbols (whitelist) to compute._"
    mon, fri = previous_week_bounds(date.today())
    df = sb_select_prices(wl, mon, fri)
    lines = []
    lines.append("*Reverse Darvas — Weekly Triggers*")
    lines.append(f"_Prev week: {mon.strftime('%d %b %Y')} → {fri.strftime('%d %b %Y')}_")
    lines.append("")
    if df.empty:
        lines.append("_No prices in Supabase for the selected week._")
        return "\n".join(lines)
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    lows = df.groupby("symbol", dropna=False)["low"].min().reset_index()
    wac_map = sb_load_wac()
    have = set()
    for _, r in lows.sort_values("symbol").iterrows():
        s = r["symbol"]; have.add(s)
        lw = float(r["low"])
        buy = max(lw - BUY_OFFSET_ABS, 0.0)
        if s in wac_map:
            tp = wac_map[s] * (1 + TP_PCT)
            lines.append(f"• `{s}` — *BUY next week:* ₹{buy:.2f} | *TP:* ₹{tp:.2f} (WAC ₹{wac_map[s]:.2f})")
        else:
            lines.append(f"• `{s}` — *BUY next week:* ₹{buy:.2f} | *TP:* N/A (no WAC)")
    missing = [s for s in wl if s not in have]
    if missing:
        lines.append("")
        lines.append("_No prices last week for:_ " + ", ".join(missing))
    return "\n".join(lines)

def send_telegram(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(text); return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    resp = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text,
                                    "parse_mode":"Markdown","disable_web_page_preview":True}, timeout=30)
    if not resp.ok:
        print(f"[telegram] fail {resp.status_code}: {resp.text}")

def main():
    msg = build_message()
    if DRY_RUN:
        print(msg)
    else:
        send_telegram(msg)

if __name__ == "__main__":
    main()
