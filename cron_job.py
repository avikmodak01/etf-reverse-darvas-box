# cron_job.py
# Run weekly on Render Cron to post BUY/SELL trigger prices to Telegram.

import os
import sys
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional

import requests
import pandas as pd
import numpy as np

# Local utility from your repo
import NseUtility


# ----------------------------- Config via env vars -----------------------------
ETF_LIST = os.getenv("ETF_LIST", "").strip()  # e.g. "ITBEES,PHARMABEES,NIFTYBEES"
BUY_OFFSET_ABS = float(os.getenv("BUY_OFFSET_ABS", "0.10"))           # rupees
TP_PCT = float(os.getenv("TP_PCT", "6.28")) / 100.0                   # 6.28% -> 0.0628

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# Optional Supabase (for WAC/Qty)
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "holdings")

DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"


# ----------------------------- NSE helpers -----------------------------
nse = NseUtility.NseUtils()

def fetch_bhav(date_obj: date) -> Optional[pd.DataFrame]:
    """Fetch bhav (equity preferred; delivery fallback). Returns uppercase, no spaces in col names."""
    ds = date_obj.strftime("%d-%m-%Y")
    df = None
    # Try equity bhav if present in your utility
    try:
        if hasattr(nse, "equity_bhav_copy"):
            df = nse.equity_bhav_copy(ds)
    except Exception:
        df = None
    # Fallback to with delivery
    if df is None or (hasattr(df, "empty") and df.empty):
        try:
            df = nse.bhav_copy_with_delivery(ds)
        except Exception:
            df = None

    if df is None or (hasattr(df, "empty") and df.empty):
        return None
    try:
        df.columns = [str(c).strip().upper().replace(" ", "") for c in df.columns]
    except Exception:
        pass
    return df


def bhav_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.upper(): c for c in df.columns}
    for cand in candidates:
        u = cand.upper()
        if u in cols:
            return cols[u]
    return None


def previous_week_range(ref_day: date) -> List[date]:
    """
    For a Saturday run, previous trading week is Mon..Fri ending on Friday (yesterday).
    If you run on another day, it still returns the most recently completed Mon..Fri window.
    """
    # find the most recent Friday before today
    weekday = ref_day.weekday()  # Mon=0..Sun=6
    # shift back to last Friday (4)
    days_back_to_fri = (weekday - 4) % 7 or 7  # ensure at least 1 week back if run Sat (weekday=5)
    last_fri = ref_day - timedelta(days=days_back_to_fri)
    mon = last_fri - timedelta(days=4)
    return [mon + timedelta(days=i) for i in range(5)]  # Mon..Fri


# ----------------------------- Supabase (optional) -----------------------------
def load_wac_map() -> Dict[str, float]:
    """
    Return {TICKER: WAC} using Supabase REST, if configured. Otherwise {}.
    Expects table columns: Ticker (text), WAC (numeric), Qty (numeric).
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        return {}
    try:
        endpoint = f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}"
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
        }
        params = {"select": "Ticker,WAC,Qty"}
        r = requests.get(endpoint, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        out = {}
        for row in data:
            t = str(row.get("Ticker", "")).upper().strip()
            wac = row.get("WAC", None)
            if t and wac is not None:
                out[t] = float(wac)
        return out
    except Exception as e:
        print(f"[WARN] Supabase load failed: {e}")
        return {}


# ----------------------------- Telegram -----------------------------
def send_telegram_message(text: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[WARN] Telegram env vars missing; printing message:\n")
        print(text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }
    resp = requests.post(url, json=payload, timeout=30)
    if not resp.ok:
        print(f"[ERROR] Telegram send failed: {resp.status_code} {resp.text}")


# ----------------------------- Core weekly computation -----------------------------
def compute_weekly_triggers(etfs: List[str]) -> str:
    """
    Returns a Markdown string with BUY (next-week) and SELL TP (if WAC available) for each ETF.
    BUY level = previous week's min(Low) - BUY_OFFSET_ABS.
    TP level  = WAC * (1 + TP_PCT) if WAC available; else 'N/A'.
    """
    days = previous_week_range(date.today())
    wac_map = load_wac_map()

    # Pre-fetch bhavs per day to reduce repeated HTTP calls
    day_frames: Dict[date, Optional[pd.DataFrame]] = {}
    for d in days:
        if d.weekday() < 5:
            day_frames[d] = fetch_bhav(d)

    lines = []
    hdr_from = days[0].strftime("%d %b %Y")
    hdr_to = days[-1].strftime("%d %b %Y")
    lines.append(f"*Reverse Darvas — Weekly Triggers*")
    lines.append(f"_Prev week: {hdr_from} → {hdr_to}_")
    lines.append("")

    for sym in etfs:
        s = sym.upper().strip()
        lows: List[float] = []
        for d in days:
            df = day_frames.get(d)
            if df is None or df.empty:
                continue
            sycol = "SYMBOL" if "SYMBOL" in df.columns else None
            opcol = bhav_col(df, ["OPEN", "OPEN_PRICE", "OPENPRICE"])
            hicol = bhav_col(df, ["HIGH", "HIGH_PRICE", "HIGHPRICE"])
            locol = bhav_col(df, ["LOW", "LOW_PRICE", "LOWPRICE"])
            clcol = bhav_col(df, ["CLOSE", "CLOSE_PRICE", "CLOSEPRICE", "LAST", "LAST_PRICE", "LASTPRICE"])
            if not all([sycol, opcol, hicol, locol, clcol]):
                continue
            sub = df[df[sycol].astype(str).str.upper() == s]
            if not sub.empty:
                try:
                    lows.append(float(sub.iloc[0][locol]))
                except Exception:
                    pass

        if not lows:
            lines.append(f"• `{s}` — No data last week")
            continue

        prev_week_low = float(np.min(lows))
        buy_level = max(prev_week_low - BUY_OFFSET_ABS, 0.0)
        # TP if we have WAC
        if s in wac_map:
            tp = float(wac_map[s] * (1.0 + TP_PCT))
            lines.append(f"• `{s}` — *BUY next week:* ₹{buy_level:.2f} | *TP:* ₹{tp:.2f} (WAC ₹{wac_map[s]:.2f})")
        else:
            lines.append(f"• `{s}` — *BUY next week:* ₹{buy_level:.2f} | *TP:* N/A (no WAC)")

    return "\n".join(lines)


# ----------------------------- Entrypoint -----------------------------
def main():
    if not ETF_LIST:
        print("Set ETF_LIST env var, e.g. ITBEES,PHARMABEES,NIFTYBEES")
        sys.exit(1)
    etfs = [x.strip() for x in ETF_LIST.replace("\n", ",").split(",") if x.strip()]

    text = compute_weekly_triggers(etfs)

    if DRY_RUN:
        print(text)
    else:
        send_telegram_message(text)

if __name__ == "__main__":
    main()
