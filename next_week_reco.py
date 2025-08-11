# next_week_reco.py — Next-week recos from Supabase (whitelist from etf_symbols)
# Usage:
#   python next_week_reco.py --ref 2025-08-09
#   python next_week_reco.py --ref 2025-08-09 --etfs ITBEES,NIFTYBEES --debug

import os
from datetime import date, datetime, timedelta
from typing import Optional, List, Dict, Tuple

import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=False)

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()
SUPABASE_PRICES_TABLE = os.getenv("SUPABASE_PRICES_TABLE", "etf_prices")
SUPABASE_HOLDINGS_TABLE = os.getenv("SUPABASE_HOLDINGS_TABLE", "holdings")
BUY_OFFSET_ABS = float(os.getenv("BUY_OFFSET_ABS", "0.10"))
TP_PCT = float(os.getenv("TP_PCT", "6.28"))

HEADERS = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}

def previous_completed_week(ref_day: date) -> List[date]:
    wd = ref_day.weekday()
    back = (wd - 4) % 7 or 7
    last_fri = ref_day - timedelta(days=back)
    mon = last_fri - timedelta(days=4)
    return [mon + timedelta(days=i) for i in range(5)]

def sb_whitelist_symbols() -> list[str]:
    r = requests.get(f"{SUPABASE_URL}/rest/v1/etf_symbols",
                     headers=HEADERS, params={"select":"symbol","order":"symbol.asc","limit":"10000"}, timeout=20)
    if not r.ok: return []
    return sorted({(row.get("symbol") or "").upper().strip() for row in r.json() if row.get("symbol")})

def sb_select_prices(symbols: List[str], mon: date, fri: date, debug=False) -> pd.DataFrame:
    wl = set(sb_whitelist_symbols())
    symbols = [s.upper().strip() for s in symbols if s.upper().strip() in wl]
    if not symbols: return pd.DataFrame(columns=["symbol","low","trading_date"])
    syms = ",".join(symbols)
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
    if debug:
        print(f"[debug] pulled {len(df)} rows from Supabase for {mon}..{fri}")
    return df

def sb_load_wac() -> Dict[str, Tuple[float, float]]:
    url = f"{SUPABASE_URL}/rest/v1/{SUPABASE_HOLDINGS_TABLE}"
    r = requests.get(url, headers=HEADERS, params={"select":"ticker,wac,qty","limit":"10000"}, timeout=20)
    if not r.ok: return {}
    out = {}
    for row in r.json():
        t = str(row.get("ticker") or "").upper().strip()
        if t:
            out[t] = (float(row.get("wac")), float(row.get("qty") or 0))
    return out

def compute_from_supabase(etfs: List[str], ref: Optional[date], offset: float, tp_pct: float, debug=False):
    ref_day = ref or date.today()
    days = previous_completed_week(ref_day)
    mon, fri = days[0], days[-1]
    df = sb_select_prices(etfs, mon, fri, debug=debug)
    if df.empty:
        return pd.DataFrame(), [], days
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    lowest = (df.groupby("symbol", dropna=False)["low"].min()
                .reset_index().rename(columns={"low":"PrevWeekLow"}))
    lowest["NextWeekBUY"] = (lowest["PrevWeekLow"] - offset).clip(lower=0).round(2)
    wac_map = sb_load_wac()
    rows = []
    for _, r in lowest.iterrows():
        s = r["symbol"]
        wac, qty, tp = (np.nan, np.nan, np.nan)
        if s in wac_map:
            wac, qty = wac_map[s]
            tp = round(wac * (1 + tp_pct/100), 2)
        rows.append({
            "Ticker": s,
            "WeekFrom": mon.isoformat(),
            "WeekTo": fri.isoformat(),
            "PrevWeekLow": round(float(r["PrevWeekLow"]), 2),
            "NextWeekBUY": round(float(r["NextWeekBUY"]), 2),
            "WAC": (None if np.isnan(wac) else round(float(wac), 4)),
            "Qty": (None if np.isnan(qty) else int(qty)),
            "TP_from_WAC": (None if np.isnan(tp) else tp)
        })
    have = {x["Ticker"] for x in rows}
    whitelist = set(sb_whitelist_symbols())
    missing = [s for s in (t.upper().strip() for t in etfs) if s not in have and s in whitelist]
    out = pd.DataFrame(rows).sort_values("Ticker").reset_index(drop=True)
    return out, missing, days

def main():
    import argparse
    p = argparse.ArgumentParser(description="Next-week Reverse Darvas recommendations (from Supabase, whitelist).")
    p.add_argument("--ref", help="Reference date YYYY-MM-DD or DD-MM-YYYY (defaults: today).")
    p.add_argument("--etfs", help="Comma-separated ETFs; default: symbols from etf_symbols table", default=None)
    p.add_argument("--offset", type=float, default=BUY_OFFSET_ABS)
    p.add_argument("--tp", type=float, default=TP_PCT)
    p.add_argument("--csv", help="Output CSV path (default: weekly_triggers_<WeekTo>.csv)")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    if not (SUPABASE_URL and SUPABASE_KEY):
        raise SystemExit("Set SUPABASE_URL and SUPABASE_KEY")

    if args.etfs:
        etfs = [s.strip().upper() for s in args.etfs.replace("\n", ",").split(",") if s.strip()]
    else:
        etfs = sb_whitelist_symbols()
    # enforce whitelist regardless
    wl = set(sb_whitelist_symbols())
    etfs = [s for s in etfs if s in wl]
    if not etfs:
        raise SystemExit("No ETFs left after enforcing whitelist (etf_symbols).")

    ref_date = None
    if args.ref:
        try:
            if len(args.ref.split("-")[0]) == 4:
                ref_date = datetime.strptime(args.ref, "%Y-%m-%d").date()
            else:
                ref_date = datetime.strptime(args.ref, "%d-%m-%Y").date()
        except ValueError:
            raise SystemExit("Invalid --ref; use YYYY-MM-DD or DD-MM-YYYY.")

    df, missing, week_days = compute_from_supabase(etfs, ref_date, args.offset, args.tp, debug=args.debug)
    wfrom, wto = week_days[0].strftime("%d %b %Y"), week_days[-1].strftime("%d %b %Y")

    print("\nReverse Darvas — Next Week Recommendations")
    print(f"Prev week window: {wfrom} → {wto}")
    print(f"Offset: ₹{args.offset:.2f} | TP: {args.tp:.2f}% (TP shown only if WAC present)\n")

    if not df.empty:
        print(df.to_string(index=False, justify="left",
                           formatters={
                               "PrevWeekLow":"{:,.2f}".format,
                               "NextWeekBUY":"{:,.2f}".format,
                               "WAC":        (lambda x: "" if pd.isna(x) else f"{x:,.4f}"),
                               "Qty":        (lambda x: "" if pd.isna(x) else f"{int(x)}"),
                               "TP_from_WAC":"{:,.2f}".format,
                           }))
        out_csv = args.csv or f"weekly_triggers_{week_days[-1].strftime('%Y%m%d')}.csv"
        df.to_csv(out_csv, index=False)
        print(f"\nSaved: {out_csv}\n")
    else:
        print("No symbols had data in the previous week window.")

    if missing:
        print("\nNo prices in Supabase last week for (whitelisted):", ", ".join(missing))

if __name__ == "__main__":
    main()
