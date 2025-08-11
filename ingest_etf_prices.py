# ingest_etf_prices.py  — obs-heavy version
import os, io, time, zipfile, re
from io import BytesIO
from datetime import date, datetime, timedelta
from typing import Optional, List, Set, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=False)

import NseUtility
nse = NseUtility.NseUtils()

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()
SUPABASE_PRICES_TABLE = os.getenv("SUPABASE_PRICES_TABLE", "etf_prices")
SUPABASE_SYMBOLS_TABLE = os.getenv("SUPABASE_SYMBOLS_TABLE", "etf_symbols")
ETF_LIST = os.getenv("ETF_LIST", "")

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "resolution=merge-duplicates"
}

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

ETF_SUFFIX_RE = re.compile(r"(BEES|IETF|ETF)$", re.IGNORECASE)
KNOWN_ETF_HINTS: Set[str] = {
    "ITBEES","JUNIORBEES","MIDCAPETF","SMALLCAP","INFRABEES","PSUBNKBEES","NIFTYBEES",
    "GOLDBEES","SILVERBEES","MON100","HNGSNGBEES","MODEFENCE","METALIETF","MID150CASE",
    "OILIETF","MOSMALL250","LOWVOLIETF","PHARMABEES","ALPL30IETF","PVTBANIETF","MOM30IETF",
    "MOCAPITAL","MIDSMALL"
}

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().upper().replace(" ", "") for c in df.columns]
    return df

def _prime(debug_net=False):
    try:
        if hasattr(nse, "headers"):
            nse.headers.update({"User-Agent": UA, "Accept":"*/*", "Referer":"https://www.nseindia.com/"})
        if getattr(nse, "session", None):
            nse.session.headers.update(getattr(nse, "headers", {}))
            r1 = nse.session.get("https://www.nseindia.com/", timeout=15)
            r2 = nse.session.get("https://www.nseindia.com/api/marketStatus", timeout=15)
            if debug_net:
                print(f"[prime] marketStatus -> {getattr(r2,'status_code',None)} {getattr(r2,'headers',{}).get('Content-Type','')}")
    except Exception as e:
        if debug_net:
            print("[prime] failed:", e)

def _archives_csv(d: date, debug_net=False) -> Tuple[Optional[pd.DataFrame], str]:
    url = f"https://archives.nseindia.com/products/content/sec_bhavdata_full_{d.strftime('%d%m%Y')}.csv"
    try:
        r = requests.get(url, headers={"User-Agent":UA,"Referer":"https://www.nseindia.com/"}, timeout=30)
        if debug_net:
            print(f"[archives] {d} -> {r.status_code} {r.headers.get('Content-Type','')} len={len(r.content)}")
            if r.content[:1] == b'<':
                print("  [archives] looks like HTML (challenge or 404 page) — showing first 120 bytes:")
                print("  ", r.text[:120].replace("\n"," "))
        if not r.ok or not r.text:
            return None, "http"
        try:
            df = pd.read_csv(io.StringIO(r.text))
            return _norm_cols(df), "ok"
        except Exception as e:
            if debug_net:
                print("[archives] parse error:", e)
            return None, "parse"
    except Exception as e:
        if debug_net:
            print("[archives] req error:", e)
        return None, "req"

def _cm_zip_bhav(d: date, debug_net=False) -> Tuple[Optional[pd.DataFrame], str]:
    mon = d.strftime('%b').upper()
    url = f"https://archives.nseindia.com/content/historical/EQUITIES/{d.strftime('%Y')}/{mon}/cm{d.strftime('%d')}{mon}{d.strftime('%Y')}bhav.csv.zip"
    try:
        r = requests.get(url, headers={"User-Agent":UA,"Referer":"https://www.nseindia.com/"}, timeout=30)
        if debug_net:
            print(f"[cmzip] {d} -> {r.status_code} {r.headers.get('Content-Type','')} len={len(r.content)}")
        if not r.ok:
            return None, "http"
        try:
            with zipfile.ZipFile(BytesIO(r.content)) as z:
                names = [n for n in z.namelist() if n.lower().endswith(".csv")]
                if not names: return None, "nocsv"
                df = pd.read_csv(z.open(names[0]))
                return _norm_cols(df), "ok"
        except Exception as e:
            if debug_net:
                print("[cmzip] parse error:", e)
            return None, "parse"
    except Exception as e:
        if debug_net:
            print("[cmzip] req error:", e)
        return None, "req"

def fetch_bhav(d: date, prefer: str = "auto", debug_net=False) -> Optional[pd.DataFrame]:
    _prime(debug_net)
    ds = d.strftime("%d-%m-%Y")
    df = None

    def try_nseutil() -> Tuple[Optional[pd.DataFrame], str]:
        try:
            if hasattr(nse, "equity_bhav_copy"):
                df0 = nse.equity_bhav_copy(ds)
                if df0 is not None and not getattr(df0, "empty", True):
                    return _norm_cols(df0), "ok"
        except Exception as e:
            if debug_net: print("[nseutil.equity_bhav_copy] error:", e)
        try:
            df1 = nse.bhav_copy_with_delivery(ds)
            if df1 is not None and not getattr(df1, "empty", True):
                return _norm_cols(df1), "ok"
        except Exception as e:
            if debug_net: print("[nseutil.bhav_copy_with_delivery] error:", e)
        return None, "miss"

    order = {
        "nseutil": ["nseutil"],
        "archives": ["archives"],
        "cmzip": ["cmzip"],
        "auto": ["nseutil", "archives", "cmzip"]
    }.get(prefer.lower(), ["nseutil", "archives", "cmzip"])

    for src in order:
        if src == "nseutil":
            df, tag = try_nseutil()
        elif src == "archives":
            df, tag = _archives_csv(d, debug_net=debug_net)
        else:
            df, tag = _cm_zip_bhav(d, debug_net=debug_net)
        if df is not None and not df.empty:
            if debug_net:
                print(f"[fetch] {d} <- {src} ✓  cols={list(df.columns)[:8]}")
            return df
        else:
            if debug_net:
                print(f"[fetch] {d} <- {src} ✗ ({tag})")
    return None

def upsert_symbols(symbols: List[str]):
    if not symbols: return
    url = f"{SUPABASE_URL}/rest/v1/{SUPABASE_SYMBOLS_TABLE}"
    r = requests.post(url, headers=HEADERS, json=[{"symbol": s} for s in symbols], timeout=60)
    if not r.ok:
        raise SystemExit(f"[Supabase symbols] {r.status_code}: {r.text}")

def upsert_prices(rows: List[dict], debug=False):
    if not rows: return
    url = f"{SUPABASE_URL}/rest/v1/{SUPABASE_PRICES_TABLE}"
    CHUNK = 500
    for i in range(0, len(rows), CHUNK):
        batch = rows[i:i+CHUNK]
        r = requests.post(url, headers=HEADERS, json=batch, timeout=60)
        if not r.ok:
            print(f"[ERROR] Upsert failed: {r.status_code} {r.text}")
            raise SystemExit("Supabase upsert failed; see error above.")
        if debug:
            print(f"  posted {len(batch)} rows")

def detect_etfs_in_bhav(df: pd.DataFrame) -> List[str]:
    if "SYMBOL" not in df.columns: return []
    symbols = df["SYMBOL"].astype(str).str.strip().str.upper()
    etfs = set([s for s in symbols if ETF_SUFFIX_RE.search(s)]) | (set(symbols) & KNOWN_ETF_HINTS)
    if "SERIES" in df.columns:
        series = df["SERIES"].astype(str).str.strip().str.upper()
        etfs = {s for s in etfs if series[symbols == s].iloc[0] == "EQ"}
    return sorted(etfs)

def main():
    import argparse
    p = argparse.ArgumentParser(description="Backfill or ingest daily ETF OHLC into Supabase.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--start", help="YYYY-MM-DD (use with --end)")
    g.add_argument("--yesterday", action="store_true")
    g.add_argument("--today", action="store_true")
    p.add_argument("--end", help="YYYY-MM-DD")
    p.add_argument("--etfs", help="Comma-separated ETFs (overrides ETF_LIST)")
    p.add_argument("--all-etfs", action="store_true", help="Detect ETFs in each bhav and ingest them all")
    p.add_argument("--sleep", type=float, default=0.2)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--debug-net", action="store_true", help="Verbose network + parsing logs")
    p.add_argument("--force", choices=["nseutil","archives","cmzip","auto"], default="auto",
                   help="Choose only one data source or auto (default).")
    args = p.parse_args()

    if not (SUPABASE_URL and SUPABASE_KEY):
        raise SystemExit("Set SUPABASE_URL and SUPABASE_KEY")

    explicit_etfs = [s.strip().upper() for s in (args.etfs or ETF_LIST).replace("\n", ",").split(",") if s.strip()]

    if args.yesterday or args.today:
        start = end = (date.today() - timedelta(days=1)) if args.yesterday else date.today()
    else:
        try:
            start = datetime.strptime(args.start, "%Y-%m-%d").date()
            end = datetime.strptime(args.end, "%Y-%m-%d").date()
        except Exception:
            raise SystemExit("Use --start YYYY-MM-DD --end YYYY-MM-DD, or --yesterday/--today")

    day = start
    total_rows, days_ok = 0, 0
    while day <= end:
        if day.weekday() < 5:
            df = fetch_bhav(day, prefer=args.force, debug_net=args.debug_net)
            if df is None or df.empty:
                print(f"[skip] {day} — no bhav (all sources failed)")
            elif "SYMBOL" not in df.columns:
                print(f"[skip] {day} — parsed but missing SYMBOL col; columns={list(df.columns)[:12]}")
            else:
                syms_today = []
                if args.all_etfs:
                    syms_today = detect_etfs_in_bhav(df)
                    if args.debug:
                        print(f"[{day}] detected ETFs: {', '.join(syms_today[:25])}{' ...' if len(syms_today)>25 else ''}")
                else:
                    syms_today = explicit_etfs
                syms_today = sorted(set(syms_today))
                if not syms_today:
                    print(f"[warn] {day} — no ETFs to ingest (empty list)")
                else:
                    upsert_symbols(syms_today)
                    pool = df["SYMBOL"].astype(str).str.strip().str.upper()
                    # Map price columns
                    opcol = "OPEN" if "OPEN" in df.columns else ("OPEN_PRICE" if "OPEN_PRICE" in df.columns else None)
                    hicol = "HIGH" if "HIGH" in df.columns else ("HIGH_PRICE" if "HIGH_PRICE" in df.columns else None)
                    locol = "LOW"  if "LOW"  in df.columns else ("LOW_PRICE"  if "LOW_PRICE"  in df.columns else None)
                    clcol = "CLOSE" if "CLOSE" in df.columns else ("CLOSE_PRICE" if "CLOSE_PRICE" in df.columns else None)
                    if not all([opcol, hicol, locol, clcol]):
                        print(f"[skip] {day} — missing price columns in bhav; cols={list(df.columns)[:12]}")
                    else:
                        rows = []
                        matched = []
                        for s in syms_today:
                            sub = df[pool == s]
                            if not sub.empty:
                                r = sub.iloc[0]
                                matched.append(s)
                                rows.append({
                                    "symbol": s,
                                    "trading_date": day.isoformat(),
                                    "open": float(r[opcol]),
                                    "high": float(r[hicol]),
                                    "low":  float(r[locol]),
                                    "close":float(r[clcol]),
                                    "source":"nse"
                                })
                        if rows:
                            if args.debug: print(f"[{day}] matched {len(matched)} / {len(syms_today)}")
                            upsert_prices(rows, debug=args.debug)
                            total_rows += len(rows); days_ok += 1
                            print(f"[ok] {day} → {len(rows)} rows")
                        else:
                            print(f"[warn] {day} → 0 matches (etf list may not exist that day)")
            time.sleep(args.sleep)
        day += timedelta(days=1)
    print(f"\nDone. Days inserted: {days_ok}, rows upserted: {total_rows}")

if __name__ == "__main__":
    main()
