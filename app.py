# app.py — Supabase-first Reverse Darvas ETF app (whitelist from etf_symbols)
# Run: streamlit run app.py

import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import requests
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv, find_dotenv

# -------------------------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------------------------
load_dotenv(find_dotenv(), override=False)
st.set_page_config(page_title="Reverse Darvas — NSE ETFs (Supabase)", layout="wide")
st.title("Reverse Darvas — NSE ETFs (Supabase)")
st.caption("Weekly low (Mon–Fri) → next-week BUY at (prev-week low − offset). TP: CMP ≥ WAC × (1 + TP%).")

# -------------------------------------------------------------------------------------
# Env / Config
# -------------------------------------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()
PRICES_TABLE = os.getenv("SUPABASE_PRICES_TABLE", "etf_prices")
HOLDINGS_TABLE = os.getenv("SUPABASE_HOLDINGS_TABLE", "holdings")
TRADES_TABLE = os.getenv("SUPABASE_TRADES_TABLE", "trades")

HEADERS = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
IST = timezone(timedelta(hours=5, minutes=30))

@dataclass
class StrategyConfig:
    per_buy_budget: float = 2000.0
    buy_offset_abs: float = 0.10
    take_profit_pct: float = 0.0628
    one_buy_per_week: bool = True
    allow_same_day_tp_after_buy: bool = True

# -------------------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------------------
def previous_completed_week(ref: date) -> Tuple[date, date]:
    back = (ref.weekday() - 4) % 7 or 7   # Friday
    fri = ref - timedelta(days=back)
    mon = fri - timedelta(days=4)
    return mon, fri

def _ensure_env():
    if not (SUPABASE_URL and SUPABASE_KEY):
        st.error("Set SUPABASE_URL and SUPABASE_KEY in environment or .env.")
        st.stop()

# -------------------------------------------------------------------------------------
# Supabase API (whitelist only)
# -------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=600)
def sb_whitelist_symbols() -> List[str]:
    """Return ONLY symbols from public.etf_symbols (whitelist)."""
    _ensure_env()
    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/etf_symbols",
        headers=HEADERS,
        params={"select": "symbol", "order": "symbol.asc", "limit": "10000"},
        timeout=20,
    )
    if not r.ok:
        st.error(f"Failed to load etf_symbols: {r.status_code} {r.text}")
        return []
    syms = sorted({(row.get("symbol") or "").upper().strip() for row in r.json() if row.get("symbol")})
    if not syms:
        st.warning("Your `etf_symbols` table is empty. Add your tickers there.")
    return syms

@st.cache_data(show_spinner=False, ttl=600)
def sb_fetch_prices(symbols: List[str], start: date, end: date) -> Dict[str, pd.DataFrame]:
    """Return {SYM: DF(Date index w/ Open High Low Close)} from Supabase etf_prices (whitelist-enforced)."""
    _ensure_env()
    allowed = set(sb_whitelist_symbols())
    symbols = [s.upper().strip() for s in symbols if s.upper().strip() in allowed]
    if not symbols:
        return {}
    syms = ",".join(symbols)
    params = [
        ("select", "symbol,trading_date,open,high,low,close"),
        ("symbol", f"in.({syms})"),
        ("trading_date", f"gte.{start.isoformat()}"),
        ("trading_date", f"lte.{end.isoformat()}"),
        ("limit", "100000"),
        ("order", "trading_date.asc")
    ]
    url = f"{SUPABASE_URL}/rest/v1/{PRICES_TABLE}"
    r = requests.get(url, headers=HEADERS, params=params, timeout=30)
    if not r.ok:
        st.error(f"Supabase prices fetch failed: {r.status_code} {r.text}")
        return {}
    df = pd.DataFrame(r.json())
    out: Dict[str, pd.DataFrame] = {}
    if df.empty:
        for s in symbols:
            out[s] = pd.DataFrame()
        return out
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df["trading_date"] = pd.to_datetime(df["trading_date"])
    for s, g in df.groupby("symbol"):
        t = (g.set_index("trading_date")
               .sort_index()[["open", "high", "low", "close"]]
               .rename(columns=str.capitalize))
        out[s] = t
    for s in symbols:
        if s not in out:
            out[s] = pd.DataFrame()
    return out

@st.cache_data(show_spinner=False, ttl=600)
def sb_weekly_lows(symbols: List[str], mon: date, fri: date) -> pd.DataFrame:
    """Return DataFrame(symbol, PrevWeekLow) (whitelist-enforced)."""
    _ensure_env()
    allowed = set(sb_whitelist_symbols())
    symbols = [s.upper().strip() for s in symbols if s.upper().strip() in allowed]
    if not symbols:
        return pd.DataFrame(columns=["symbol", "PrevWeekLow"])
    syms = ",".join(symbols)
    params = [
        ("select", "symbol,low"),
        ("symbol", f"in.({syms})"),
        ("trading_date", f"gte.{mon.isoformat()}"),
        ("trading_date", f"lte.{fri.isoformat()}"),
        ("limit", "100000")
    ]
    url = f"{SUPABASE_URL}/rest/v1/{PRICES_TABLE}"
    r = requests.get(url, headers=HEADERS, params=params, timeout=30)
    if not r.ok:
        st.error(f"Supabase prices fetch failed: {r.status_code} {r.text}")
        return pd.DataFrame(columns=["symbol", "PrevWeekLow"])
    df = pd.DataFrame(r.json())
    if df.empty:
        return pd.DataFrame(columns=["symbol", "PrevWeekLow"])
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    low = df.groupby("symbol", dropna=False)["low"].min().reset_index().rename(columns={"low": "PrevWeekLow"})
    return low

@st.cache_data(show_spinner=False, ttl=300)
def sb_get_holdings() -> pd.DataFrame:
    _ensure_env()
    r = requests.get(f"{SUPABASE_URL}/rest/v1/{HOLDINGS_TABLE}",
                     headers=HEADERS, params={"select": "ticker,wac,qty", "limit": "10000"}, timeout=20)
    if not r.ok:
        st.warning(f"Holdings fetch failed: {r.status_code}")
        return pd.DataFrame(columns=["Ticker", "WAC", "Qty"])
    df = pd.DataFrame(r.json())
    if df.empty:
        return pd.DataFrame(columns=["Ticker", "WAC", "Qty"])
    df = df.rename(columns={"ticker": "Ticker", "wac": "WAC", "qty": "Qty"})
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    return df

def sb_upsert_holdings(rows: pd.DataFrame) -> Tuple[bool, str]:
    _ensure_env()
    if rows is None or rows.empty:
        return False, "No rows"
    payload = [{"ticker": str(r["Ticker"]).upper().strip(),
                "wac": float(r["WAC"]), "qty": float(r.get("Qty", 0) or 0)}
               for _, r in rows.iterrows()]
    r = requests.post(f"{SUPABASE_URL}/rest/v1/{HOLDINGS_TABLE}",
                      headers={**HEADERS, "Content-Type": "application/json",
                               "Prefer": "resolution=merge-duplicates"},
                      json=payload, timeout=30)
    if not r.ok:
        return False, f"{r.status_code}: {r.text}"
    sb_get_holdings.clear()
    return True, "Saved."

@st.cache_data(show_spinner=False, ttl=300)
def sb_get_trades(symbols: Optional[List[str]] = None, start: Optional[date] = None, end: Optional[date] = None, limit: int = 10000) -> pd.DataFrame:
    _ensure_env()
    params = [("select", "id,ticker,side,qty,price,traded_at,note"), ("limit", str(limit)), ("order", "traded_at.asc")]
    if symbols:
        wl = set(sb_whitelist_symbols())
        symbols = [s.upper().strip() for s in symbols if s.upper().strip() in wl]
        if symbols:
            syms = ",".join(symbols)
            params.append(("ticker", f"in.({syms})"))
    if start:
        params.append(("traded_at", f"gte.{start.isoformat()}"))
    if end:
        params.append(("traded_at", f"lte.{end.isoformat()}T23:59:59"))
    r = requests.get(f"{SUPABASE_URL}/rest/v1/{TRADES_TABLE}", headers=HEADERS, params=params, timeout=30)
    if not r.ok:
        st.warning(f"Trades fetch failed: {r.status_code}")
        return pd.DataFrame(columns=["id", "ticker", "side", "qty", "price", "traded_at", "note"])
    df = pd.DataFrame(r.json())
    if df.empty:
        return pd.DataFrame(columns=["id", "ticker", "side", "qty", "price", "traded_at", "note"])
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["traded_at"] = pd.to_datetime(df["traded_at"]).dt.tz_convert(IST).dt.tz_localize(None)
    return df

def sb_insert_trade(ticker: str, side: str, price: float, qty: float, traded_at: datetime, note: str = "") -> Tuple[bool, str]:
    _ensure_env()
    payload = [{"ticker": ticker.upper().strip(), "side": side, "price": float(price),
                "qty": float(qty), "traded_at": traded_at.astimezone(IST).isoformat(), "note": note}]
    r = requests.post(f"{SUPABASE_URL}/rest/v1/{TRADES_TABLE}",
                      headers={**HEADERS, "Content-Type": "application/json"},
                      json=payload, timeout=30)
    if not r.ok:
        return False, f"{r.status_code}: {r.text}"
    sb_get_trades.clear()
    return True, "Saved."

# -------------------------------------------------------------------------------------
# Strategy (simulate from OHLC)
# -------------------------------------------------------------------------------------
def _prep_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    need = {"Open", "High", "Low", "Close"}
    if not need.issubset(df.columns):
        return pd.DataFrame()
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    out = out.sort_index().dropna()
    return out

def _attach_week_levels(d: pd.DataFrame, buy_offset_abs: float) -> pd.DataFrame:
    d = d.copy()
    d["week_end"] = d.index.to_period("W-FRI").to_timestamp("W-FRI")
    prev_low = d.groupby("week_end")["Low"].min().shift(1)
    d["week_buy_level"] = (prev_low - buy_offset_abs)
    return d

def simulate_ticker(daily: pd.DataFrame, cfg: StrategyConfig, ticker: str) -> pd.DataFrame:
    d = _prep_daily(daily)
    if d.empty:
        return pd.DataFrame(columns=["Date", "Ticker", "Side", "Price", "Qty", "CashFlow", "HoldingsQty", "AvgCost", "Note"])
    d = _attach_week_levels(d, cfg.buy_offset_abs).dropna(subset=["week_buy_level"])
    holdings_qty = 0
    avg_cost = 0.0
    records = []
    last_week_end = None
    bought_this_week = False
    for dt, row in d.iterrows():
        week_end = row["week_end"]; l = float(row["Low"]); c = float(row["Close"]); buy = float(row["week_buy_level"])
        if last_week_end is None or week_end != last_week_end:
            bought_this_week = False
            last_week_end = week_end
        buy_filled = False
        if (not bought_this_week or not cfg.one_buy_per_week) and l <= buy:
            qty = int(cfg.per_buy_budget // buy)
            if qty > 0:
                cost = qty * buy
                avg_cost = (avg_cost * holdings_qty + cost) / (holdings_qty + qty) if holdings_qty + qty > 0 else 0.0
                holdings_qty += qty
                records.append({"Date": dt, "Ticker": ticker, "Side": "BUY", "Price": round(buy, 2),
                                "Qty": qty, "CashFlow": -round(cost, 2), "HoldingsQty": holdings_qty,
                                "AvgCost": round(avg_cost, 4), "Note": f"Buy trigger {buy:.2f}"})
                bought_this_week = True
                buy_filled = True
        if holdings_qty > 0:
            tp = avg_cost * (1 + cfg.take_profit_pct)
            if (not buy_filled or cfg.allow_same_day_tp_after_buy) and c >= tp:
                proceeds = holdings_qty * tp
                records.append({"Date": dt, "Ticker": ticker, "Side": "SELL", "Price": round(tp, 2),
                                "Qty": holdings_qty, "CashFlow": round(proceeds, 2), "HoldingsQty": 0,
                                "AvgCost": 0.0, "Note": f"TP hit {tp:.2f}"})
                holdings_qty = 0
                avg_cost = 0.0
    tlog = pd.DataFrame.from_records(records)
    if not tlog.empty:
        tlog.sort_values(["Date", "Side"], inplace=True)
        tlog.reset_index(drop=True, inplace=True)
    return tlog

def summarize_roundtrips(trades: pd.DataFrame) -> pd.DataFrame:
    cols = ["Ticker", "ExitDate", "RoundtripQty", "AvgEntry", "ExitPrice", "RealizedPnL", "ReturnPct_on_Cost", "HoldingDays"]
    if trades is None or trades.empty:
        return pd.DataFrame(columns=cols)

    out = []
    t = trades.copy()
    t["grp"] = (t["Side"] == "SELL").cumsum()

    for (tkr, gid), g in t.groupby(["Ticker", "grp"], sort=False):
        if (g["Side"] == "SELL").any():
            buys = g[g["Side"] == "BUY"]
            sell = g[g["Side"] == "SELL"].iloc[-1]
            cost_in = -buys["CashFlow"].sum()
            qty_in = buys["Qty"].sum()
            avg_entry = (cost_in / qty_in) if qty_in > 0 else np.nan
            realized = g["CashFlow"].sum()
            ret = 100.0 * realized / cost_in if cost_in > 0 else np.nan
            first = buys["Date"].min() if not buys.empty else sell["Date"]
            days = (pd.to_datetime(sell["Date"]) - pd.to_datetime(first)).days
            out.append({
                "Ticker": tkr,
                "ExitDate": sell["Date"],
                "RoundtripQty": int(qty_in),
                "AvgEntry": (round(avg_entry, 4) if pd.notna(avg_entry) else np.nan),
                "ExitPrice": sell["Price"],
                "RealizedPnL": round(realized, 2),
                "ReturnPct_on_Cost": (round(ret, 2) if pd.notna(ret) else np.nan),
                "HoldingDays": days
            })

    df = pd.DataFrame(out, columns=cols)
    if not df.empty:
        df = df.sort_values(["Ticker", "ExitDate"]).reset_index(drop=True)
    return df

# -------------------------------------------------------------------------------------
# P&L from trades table (WAC method)
# -------------------------------------------------------------------------------------
def compute_holdings_and_realized_from_trades(trades: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """Process trades chronologically to compute per-ticker WAC/Qty and realized P&L."""
    if trades is None or trades.empty:
        return pd.DataFrame(columns=["Ticker", "Qty", "WAC", "RealizedPnL"]), 0.0
    rows = []; realized_total = 0.0
    for tkr, g in trades.sort_values("traded_at").groupby("ticker"):
        qty = 0.0; wac = 0.0; realized = 0.0
        for _, r in g.iterrows():
            side = str(r["side"]).upper(); q = float(r["qty"]); p = float(r["price"])
            if side == "BUY":
                new_qty = qty + q
                wac = (wac * qty + p * q) / new_qty if new_qty > 0 else 0.0
                qty = new_qty
            elif side == "SELL":
                sell_qty = min(qty, q)
                realized += (p - wac) * sell_qty
                qty -= sell_qty
                if qty <= 0:
                    wac = 0.0
        rows.append({"Ticker": tkr, "Qty": qty, "WAC": wac, "RealizedPnL": realized})
        realized_total += realized
    return pd.DataFrame(rows).sort_values("Ticker").reset_index(drop=True), realized_total

@st.cache_data(show_spinner=False, ttl=300)
def sb_latest_closes(symbols: List[str]) -> Dict[str, float]:
    """Get latest available Close for each symbol from Supabase (whitelist-enforced)."""
    if not symbols:
        return {}
    allowed = set(sb_whitelist_symbols())
    symbols = [s.upper().strip() for s in symbols if s.upper().strip() in allowed]
    if not symbols:
        return {}
    syms = ",".join(symbols)
    params = [("select", "symbol,trading_date,close"),
              ("symbol", f"in.({syms})"),
              ("order", "trading_date.desc"),
              ("limit", "100000")]
    r = requests.get(f"{SUPABASE_URL}/rest/v1/{PRICES_TABLE}", headers=HEADERS, params=params, timeout=30)
    if not r.ok:
        return {}
    df = pd.DataFrame(r.json())
    if df.empty:
        return {}
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df["trading_date"] = pd.to_datetime(df["trading_date"])
    latest = df.sort_values(["symbol", "trading_date"]).groupby("symbol").tail(1)
    return {row["symbol"]: float(row["close"]) for _, row in latest.iterrows()}

# -------------------------------------------------------------------------------------
# Sidebar (whitelist + Select all / Clear)
# -------------------------------------------------------------------------------------
st.sidebar.header("Backtest Settings")
start_date = st.sidebar.date_input("Start date", date(2020, 1, 1), key="s_start")
end_date = st.sidebar.date_input("End date", date.today(), key="s_end")

ALL_ETFS = sb_whitelist_symbols()
c_sel1, c_sel2 = st.sidebar.columns(2)
if c_sel1.button("Select all", key="btn_sel_all"):
    st.session_state["s_etfs"] = ALL_ETFS
    st.rerun()
if c_sel2.button("Clear", key="btn_sel_none"):
    st.session_state["s_etfs"] = []
    st.rerun()

etfs = st.sidebar.multiselect("ETFs", options=ALL_ETFS, default=ALL_ETFS, key="s_etfs")

st.sidebar.subheader("Strategy Parameters")
per_buy_budget = st.sidebar.number_input("Per-buy budget (₹)", min_value=100.0, step=100.0, value=2000.0, key="s_budget")
buy_offset_abs = st.sidebar.number_input("Buy offset below week low (₹)", min_value=0.0, step=0.05, value=0.10, key="s_offset")
tp_pct = st.sidebar.number_input("Take profit (%) over WAC", min_value=0.1, step=0.01, value=6.28, key="s_tp")
one_buy_per_week = st.sidebar.checkbox("At most 1 buy per week", value=True, key="s_onebuy")
allow_same_day_tp = st.sidebar.checkbox("Allow same-day TP after buy", value=True, key="s_sameday")

cfg = StrategyConfig(
    per_buy_budget=per_buy_budget,
    buy_offset_abs=buy_offset_abs,
    take_profit_pct=tp_pct / 100.0,
    one_buy_per_week=one_buy_per_week,
    allow_same_day_tp_after_buy=allow_same_day_tp
)

# -------------------------------------------------------------------------------------
# Tabs
# -------------------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Weekly Triggers", "Backtest", "Holdings", "Trades"])

# ---- Weekly Triggers ---------------------------------------------------------------
with tab1:
    st.subheader("Next-Week BUY Levels")
    ref_for_week = st.date_input("Reference date (compute previous completed Mon–Fri)", value=date.today(), key="t1_ref")
    mon, fri = previous_completed_week(ref_for_week)
    st.caption(f"Prev week window: {mon.strftime('%d %b %Y')} → {fri.strftime('%d %b %Y')}")
    if st.button("Compute Weekly Triggers", type="primary", key="btn_weekly"):
        lows = sb_weekly_lows(etfs, mon, fri)
        if lows.empty:
            st.warning("No prices found in Supabase for the chosen week.")
        else:
            lows["NextWeekBUY"] = (lows["PrevWeekLow"] - buy_offset_abs).clip(lower=0).round(2)
            wac_df = sb_get_holdings()
            wac_map = {r["Ticker"]: (r["WAC"], r["Qty"]) for _, r in wac_df.iterrows()} if not wac_df.empty else {}
            rows = []
            for _, r in lows.sort_values("symbol").iterrows():
                s = r["symbol"]; wac, qty, tpv = (np.nan, np.nan, np.nan)
                if s in wac_map:
                    wac, qty = wac_map[s]
                    tpv = round(wac * (1 + tp_pct / 100.0), 2)
                rows.append({"Ticker": s, "WeekFrom": mon.isoformat(), "WeekTo": fri.isoformat(),
                             "PrevWeekLow": round(float(r["PrevWeekLow"]), 2),
                             "NextWeekBUY": round(float(r["NextWeekBUY"]), 2),
                             "WAC": ("" if pd.isna(wac) else round(float(wac), 4)),
                             "Qty": ("" if pd.isna(qty) else int(qty)),
                             "TP_from_WAC": ("" if pd.isna(tpv) else tpv)})
            out = pd.DataFrame(rows)
            st.dataframe(out, use_container_width=True)
            st.download_button("Download CSV", out.to_csv(index=False).encode("utf-8"),
                               file_name=f"weekly_triggers_{fri.strftime('%Y%m%d')}.csv",
                               mime="text/csv", key="btn_dl_weekly")

# ---- Backtest ----------------------------------------------------------------------
with tab2:
    st.subheader("Backtest")
    run_bt = st.button("Run Backtest", type="primary", key="btn_bt_run")
    if run_bt:
        with st.spinner("Fetching OHLC from Supabase…"):
            hist = sb_fetch_prices(etfs, start_date, end_date)
        trade_logs = []
        ok_syms = []
        for s in etfs:
            df = hist.get(s.upper(), pd.DataFrame())
            if df is None or df.empty:
                continue
            ok_syms.append(s)
            tlog = simulate_ticker(df, cfg, s.upper())
            if not tlog.empty:
                trade_logs.append(tlog)
        trades = (pd.concat(trade_logs, ignore_index=True)
                  if trade_logs else pd.DataFrame(columns=["Date", "Ticker", "Side", "Price", "Qty", "CashFlow", "HoldingsQty", "AvgCost", "Note"]))
        summ = summarize_roundtrips(trades)

        c1, c2, c3, c4, c5 = st.columns(5)
        total_buys = int(trades[trades["Side"] == "BUY"]["Qty"].sum()) if not trades.empty else 0
        exits = int((trades["Side"] == "SELL").sum()) if not trades.empty else 0
        realized_total = float(summ["RealizedPnL"].sum()) if not summ.empty else 0.0
        # Unrealized from last Close
        last_close_map = sb_latest_closes(ok_syms)
        pos_rows = []
        if not trades.empty:
            for tkr, g in trades.groupby("Ticker"):
                last = g.iloc[-1]
                qty = int(last.get("HoldingsQty", 0) or 0)
                avg = float(last.get("AvgCost", 0.0) or 0.0)
                lc = float(last_close_map.get(tkr, np.nan))
                inv = qty * avg
                mv = qty * lc if qty > 0 and not np.isnan(lc) else 0.0
                pos_rows.append({"Ticker": tkr, "Qty": qty, "AvgCost": round(avg, 4), "LastClose": (None if np.isnan(lc) else round(lc, 2)),
                                 "Invested": round(inv, 2), "MktValue": round(mv, 2), "UnrealizedPnL": round(mv - inv, 2)})
        pos = pd.DataFrame(pos_rows).sort_values("Ticker") if pos_rows else pd.DataFrame(columns=["Ticker", "Qty", "AvgCost", "LastClose", "Invested", "MktValue", "UnrealizedPnL"])
        unreal_total = float(pos["UnrealizedPnL"].sum()) if not pos.empty else 0.0
        win_rate = float((summ["RealizedPnL"] > 0).mean() * 100) if not summ.empty else 0.0

        c1.metric("Total Buy Qty", f"{total_buys:,}")
        c2.metric("Exit Trades", f"{exits:,}")
        c3.metric("Realized P&L (₹)", f"{realized_total:,.0f}")
        c4.metric("Unrealized P&L (₹)", f"{unreal_total:,.0f}")
        c5.metric("Win Rate", f"{win_rate:.1f}%")

        st.markdown("#### Open Positions (Backtest)")
        st.dataframe(pos, use_container_width=True)
        st.markdown("#### Trades")
        st.dataframe(trades.assign(Date=pd.to_datetime(trades["Date"]).dt.date) if not trades.empty else trades,
                     use_container_width=True)
        st.download_button("Download Trades CSV",
                           trades.to_csv(index=False).encode("utf-8"),
                           file_name="rd_trades.csv", mime="text/csv", key="btn_bt_trades_dl")

        st.markdown("#### Roundtrip Summary")
        st.dataframe(summ.assign(ExitDate=pd.to_datetime(summ["ExitDate"]).dt.date) if not summ.empty else summ,
                     use_container_width=True)
        st.download_button("Download Summary CSV",
                           summ.to_csv(index=False).encode("utf-8"),
                           file_name="rd_summary.csv", key="btn_bt_summ_dl")

# ---- Holdings ----------------------------------------------------------------------
with tab3:
    st.subheader("Holdings (Supabase)")
    existing = sb_get_holdings()
    st.dataframe(existing, use_container_width=True)
    edited = st.data_editor(existing if not existing.empty else pd.DataFrame(columns=["Ticker", "WAC", "Qty"]),
                            num_rows="dynamic", key="ed_holdings", use_container_width=True)
    if st.button("Save holdings", type="primary", key="btn_holdings_save"):
        cleaned = (edited[["Ticker", "WAC", "Qty"]]
                   .dropna(subset=["Ticker"])
                   .assign(Ticker=lambda d: d["Ticker"].astype(str).str.upper().str.strip()))
        ok, msg = sb_upsert_holdings(cleaned)
        st.success(msg) if ok else st.error(msg)

# ---- Trades ------------------------------------------------------------------------
with tab4:
    st.subheader("Trades (Supabase ledger)")
    trades = sb_get_trades(symbols=etfs, start=start_date, end=end_date, limit=10000)
    st.dataframe(trades, use_container_width=True, height=320)

    # metrics from ledger
    ledger_pos, ledger_realized = compute_holdings_and_realized_from_trades(trades)
    last_close_map = sb_latest_closes(ledger_pos["Ticker"].tolist() if not ledger_pos.empty else [])
    if not ledger_pos.empty:
        ledger_pos = ledger_pos.assign(
            LastClose=lambda d: d["Ticker"].map(last_close_map).astype(float),
            Invested=lambda d: d["Qty"] * d["WAC"],
            MktValue=lambda d: d["Qty"] * d["LastClose"],
            UnrealizedPnL=lambda d: d["MktValue"] - d["Invested"]
        )
    c1, c2, c3 = st.columns(3)
    c1.metric("Ledger Realized P&L (₹)", f"{ledger_realized:,.0f}")
    c2.metric("Ledger Unrealized P&L (₹)", f"{(0.0 if ledger_pos.empty else ledger_pos['UnrealizedPnL'].sum()):,.0f}")
    c3.metric("Open Positions", "0" if ledger_pos.empty else f"{int((ledger_pos['Qty'] > 0).sum())}")

    st.markdown("#### Open Positions (from trades)")
    st.dataframe(ledger_pos, use_container_width=True)

    st.markdown("---")
    st.markdown("### Add Trade")
    col1, col2, col3, col4, col5 = st.columns([1.2, 1, 1, 1, 2])
    t_symbol = col1.text_input("Ticker", value=(etfs[0] if etfs else ""), key="t_trade_sym").upper().strip()
    t_side = col2.selectbox("Side", options=["BUY", "SELL"], key="t_trade_side")
    t_price = col3.number_input("Price (₹)", min_value=0.0, step=0.05, value=0.0, key="t_trade_price")
    t_qty = col4.number_input("Qty", min_value=0.0, step=1.0, value=0.0, key="t_trade_qty")
    t_date = col5.date_input("Date (IST)", value=date.today(), key="t_trade_date")
    t_note = st.text_input("Note (optional)", value="", key="t_trade_note")
    if st.button("Save Trade", type="primary", key="btn_trade_save"):
        if not t_symbol or t_price <= 0 or t_qty <= 0:
            st.error("Enter valid Ticker, Price and Qty.")
        else:
            when = datetime(t_date.year, t_date.month, t_date.day, 9, 15, tzinfo=IST)  # market time
            ok, msg = sb_insert_trade(t_symbol, t_side, t_price, t_qty, when, t_note)
            if not ok:
                st.error(msg)
            else:
                # Update holdings WAC/Qty like the bot
                h = sb_get_holdings()
                prev = h[h["Ticker"] == t_symbol]
                if not prev.empty:
                    prev_wac = float(prev.iloc[0]["WAC"]); prev_qty = float(prev.iloc[0]["Qty"])
                else:
                    prev_wac = 0.0; prev_qty = 0.0
                if t_side == "BUY":
                    new_qty = prev_qty + t_qty
                    new_wac = (prev_wac * prev_qty + t_price * t_qty) / new_qty if new_qty > 0 else 0.0
                else:  # SELL
                    sell_qty = min(prev_qty, t_qty)
                    new_qty = max(prev_qty - sell_qty, 0.0)
                    new_wac = prev_wac if new_qty > 0 else 0.0
                ok2, msg2 = sb_upsert_holdings(pd.DataFrame([{"Ticker": t_symbol, "WAC": new_wac, "Qty": new_qty}]))
                if ok and ok2:
                    st.success("Trade saved & holdings updated.")
                elif ok and not ok2:
                    st.warning("Trade saved, but holdings update failed: " + msg2)
                # refresh caches
                sb_get_trades.clear(); sb_get_holdings.clear()
                st.rerun()
