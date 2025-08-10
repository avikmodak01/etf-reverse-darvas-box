# app.py
# Run: streamlit run app.py

import os
import requests
import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from datetime import date, timedelta

# ---- Local utility (place NseUtility.py & NseCommandlist.py next to this file)
import NseUtility


# ======================================================================================
# Streamlit UI config
# ======================================================================================
st.set_page_config(page_title="Reverse Darvas ETF Backtester — NSE", layout="wide")
st.title("Reverse Darvas ETF Backtester — NSE Bhav Copy")
st.caption(
    "Data: NSE daily bhav via `NseUtility`. Rules: Weekly low (Mon–Fri) → next-week limit-buy at (prior-week low − offset). "
    "Take-profit when **CMP (daily Close)** ≥ WAC × (1 + TP%), sell **ALL** at the TP price."
)
st.markdown(
    """
<style>
.block-container {padding-top: 1rem; padding-bottom: 2rem;}
.small-note {font-size: 0.9rem; color: #777;}
.highlight-badge {display:inline-block; padding:0.25rem 0.5rem; border-radius:0.5rem; background:#111; color:#fff; font-weight:600; font-size:0.8rem; margin-left:0.25rem;}
</style>
""",
    unsafe_allow_html=True,
)


# ======================================================================================
# Strategy config & core backtest logic
# ======================================================================================
@dataclass
class Config:
    per_buy_budget: float = 2000.0          # ₹ per buy
    buy_offset_abs: float = 0.10            # absolute ₹ below prior week low
    take_profit_pct: float = 0.0628         # 6.28% above WAC
    one_buy_per_week: bool = True           # at most one buy per week per ETF
    allow_same_day_tp_after_buy: bool = True  # if Close hits TP same day as buy, allow exit


def _prep_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize to ['Open','High','Low','Close'] with DatetimeIndex."""
    if df is None or df.empty:
        return pd.DataFrame()

    # Map common NSE bhav column names
    colmap = {}
    for col in df.columns:
        u = str(col).strip().upper().replace(" ", "")
        if u in ("OPEN", "OPEN_PRICE", "OPENPRICE"):
            colmap[col] = "Open"
        elif u in ("HIGH", "HIGH_PRICE", "HIGHPRICE"):
            colmap[col] = "High"
        elif u in ("LOW", "LOW_PRICE", "LOWPRICE"):
            colmap[col] = "Low"
        elif u in ("CLOSE", "CLOSE_PRICE", "CLOSEPRICE", "LAST", "LAST_PRICE", "LASTPRICE"):
            colmap[col] = "Close"

    out = df.rename(columns=colmap)
    need = {"Open", "High", "Low", "Close"}
    if not need.issubset(out.columns):
        return pd.DataFrame()

    out = out[["Open", "High", "Low", "Close"]].copy()

    # Ensure DatetimeIndex
    if not isinstance(out.index, pd.DatetimeIndex):
        for cand in ("TIMESTAMP", "DATE", "TradeDate", "Timestamp"):
            if cand in df.columns:
                try:
                    out.index = pd.to_datetime(df[cand])
                    break
                except Exception:
                    pass
        if not isinstance(out.index, pd.DatetimeIndex):
            out.index = pd.to_datetime(out.index, errors="coerce")

    out.sort_index(inplace=True)
    out.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)
    return out


def _attach_week_levels(d: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Compute previous week's low and attach next-week buy level to each row."""
    d = d.copy()
    d["week_end"] = d.index.to_period("W-FRI").to_timestamp("W-FRI")
    weekly_low = d.groupby("week_end")["Low"].min()
    buy_level_by_week = (weekly_low.shift(1) - cfg.buy_offset_abs).dropna()
    d = d.join(buy_level_by_week.rename("week_buy_level"), on="week_end")
    return d


def simulate_ticker(daily: pd.DataFrame, cfg: Config, ticker: str) -> pd.DataFrame:
    """
    Reverse Darvas with TP on CMP (Close):
    - BUY: next week limit at (prev-week low - offset), 1 buy/week optional
    - SELL: when Close >= WAC * (1 + TP%), sell ALL at TP price
    """
    d = _prep_daily(daily)
    if d.empty:
        return pd.DataFrame(columns=["Date", "Ticker", "Side", "Price", "Qty", "CashFlow", "HoldingsQty", "AvgCost", "Note"])
    d = _attach_week_levels(d, cfg)
    d = d.dropna(subset=["week_buy_level"]).copy()

    holdings_qty = 0
    avg_cost = 0.0
    last_week_end = None
    bought_this_week = False
    records = []

    for dt, row in d.iterrows():
        week_end = row["week_end"]
        h, l, c = float(row["High"]), float(row["Low"]), float(row["Close"])
        buy_level = float(row["week_buy_level"])

        # New week → reset weekly buy flag
        if (last_week_end is None) or (week_end != last_week_end):
            bought_this_week = False
            last_week_end = week_end

        buy_filled = False
        # BUY at limit if day's Low touches buy_level
        if (not bought_this_week or not cfg.one_buy_per_week) and l <= buy_level:
            qty = int(cfg.per_buy_budget // buy_level)
            if qty > 0:
                cost = qty * buy_level
                new_total_cost = avg_cost * holdings_qty + cost
                holdings_qty += qty
                avg_cost = new_total_cost / holdings_qty

                records.append({
                    "Date": dt, "Ticker": ticker, "Side": "BUY",
                    "Price": round(buy_level, 2),
                    "Qty": qty, "CashFlow": -round(cost, 2),
                    "HoldingsQty": holdings_qty,
                    "AvgCost": round(avg_cost, 4),
                    "Note": f"Buy @ weekly trigger {buy_level:.2f}"
                })
                buy_filled = True
                bought_this_week = True

        # SELL on CMP (Close) reaching TP
        if holdings_qty > 0:
            tp_price = avg_cost * (1.0 + cfg.take_profit_pct)
            can_tp_today = True
            if buy_filled and not cfg.allow_same_day_tp_after_buy:
                can_tp_today = False
            if can_tp_today and c >= tp_price:
                proceeds = holdings_qty * tp_price
                pnl = proceeds - (avg_cost * holdings_qty)
                records.append({
                    "Date": dt, "Ticker": ticker, "Side": "SELL",
                    "Price": round(tp_price, 2),
                    "Qty": holdings_qty, "CashFlow": round(proceeds, 2),
                    "HoldingsQty": 0, "AvgCost": 0.0,
                    "Note": f"TP (CMP>=) hit; sold @ {tp_price:.2f}; PnL {pnl:.2f}"
                })
                holdings_qty = 0
                avg_cost = 0.0

    trades = pd.DataFrame.from_records(records)
    if not trades.empty:
        trades.sort_values(["Date", "Ticker", "Side"], inplace=True)
        trades.reset_index(drop=True, inplace=True)
    return trades


def summarize_roundtrips(trades: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-ticker roundtrips (flat → buys → sell all)."""
    cols = [
        "Ticker", "ExitDate", "RoundtripQty", "AvgEntry",
        "ExitPrice", "RealizedPnL", "ReturnPct_on_Cost", "HoldingDays"
    ]
    if trades is None or trades.empty:
        return pd.DataFrame(columns=cols)

    out = []
    trades = trades.copy()
    trades["grp"] = (trades["Side"] == "SELL").cumsum()

    for (tkr, grp_id), g in trades.groupby(["Ticker", "grp"], dropna=False):
        if (g["Side"] == "SELL").any():
            buys = g[g["Side"] == "BUY"]
            sell = g[g["Side"] == "SELL"].iloc[-1]

            cost_in = -buys["CashFlow"].sum()          # positive
            qty_in = buys["Qty"].sum()
            avg_entry = (cost_in / qty_in) if qty_in > 0 else np.nan
            realized = g["CashFlow"].sum()             # BUYS negative, SELL positive
            ret_pct = 100.0 * realized / cost_in if cost_in > 0 else np.nan

            first_buy_date = buys["Date"].min() if not buys.empty else sell["Date"]
            holding_days = (pd.to_datetime(sell["Date"]) - pd.to_datetime(first_buy_date)).days

            out.append({
                "Ticker": tkr,
                "ExitDate": sell["Date"],
                "RoundtripQty": int(qty_in),
                "AvgEntry": round(avg_entry, 4) if pd.notna(avg_entry) else np.nan,
                "ExitPrice": sell["Price"],
                "RealizedPnL": round(realized, 2),
                "ReturnPct_on_Cost": round(ret_pct, 2) if pd.notna(ret_pct) else np.nan,
                "HoldingDays": holding_days
            })

    if not out:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(out)
    return df.sort_values(["Ticker", "ExitDate"]).reset_index(drop=True)


# ======================================================================================
# NSE data via NseUtility (equity bhav preferred; delivery bhav fallback)
# ======================================================================================
nse = NseUtility.NseUtils()

@st.cache_data(show_spinner=False)
def fetch_bhav_for_date(d: date) -> Optional[pd.DataFrame]:
    """Fetch NSE bhav for a date. Try equity bhav first, then with-delivery. Returns UPPERCASE/no-space columns."""
    ds = d.strftime("%d-%m-%Y")
    df = None
    try:
        if hasattr(nse, "equity_bhav_copy"):
            df = nse.equity_bhav_copy(ds)
    except Exception:
        df = None

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


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.upper(): c for c in df.columns}
    for cand in candidates:
        u = cand.upper()
        if u in cols:
            return cols[u]
    return None


@st.cache_data(show_spinner=False)
def load_history_for_symbols(symbols: List[str], start: date, end: date) -> Dict[str, pd.DataFrame]:
    """
    Build daily OHLC for each symbol from NSE bhav copies (equity preferred; delivery fallback).
    Handles both column schemes:
      - Equity bhav: OPEN/HIGH/LOW/CLOSE
      - Delivery bhav: OPEN_PRICE/HIGH_PRICE/LOW_PRICE/CLOSE_PRICE
    """
    syms = [s.upper().strip() for s in symbols]
    per_sym_rows: Dict[str, List[dict]] = {s: [] for s in syms}
    first_available_symbol_set: Optional[set] = None

    d = start
    total_days = (end - start).days + 1
    prog = st.progress(0, text="Downloading NSE bhav copies...")
    step_i = 0

    while d <= end:
        if d.weekday() < 5:  # Mon-Fri
            bhav = fetch_bhav_for_date(d)
            if bhav is not None and not bhav.empty:
                if first_available_symbol_set is None:
                    if "SYMBOL" in bhav.columns:
                        try:
                            first_available_symbol_set = set(map(str.upper, bhav["SYMBOL"].astype(str).tolist()))
                        except Exception:
                            first_available_symbol_set = set()
                    else:
                        first_available_symbol_set = set()

                sycol = "SYMBOL" if "SYMBOL" in bhav.columns else None
                opcol = _pick_col(bhav, ["OPEN", "OPEN_PRICE", "OPENPRICE"])
                hicol = _pick_col(bhav, ["HIGH", "HIGH_PRICE", "HIGHPRICE"])
                locol = _pick_col(bhav, ["LOW", "LOW_PRICE", "LOWPRICE"])
                clcol = _pick_col(bhav, ["CLOSE", "CLOSE_PRICE", "CLOSEPRICE", "LAST", "LAST_PRICE", "LASTPRICE"])

                if all([sycol, opcol, hicol, locol, clcol]):
                    for s in syms:
                        try:
                            sub = bhav[bhav[sycol].astype(str).str.upper() == s]
                            if not sub.empty:
                                r = sub.iloc[0]
                                per_sym_rows[s].append({
                                    "Date": d,
                                    "Open": float(r[opcol]),
                                    "High": float(r[hicol]),
                                    "Low": float(r[locol]),
                                    "Close": float(r[clcol]),
                                })
                        except Exception:
                            pass

        step_i += 1
        prog.progress(min(step_i / max(total_days, 1), 1.0))
        d += timedelta(days=1)

    out: Dict[str, pd.DataFrame] = {}
    for sym, rows in per_sym_rows.items():
        if not rows:
            out[sym] = pd.DataFrame()
        else:
            df = pd.DataFrame(rows).set_index("Date").sort_index()
            out[sym] = df[["Open", "High", "Low", "Close"]]

    missing = [s for s, df in out.items() if df.empty]
    if missing:
        msg = "No rows collected for: " + ", ".join(missing[:15]) + (" ..." if len(missing) > 15 else "")
        if first_available_symbol_set is not None:
            selected_in_bhav = [s for s in syms if s in first_available_symbol_set]
            st.warning(msg + f"\n\nTip: {len(selected_in_bhav)}/{len(syms)} matched the bhav universe on the first fetched day.")
        else:
            st.warning(msg)

    return out


# ======================================================================================
# Realized & Unrealized P&L helpers
# ======================================================================================
def _build_last_close_map(history: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    out = {}
    for sym, df in history.items():
        if df is not None and not df.empty and "Close" in df.columns:
            last_close = df["Close"].dropna()
            out[sym.upper()] = float(last_close.iloc[-1]) if not last_close.empty else float("nan")
        else:
            out[sym.upper()] = float("nan")
    return out


def positions_snapshot(trades: pd.DataFrame,
                       summary: pd.DataFrame,
                       history: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, float, float]:
    """
    End-of-period positions table + totals.
    Uses last trade snapshot (HoldingsQty, AvgCost). Unrealized P&L uses last Close in 'history'.
    """
    cols = ["Ticker", "Qty", "AvgCost", "LastClose", "Invested", "MktValue",
            "UnrealizedPnL", "UnrealizedPct", "RealizedPnL"]
    if trades is None or trades.empty:
        return pd.DataFrame(columns=cols), 0.0, 0.0

    last_close_map = _build_last_close_map(history)
    realized_by_ticker = {}
    if summary is not None and not summary.empty:
        realized_by_ticker = summary.groupby("Ticker", dropna=False)["RealizedPnL"].sum().to_dict()

    rows = []
    for t in sorted(trades["Ticker"].astype(str).unique()):
        g = trades[trades["Ticker"] == t].sort_values("Date")
        last = g.iloc[-1]
        qty = int(last.get("HoldingsQty", 0) or 0)
        avg_cost = float(last.get("AvgCost", 0.0) or 0.0)

        last_close = float(last_close_map.get(t.upper(), float("nan")))
        invested = float(qty * avg_cost)
        mkt_value = float(qty * last_close) if qty > 0 and not np.isnan(last_close) else 0.0
        unrealized = float(mkt_value - invested)
        unreal_pct = float((unrealized / invested) * 100.0) if invested > 0 else float("nan")
        realized = float(realized_by_ticker.get(t, 0.0))

        rows.append({
            "Ticker": t,
            "Qty": qty,
            "AvgCost": round(avg_cost, 4),
            "LastClose": round(last_close, 4) if not np.isnan(last_close) else np.nan,
            "Invested": round(invested, 2),
            "MktValue": round(mkt_value, 2),
            "UnrealizedPnL": round(unrealized, 2),
            "UnrealizedPct": round(unreal_pct, 2) if not np.isnan(unreal_pct) else np.nan,
            "RealizedPnL": round(realized, 2),
        })

    pos = pd.DataFrame(rows, columns=cols)
    total_realized = float(pos["RealizedPnL"].sum()) if not pos.empty else 0.0
    total_unrealized = float(pos["UnrealizedPnL"].sum()) if not pos.empty else 0.0
    return pos.sort_values("Ticker").reset_index(drop=True), total_realized, total_unrealized


# ======================================================================================
# Supabase WAC/Qty storage (Holdings)
# ======================================================================================
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "holdings")

def sb_ok() -> bool:
    return bool(SUPABASE_URL and SUPABASE_KEY)

def sb_headers():
    return {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}

def sb_get_holdings() -> pd.DataFrame:
    if not sb_ok():
        return pd.DataFrame(columns=["Ticker", "WAC", "Qty"])
    try:
        r = requests.get(f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}",
                         headers=sb_headers(),
                         params={"select": "Ticker,WAC,Qty"},
                         timeout=30)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
        if df.empty:
            return pd.DataFrame(columns=["Ticker", "WAC", "Qty"])
        df["Ticker"] = df["Ticker"].astype(str).str.upper()
        return df
    except Exception:
        return pd.DataFrame(columns=["Ticker", "WAC", "Qty"])

def sb_upsert_holdings(rows: pd.DataFrame) -> tuple[bool, str]:
    if not sb_ok():
        return False, "Supabase env vars not set."
    if rows is None or rows.empty:
        return False, "No rows to upsert."
    try:
        payload = rows.to_dict(orient="records")
        r = requests.post(f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}",
                          headers={**sb_headers(), "Content-Type": "application/json",
                                   "Prefer": "resolution=merge-duplicates"},
                          json=payload, timeout=30)
        if not r.ok:
            return False, f"HTTP {r.status_code}: {r.text}"
        return True, "Upserted."
    except Exception as e:
        return False, str(e)


# ======================================================================================
# Backtest runner
# ======================================================================================
def run_strategy(etfs: List[str], start: date, end: date, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, bool], Dict[str, pd.DataFrame]]:
    history = load_history_for_symbols(etfs, start, end)
    trade_logs = []
    ok_map: Dict[str, bool] = {}

    for t in etfs:
        df = history.get(t.upper(), pd.DataFrame())
        ok_map[t] = df is not None and not df.empty
        if not ok_map[t]:
            continue
        tlog = simulate_ticker(df, cfg, t.upper())
        if not tlog.empty:
            trade_logs.append(tlog)

    trades = pd.concat(trade_logs, ignore_index=True) if trade_logs else pd.DataFrame(
        columns=["Date", "Ticker", "Side", "Price", "Qty", "CashFlow", "HoldingsQty", "AvgCost", "Note"]
    )
    summary = summarize_roundtrips(trades)
    return trades, summary, ok_map, history


# ======================================================================================
# Sidebar controls
# ======================================================================================
DEFAULT_NAMES = [
    "ITBEES","JUNIORBEES","MIDCAPETF","SMALLCAP","INFRABEES","PSUBNKBEES",
    "NIFTYBEES","GOLDBEES","SILVERBEES","MON100","HNGSNGBEES","MODEFENCE",
    "METALIETF","MID150CASE","OILIETF","MOSMALL250","LOWVOLIETF","PHARMABEES",
    "ALPL30IETF","PVTBANIETF","MOM30IETF","MOCAPITAL","MIDSMALL"
]

st.sidebar.header("Backtest Settings")
start_date = st.sidebar.date_input("Start date", value=date(2020, 1, 1), key="start_date")
end_date = st.sidebar.date_input("End date", value=date.today(), key="end_date")

etfs = st.sidebar.multiselect(
    "ETFs (exact NSE SYMBOL as in bhav copy, no suffix)",
    options=DEFAULT_NAMES,
    default=DEFAULT_NAMES[:10],
    key="etf_multiselect"
)

st.sidebar.subheader("Strategy Parameters")
per_buy_budget = st.sidebar.number_input("Per-buy budget (₹)", min_value=100.0, step=100.0, value=2000.0, key="param_budget")
buy_offset_abs = st.sidebar.number_input("Buy offset below week low (₹)", min_value=0.0, step=0.05, value=0.10, key="param_offset")
tp_pct = st.sidebar.number_input("Take profit (%) over WAC", min_value=0.1, step=0.01, value=6.28, key="param_tp")
one_buy_per_week = st.sidebar.checkbox("At most 1 buy per week", value=True, key="param_onebuy")
allow_same_day_tp = st.sidebar.checkbox("Allow same-day TP after fresh buy", value=True, key="param_samedaytp")

cfg = Config(
    per_buy_budget=per_buy_budget,
    buy_offset_abs=buy_offset_abs,
    take_profit_pct=tp_pct / 100.0,
    one_buy_per_week=one_buy_per_week,
    allow_same_day_tp_after_buy=allow_same_day_tp
)

st.markdown(
    f"**Per Buy** ₹{per_buy_budget:,.0f}"
    f"<span class='highlight-badge'>Offset −₹{buy_offset_abs:.2f}</span>"
    f"<span class='highlight-badge'>TP +{tp_pct:.2f}%</span>"
    f"<span class='highlight-badge'>{'1 buy/wk' if one_buy_per_week else 'multi-buy/wk'}</span>",
    unsafe_allow_html=True
)

# --------------------------------------------------------------------------------------
# Multi-ETF backtest
# --------------------------------------------------------------------------------------
run_btn = st.sidebar.button("Run Backtest", type="primary", key="run_backtest")

if run_btn:
    with st.spinner("Running backtest..."):
        trades, summary, ok_map, history = run_strategy(etfs, start_date, end_date, cfg)

    bad = [t for t, ok in ok_map.items() if not ok]
    if bad:
        st.warning(f"No data found for {len(bad)} ticker(s): {', '.join(bad[:12])}{' ...' if len(bad) > 12 else ''}")

    # KPIs incl. realized & unrealized
    pos, total_realized, total_unrealized = positions_snapshot(trades, summary, history)

    c1, c2, c3, c4, c5 = st.columns(5)
    total_buys = int(trades[trades["Side"] == "BUY"]["Qty"].sum()) if not trades.empty else 0
    total_sells = int((trades["Side"] == "SELL").sum()) if not trades.empty else 0
    win_rate = float((summary["RealizedPnL"] > 0).mean() * 100) if not summary.empty else 0.0

    c1.metric("Total Buy Quantity", f"{total_buys:,}")
    c2.metric("Exit Trades", f"{total_sells:,}")
    c3.metric("Realized P&L (₹)", f"{total_realized:,.0f}")
    c4.metric("Unrealized P&L (₹)", f"{total_unrealized:,.0f}")
    c5.metric("Win Rate", f"{win_rate:.1f}%")

    st.markdown("### Open Positions (Mark-to-Market)")
    st.dataframe(pos, use_container_width=True)
    st.download_button(
        "Download Positions CSV",
        data=pos.to_csv(index=False).encode("utf-8"),
        file_name="reverse_darvas_positions.csv",
        mime="text/csv",
        disabled=pos.empty,
        key="dl_positions"
    )

    st.markdown("### Trades")
    st.dataframe(
        trades.assign(Date=pd.to_datetime(trades["Date"]).dt.date) if not trades.empty else trades,
        use_container_width=True
    )
    st.download_button(
        "Download Trades CSV",
        data=trades.to_csv(index=False).encode("utf-8"),
        file_name="reverse_darvas_trades.csv",
        mime="text/csv",
        disabled=trades.empty,
        key="dl_trades"
    )

    st.markdown("### Roundtrip Summary (Realized)")
    st.dataframe(
        summary.assign(ExitDate=pd.to_datetime(summary["ExitDate"]).dt.date) if not summary.empty else summary,
        use_container_width=True
    )
    st.download_button(
        "Download Summary CSV",
        data=summary.to_csv(index=False).encode("utf-8"),
        file_name="reverse_darvas_summary.csv",
        mime="text/csv",
        disabled=summary.empty,
        key="dl_summary"
    )

    # ----------------------------------------------------------------------------------
    # Supabase holdings (WAC/Qty) manager
    # ----------------------------------------------------------------------------------
    st.markdown("### Holdings (WAC / Qty) — Supabase")
    if not sb_ok():
        st.info("Set SUPABASE_URL and SUPABASE_KEY env vars to enable holdings storage.")
    else:
        colx, coly = st.columns([2, 1])
        with colx:
            existing = sb_get_holdings()
            st.caption("Existing records")
            st.dataframe(existing, use_container_width=True)

        with coly:
            st.caption("Quick actions")
            # Build rows from current open positions (Qty>0)
            candidate = (pos[['Ticker', 'Qty', 'AvgCost']].rename(columns={'AvgCost': 'WAC'})
                         if 'pos' in locals() and not pos.empty else pd.DataFrame(columns=['Ticker', 'WAC', 'Qty']))
            candidate = candidate[candidate['Qty'] > 0][['Ticker', 'WAC', 'Qty']]
            if st.button("Upsert open positions as holdings", key="btn_upsert_positions", type="primary", disabled=candidate.empty):
                ok, msg = sb_upsert_holdings(candidate.assign(Ticker=lambda d: d['Ticker'].str.upper()))
                st.success(msg) if ok else st.error(msg)

        st.caption("Manual edit & upsert")
        edit_df = st.data_editor(
            existing if not existing.empty else pd.DataFrame(columns=['Ticker', 'WAC', 'Qty']),
            num_rows="dynamic", key="editor_holdings", use_container_width=True
        )
        if st.button("Save edited holdings", key="btn_save_holdings"):
            cleaned = (edit_df[['Ticker', 'WAC', 'Qty']]
                       .dropna(subset=['Ticker'])
                       .assign(Ticker=lambda d: d['Ticker'].astype(str).str.upper()))
            ok, msg = sb_upsert_holdings(cleaned)
            st.success(msg) if ok else st.error(msg)

else:
    st.info("Pick date range & ETFs in the sidebar, then **Run Backtest**.")
    with st.expander("Notes"):
        st.markdown(
            "- This app pulls **daily bhav copy** from NSE via your `NseUtility.py`.\n"
            "- For long ranges & many ETFs, first run may take time (1 request per trading day). Subsequent runs are cached."
        )

# --------------------------------------------------------------------------------------
# Single ETF quick test (with chart)
# --------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("Single ETF Test")

colA, colB = st.columns([2, 1])
with colA:
    single_etf = st.selectbox(
        "Pick one ETF (exact NSE SYMBOL as in bhav copy)",
        options=DEFAULT_NAMES,
        index=0,
        key="single_etf_symbol"
    )
    custom_etf = st.text_input("…or type a custom symbol (overrides selection)", value="", key="single_etf_custom")
    single_symbol = (custom_etf.strip().upper() or single_etf.strip().upper())

with colB:
    run_single = st.button("Run Single ETF", key="run_single_etf")

if run_single:
    with st.spinner(f"Fetching & backtesting {single_symbol}…"):
        hist = load_history_for_symbols([single_symbol], start_date, end_date)
        df_single = hist.get(single_symbol.upper(), pd.DataFrame())
        if df_single is None or df_single.empty:
            st.error(
                f"No data for {single_symbol}. "
                "Tip: ensure it matches the NSE bhav ‘SYMBOL’. Try a shorter, recent date range."
            )
        else:
            tlog = simulate_ticker(df_single, cfg, single_symbol.upper())
            summ = summarize_roundtrips(tlog)

            # Positions snapshot for single symbol
            pos_single, realized_total, unreal_total = positions_snapshot(
                tlog, summ, {single_symbol.upper(): df_single}
            )

            # KPIs
            k1, k2, k3, k4, k5 = st.columns(5)
            total_qty = int(tlog[tlog["Side"] == "BUY"]["Qty"].sum()) if not tlog.empty else 0
            exits = int((tlog["Side"] == "SELL").sum()) if not tlog.empty else 0
            win_rate = float((summ["RealizedPnL"] > 0).mean() * 100) if not summ.empty else 0.0

            k1.metric("Total Buy Qty", f"{total_qty:,}")
            k2.metric("Exit Trades", f"{exits:,}")
            k3.metric("Realized P&L (₹)", f"{realized_total:,.0f}")
            k4.metric("Unrealized P&L (₹)", f"{unreal_total:,.0f}")
            k5.metric("Win Rate", f"{win_rate:.1f}%")

            st.markdown("#### Open Position (Mark-to-Market)")
            st.dataframe(pos_single, use_container_width=True)

            st.markdown("#### Trades")
            st.dataframe(
                tlog.assign(Date=pd.to_datetime(tlog["Date"]).dt.date) if not tlog.empty else tlog,
                use_container_width=True
            )
            st.download_button(
                "Download Single-ETF Trades CSV",
                data=tlog.to_csv(index=False).encode("utf-8"),
                file_name=f"{single_symbol}_trades.csv",
                mime="text/csv",
                disabled=tlog.empty,
                key="dl_single_trades"
            )

            st.markdown("#### Roundtrip Summary")
            st.dataframe(
                summ.assign(ExitDate=pd.to_datetime(summ["ExitDate"]).dt.date) if not summ.empty else summ,
                use_container_width=True
            )
            st.download_button(
                "Download Single-ETF Summary CSV",
                data=summ.to_csv(index=False).encode("utf-8"),
                file_name=f"{single_symbol}_summary.csv",
                mime="text/csv",
                disabled=summ.empty,
                key="dl_single_summary"
            )

            # Chart (Plotly optional)
            try:
                import plotly.graph_objects as go
                px = df_single.reset_index().rename(columns={"index": "Date"})
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=px["Date"], y=px["Close"], mode="lines", name="Close"))

                if not tlog.empty:
                    buys = tlog[tlog["Side"] == "BUY"]
                    sells = tlog[tlog["Side"] == "SELL"]
                    if not buys.empty:
                        fig.add_trace(go.Scatter(
                            x=buys["Date"], y=buys["Price"],
                            mode="markers", name="BUY",
                            marker_symbol="triangle-up", marker_size=10
                        ))
                    if not sells.empty:
                        fig.add_trace(go.Scatter(
                            x=sells["Date"], y=sells["Price"],
                            mode="markers", name="SELL (TP)",
                            marker_symbol="triangle-down", marker_size=10
                        ))

                fig.update_layout(
                    title=f"{single_symbol} — Close with BUY/SELL signals",
                    xaxis_title="Date", yaxis_title="Price (₹)",
                    height=520,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.info(f"Plotly not installed or chart error: {e}")
