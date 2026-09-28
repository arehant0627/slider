"""Data for the weekly options picker: option chains from ThetaData, prices and earnings dates from Yahoo Finance,
the Nasdaq-100 list from Wikipedia, and the 3-month T-bill rate from FRED. Plus a demo market for trying the app."""
from __future__ import annotations

import datetime as dt
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

ETFS = ["QQQ", "SMH", "XLK"]
NY = "America/New_York"
HERE = os.path.dirname(os.path.abspath(__file__))


def now_ny() -> pd.Timestamp:
    return pd.Timestamp.now(tz=NY)


# ---------------------------------------------------------------- ThetaData
def make_client(api_key: str):
    from thetadata import ThetaClient
    return ThetaClient(api_key=api_key, dataframe_type="pandas")


def _call(fn, tries=3, **kw):
    """Call a ThetaData method; 'no data' comes back as an empty frame, other errors retry."""
    delay = 1.5
    for attempt in range(tries):
        try:
            out = fn(**kw)
            return out if out is not None else pd.DataFrame()
        except Exception as e:                                  # noqa: BLE001
            if "no data" in str(e).lower() or type(e).__name__ == "NoDataFoundError":
                return pd.DataFrame()
            if attempt == tries - 1:
                raise
            time.sleep(delay)
            delay *= 2


def upcoming_expirations(client, symbol="QQQ", n=3, today=None) -> list[dt.date]:
    """The last expiration of each of the next `n` weeks (Friday, or Thursday in a holiday week)."""
    df = _call(client.option_list_expirations, symbol=symbol)
    if df.empty:
        return []
    col = next((c for c in df.columns if "exp" in str(c).lower()), df.columns[0])
    d = pd.to_datetime(df[col].astype(str), errors="coerce").dropna().dt.date
    today = today or now_ny().date()
    d = sorted({x for x in d if x >= today})
    weeks = {}
    for x in d:
        weeks.setdefault(x.isocalendar()[:2], []).append(x)
    return [max(v) for _, v in sorted(weeks.items())][:n]


def default_expiration_index(exps: list[dt.date], now=None) -> int:
    """This week's expiration Monday-Thursday; next week's from Friday noon through the weekend."""
    now = now or now_ny()
    if not exps:
        return 0
    rolled = now.weekday() >= 5 or (now.weekday() == 4 and now.hour >= 12)
    if rolled and exps[0].isocalendar()[:2] == now.date().isocalendar()[:2] and len(exps) > 1:
        return 1
    return 0


def _norm(df: pd.DataFrame, expiration: dt.date):
    """Standard columns, only the requested expiration (a response that mixes expirations would scramble the
    chain), and a note of what came back, for the app's data check."""
    df = df.rename(columns=lambda c: str(c).lower())
    diag = {"rows returned": len(df)}
    if "expiration" in df.columns:
        e = pd.to_datetime(df["expiration"].astype(str), errors="coerce").dt.date
        diag["expirations returned"] = int(e.nunique())
        df = df[e == expiration]
    keep = [c for c in ["strike", "right", "bid", "ask", "timestamp"] if c in df.columns]
    diag["rows used"] = len(df)
    return df[keep].copy(), diag


def last_trading_day(now=None) -> dt.date:
    now = now or now_ny()
    d = now.date()
    if now.weekday() < 5 and now.hour * 60 + now.minute >= 16 * 60:
        return d
    d -= dt.timedelta(days=1)
    while d.weekday() >= 5:
        d -= dt.timedelta(days=1)
    return d


def fetch_chain(client, symbol: str, expiration: dt.date):
    """Live snapshot of the whole chain; if the market is closed (no snapshot), the last quotes of the previous session."""
    snap = _call(client.option_snapshot_quote, symbol=symbol, expiration=expiration)
    if len(snap):
        q, diag = _norm(snap, expiration)
        q = q[(pd.to_numeric(q["bid"], errors="coerce") > 0) | (pd.to_numeric(q["ask"], errors="coerce") > 0)]
        if len(q):
            ts = pd.to_datetime(q["timestamp"], utc=True).max().tz_convert(NY) if "timestamp" in q else now_ny()
            return q, "live", ts, diag
    day = last_trading_day()
    h = _call(client.option_history_quote, symbol=symbol, expiration=expiration, interval="30m", date=day,
              start_time="15:00:00", end_time="16:00:00")
    if not len(h):
        return pd.DataFrame(), "none", None, {}
    q, diag = _norm(h, expiration)
    q["timestamp"] = pd.to_datetime(q["timestamp"], utc=True)
    q = q[(pd.to_numeric(q["bid"], errors="coerce") > 0) | (pd.to_numeric(q["ask"], errors="coerce") > 0)]
    q = q.sort_values("timestamp").groupby(["strike", "right"], as_index=False).last()
    ts = q["timestamp"].max().tz_convert(NY) if len(q) else None
    return q, "last close", ts, diag


def fetch_chains(client, symbols, expiration, workers=2, progress=None):
    """{symbol: (quotes, source, time)} for every symbol, 2 at a time (the Value plan's limit)."""
    from thetadata import ThetaClient
    import threading
    local = threading.local()

    def one(sym):
        if not hasattr(local, "c"):
            local.c = ThetaClient(existing_authorized_client=client, dataframe_type="pandas")
        try:
            return sym, fetch_chain(local.c, sym, expiration)
        except Exception as e:                                  # noqa: BLE001
            return sym, (pd.DataFrame(), f"error: {str(e)[:80]}", None, {})

    out = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(one, s) for s in symbols]
        for i, f in enumerate(as_completed(futs), 1):
            s, r = f.result()
            out[s] = r
            if progress:
                progress(i / len(symbols), s)
    return out


# ---------------------------------------------------------------- other sources
def nasdaq100() -> list[str]:
    """Current Nasdaq-100 members: universe.csv next to the app if present, else Wikipedia."""
    p = os.path.join(HERE, "universe.csv")
    if os.path.exists(p):
        return sorted(pd.read_csv(p).iloc[:, 0].astype(str).str.strip().str.upper().unique())
    import requests
    r = requests.get("https://en.wikipedia.org/wiki/Nasdaq-100", timeout=20,
                     headers={"User-Agent": "Mozilla/5.0 (options picker; personal use)"})
    for t in pd.read_html(io.StringIO(r.text)):
        cols = {str(c).lower(): c for c in t.columns}
        key = next((cols[k] for k in ("ticker", "symbol") if k in cols), None)
        if key is not None and len(t) >= 90:
            return sorted(t[key].astype(str).str.strip().str.upper().unique())
    raise RuntimeError("Couldn't find the Nasdaq-100 table on Wikipedia; add a universe.csv with one ticker per line.")


def price_history(symbols, years=4):
    """Daily closes (split- and dividend-adjusted) from Yahoo Finance."""
    import yfinance as yf
    px = yf.download(sorted(set(symbols)), period=f"{years}y", auto_adjust=True, progress=False, threads=True)
    close = px["Close"] if isinstance(px.columns, pd.MultiIndex) else px[["Close"]].rename(columns={"Close": symbols[0]})
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.dropna(how="all", axis=1)


def features_from_prices(close: pd.DataFrame, now=None):
    """Weekly log returns (completed weeks only), 20-day realized vol, and last week's move in standard deviations."""
    now = now or now_ny()
    close = close.ffill()                    # today's row can be blank for some names while the market is open
    lr = np.log(close).diff()
    rv20 = (lr.rolling(20).std() * np.sqrt(252)).iloc[-1].to_dict()
    wk = np.log(close.resample("W-FRI").last()).diff()
    today = pd.Timestamp(now.date())
    wk = wk[wk.index <= today if (now.weekday() == 4 and now.hour >= 16) or now.weekday() >= 5 else wk.index < today]
    z = (wk.iloc[-1] / (pd.Series(rv20) * np.sqrt(5 / 252))).to_dict() if len(wk) else {}
    return wk, rv20, z


def earnings_between(symbols, start: dt.datetime, end: dt.datetime, workers=8):
    """Tickers reporting between now and expiration (Yahoo Finance), plus the ones it couldn't check."""
    import yfinance as yf

    def one(t):
        try:
            ed = yf.Ticker(t).get_earnings_dates(limit=8)
            if ed is None or ed.empty:
                return t, False, True
            d = pd.to_datetime(ed.index).tz_localize(None) if ed.index.tz is None else ed.index.tz_convert(NY).tz_localize(None)
            return t, bool(((d >= pd.Timestamp(start).tz_localize(None)) & (d <= pd.Timestamp(end).tz_localize(None))).any()), False
        except Exception:                                       # noqa: BLE001
            return t, False, True

    hit, unknown = set(), set()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for t, has, unk in pool.map(one, [s for s in symbols if s not in ETFS]):
            if has:
                hit.add(t)
            if unk:
                unknown.add(t)
    return hit, unknown


def tbill_rate() -> float:
    try:
        raw = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO")
        v = pd.to_numeric(raw.iloc[:, 1], errors="coerce").dropna()
        return float(v.iloc[-1]) / 100
    except Exception:                                           # noqa: BLE001
        return 0.04


def backtest_table():
    p = os.path.join(HERE, "target_table.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


# ---------------------------------------------------------------- demo market
def demo_market(seed=3, now=None):
    """A made-up market with the same shapes as the real feeds, for trying the app without a ThetaData key."""
    from scipy.stats import norm
    rng = np.random.default_rng(seed)
    now = now or now_ny()
    names = ETFS + ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "AMD", "COST", "NFLX", "ADBE", "CSCO",
                    "INTC", "QCOM", "TXN", "AMAT", "MU", "PEP", "SBUX", "PYPL"]
    days = pd.bdate_range(end=pd.Timestamp(now.date()) - pd.Timedelta(days=1), periods=4 * 252)
    m = 0.18 / np.sqrt(252) * rng.standard_normal(len(days)) + 0.0004
    close, vol = {}, {}
    for i, t in enumerate(names):
        beta = 1.0 if t == "QQQ" else rng.uniform(0.8, 1.6)
        idio = 0.02 if t in ETFS else rng.uniform(0.15, 0.40)
        r = beta * m + idio / np.sqrt(252) * rng.standard_normal(len(days))
        close[t] = pd.Series(rng.uniform(60, 450) * np.exp(np.cumsum(r)), index=days)
        vol[t] = float(np.sqrt((beta * 0.18) ** 2 + idio ** 2))
    close = pd.DataFrame(close)
    close["QQQ"] = close["QQQ"] / close["QQQ"].iloc[-1] * 480
    first = now.date() + dt.timedelta(days=(4 - now.weekday()) % 7)
    if first == now.date() and now.hour >= 16:
        first += dt.timedelta(days=7)
    exps = [first + dt.timedelta(days=7 * i) for i in range(3)]

    def chain(t, expiration):
        S = float(close[t].iloc[-1])
        T = max(((pd.Timestamp(expiration).tz_localize(NY) + pd.Timedelta(hours=16)) - now).total_seconds() / (365 * 86400), 2 / 365)
        step = 0.5 if S < 50 else (1.0 if S < 200 else (2.5 if S < 400 else 5.0))
        ks = np.arange(np.floor(S * 0.8 / step) * step, S * 1.2, step)
        sk = vol[t] * 1.15 * (1 + 0.6 * np.maximum(0, np.log(S / ks)))
        d1 = (np.log(S / ks) + 0.5 * sk * sk * T) / (sk * np.sqrt(T))
        call = S * norm.cdf(d1) - ks * norm.cdf(d1 - sk * np.sqrt(T))
        put = call - (S - ks)
        rows = []
        for right, pr in (("CALL", call), ("PUT", put)):
            half = np.maximum(0.01, 0.02 * pr)
            rows.append(pd.DataFrame({"strike": ks, "right": right, "bid": np.round(np.maximum(pr - half, 0), 2),
                                      "ask": np.round(pr + half, 2)}))
        return pd.concat(rows, ignore_index=True)

    earnings = {"NFLX", "INTC"}
    return dict(names=names, close=close, expirations=exps, chain=chain, earnings=earnings, rate=0.04)
