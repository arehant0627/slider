"""Check the ThetaData layer against a fake client: expirations, live snapshot, closed-market fallback, threading."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys, types, datetime as dt
import numpy as np, pandas as pd

class NoDataFoundError(Exception):
    pass

MODE = {"open": True}
CALLS = []

class FakeClient:
    def __init__(self, api_key=None, existing_authorized_client=None, dataframe_type="pandas"):
        pass
    def option_list_expirations(self, symbol):
        days = pd.date_range("2026-09-21", "2026-11-30")
        return pd.DataFrame({"symbol": symbol, "expiration": [d.strftime("%Y-%m-%d") for d in days if d.weekday() < 5]})
    def _chain(self, symbol, ts):
        ks = np.arange(90, 111, 1.0)
        rows = []
        for right in ("CALL", "PUT"):
            intrinsic = np.maximum(100 - ks, 0) if right == "CALL" else np.maximum(ks - 100, 0)
            mid = intrinsic + 1.2 * np.exp(-((ks - 100) / 6) ** 2) + 0.05
            rows.append(pd.DataFrame({"symbol": symbol, "expiration": "2026-10-02", "strike": ks, "right": right,
                                      "timestamp": ts, "bid_size": 5, "bid": np.round(mid - 0.03, 2), "ask_size": 5,
                                      "ask": np.round(mid + 0.03, 2)}))
        return pd.concat(rows, ignore_index=True)
    def option_snapshot_quote(self, symbol, expiration, **kw):
        CALLS.append(("snap", symbol))
        if symbol == "BAD":
            raise RuntimeError("UNAVAILABLE: transient")
        if not MODE["open"]:
            raise NoDataFoundError("No data found for: option_snapshot_quote")
        return self._chain(symbol, pd.Timestamp("2026-09-28 14:31", tz="America/New_York").to_pydatetime())
    def option_history_quote(self, symbol, expiration, interval, date, start_time, end_time):
        CALLS.append(("hist", symbol, date, interval))
        a = self._chain(symbol, pd.Timestamp(f"{date} 15:30", tz="America/New_York").to_pydatetime())
        b = self._chain(symbol, pd.Timestamp(f"{date} 16:00", tz="America/New_York").to_pydatetime())
        b["bid"] += 0.01
        return pd.concat([a, b], ignore_index=True)

sys.modules["thetadata"] = types.SimpleNamespace(ThetaClient=FakeClient)
import data

c = FakeClient()
exps = data.upcoming_expirations(c, today=dt.date(2026, 9, 28))
assert exps == [dt.date(2026, 10, 2), dt.date(2026, 10, 9), dt.date(2026, 10, 16)], exps
print("CHECK expirations: last listed date of each coming week:", exps)

q, src, ts = data.fetch_chain(c, "AAPL", exps[0])
assert src == "live" and len(q) == 42 and {"strike", "right", "bid", "ask"} <= set(q.columns)
MODE["open"] = False
q2, src2, ts2 = data.fetch_chain(c, "AAPL", exps[0])
assert src2 == "last close" and len(q2) == 42 and ts2.hour == 16
m = q.merge(q2, on=["strike", "right"], suffixes=("_live", "_close"))
assert len(m) == 42 and np.allclose(m["bid_close"], m["bid_live"] + 0.01)   # took the 16:00 bar, not 15:30
print(f"CHECK chain: live snapshot when open; when closed, last quotes of {CALLS[-1][2]} ({ts2:%H:%M})")

MODE["open"] = True
got = data.fetch_chains(c, ["AAPL", "MSFT", "BAD", "QQQ"], exps[0])
assert set(got) == {"AAPL", "MSFT", "BAD", "QQQ"} and got["BAD"][1].startswith("error") and got["QQQ"][1] == "live"
print("CHECK fetch_chains: 2 at a time, one bad symbol reported without stopping the rest")

# last trading day: Monday morning -> Friday; Saturday -> Friday; weekday after close -> today
assert data.last_trading_day(pd.Timestamp("2026-09-28 09:00", tz=data.NY)) == dt.date(2026, 9, 25)
assert data.last_trading_day(pd.Timestamp("2026-09-26 12:00", tz=data.NY)) == dt.date(2026, 9, 25)
assert data.last_trading_day(pd.Timestamp("2026-09-29 17:00", tz=data.NY)) == dt.date(2026, 9, 29)
print("CHECK last trading day for closed-market quotes")

# the priced chain from the fake gives a sensible forward price and deltas
import core
pq, s = core.price_chain(q, 4.2 / 365, 0.04)
assert abs(s["S"] - 100) < 0.5 and pq.loc[~pq["is_call"] & (pq["strike"] == 95), "delta"].between(-0.5, 0).all()
print(f"CHECK pricing the fake chain: price {s['S']:.2f}, ATM vol {s['atm_iv']:.0%}")
print("ALL DATA CHECKS PASSED")
