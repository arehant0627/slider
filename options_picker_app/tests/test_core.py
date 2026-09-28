"""Headless checks of the picker logic on the demo market."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, pandas as pd
import core, data

W = data.demo_market(now=pd.Timestamp("2026-09-28 11:00", tz=data.NY))
now = pd.Timestamp("2026-09-28 11:00", tz=data.NY)
exp = W["expirations"][0]
assert exp.weekday() == 4 and exp >= now.date()
weekly, rv20, lwz = data.features_from_prices(W["close"], now)
assert weekly.index.max() < pd.Timestamp(now.date())
chains = {t: W["chain"](t, exp) for t in W["names"]}
use = dict(stock_puts=True, etf_puts=True, covered_calls=True)

# 1) plain: $100k, 0.5% target
cand, info, skipped = core.build_candidates(chains, pd.Timestamp(exp), now, W["rate"], rv20, lwz, W["earnings"], {}, use, {})
assert set(skipped) == W["earnings"], skipped
assert {"put", "call_new"} <= set(cand["kind"]) and "call_held" not in set(cand["kind"])
p = cand[cand["kind"] == "put"]
assert p.groupby("ticker")["delta"].min().ge(-0.41).all()
S_err = max(abs(info[t]["S"] / W["close"][t].iloc[-1] - 1) for t in info)
assert S_err < 0.01, S_err
td = info["QQQ"]["td"]
assert 4.5 < td < 5.0, td                 # Monday 11:00 -> ~0.77 of today + 4 days
res = core.optimize(cand, info, {}, 100_000, 500, weekly, {})
assert res["status"] == "target reached", res["status"]
assert res["time_value"] >= 500 - 1e-6 and res["capital_used"] <= 100_000 + 1
assert res["worst5_loss"] > 0 and 0 <= res["chance_of_loss"] <= 1 and res["crash10"] < 0
print(f"CHECK $100k @ 0.5%: {len(res['trades'])} trades, time value ${res['time_value']:,.0f}, capital ${res['capital_used']:,.0f}, "
      f"worst-5% ${res['worst5_loss']:,.0f}, P(loss) {res['chance_of_loss']:.0%}, -10% market ${res['crash10']:,.0f}")
print(res["trades"][["kind", "ticker", "strike", "contracts", "px", "delta"]].to_string(index=False))

# 2) risk rises with the target, and an impossible target reports the maximum
r2 = core.optimize(cand, info, {}, 100_000, 2_000, weekly, {})
assert r2["worst5_loss"] > res["worst5_loss"]
r3 = core.optimize(cand, info, {}, 100_000, 50_000, weekly, {})
assert r3["status"].startswith("target out of reach") and r3["capital_used"] <= 100_000 + 1
print(f"CHECK higher target = more risk (${r2['worst5_loss']:,.0f} at $2k); impossible target: '{r3['status']}'")

# 3) holdings: keep-and-call or sell; calls never exceed shares; earnings names get sold
hold = {"AAPL": 250, "NFLX": 100, "MSFT": 40}
cand_h, info_h, sk_h = core.build_candidates(chains, pd.Timestamp(exp), now, W["rate"], rv20, lwz, W["earnings"], hold, use, {})
assert "NFLX" in info_h and "NFLX" in sk_h                  # priced (for its value) but no calls: earnings week
assert cand_h.loc[cand_h["kind"] == "call_held", "ticker"].unique().tolist() == ["AAPL"]
r4 = core.optimize(cand_h, info_h, hold, 20_000, 600, weekly, {})
h = r4["holdings"].set_index("ticker")
assert h.loc["NFLX", "keep"] == 0 and h.loc["MSFT", "keep"] == 0 and h.loc["AAPL", "keep"] <= 200
kept_calls = r4["trades"].loc[r4["trades"]["kind"] == "call_held", "contracts"].sum()
assert 100 * kept_calls == h.loc["AAPL", "keep"]
assert r4["budget"] > 20_000 and r4["capital_used"] <= r4["budget"] + 1
print("CHECK holdings:", h[["shares", "keep", "sell", "decision"]].to_dict("index"))
print(f"  with $20k cash + holdings worth ${r4['held_value']:,.0f}: target {r4['status']}, capital ${r4['capital_used']:,.0f}")

# 3b) when calls on held shares are the only way to reach the target, it keeps shares and never over-writes calls
only_held = dict(stock_puts=False, etf_puts=False, covered_calls=False)
c7, i7, _ = core.build_candidates(chains, pd.Timestamp(exp), now, W["rate"], rv20, lwz, W["earnings"], hold, only_held, {})
assert set(c7["kind"]) == {"call_held"}
r7 = core.optimize(c7, i7, hold, 0, 150, weekly, {})
h7 = r7["holdings"].set_index("ticker")
k7 = r7["trades"].groupby("ticker")["contracts"].sum()
assert h7.loc["AAPL", "keep"] > 0 and 100 * k7["AAPL"] == h7.loc["AAPL", "keep"] and h7.loc["AAPL", "keep"] <= 200
assert r7["time_value"] >= 150
print(f"CHECK keep path: kept {h7.loc['AAPL', 'keep']} AAPL shares with {k7['AAPL']} calls ({h7.loc['AAPL', 'decision']}), "
      f"strikes {r7['trades']['strike'].tolist()} vs price {i7['AAPL']['S']:.2f}")

# 4) instruments switch off cleanly
c5, i5, _ = core.build_candidates(chains, pd.Timestamp(exp), now, W["rate"], rv20, lwz, W["earnings"], {},
                                  dict(stock_puts=False, etf_puts=True, covered_calls=False), {})
assert set(c5["kind"]) == {"put"} and set(c5["ticker"]) <= set(data.ETFS)
r5 = core.optimize(c5, i5, {}, 100_000, 300, weekly, {})
assert set(r5["trades"]["ticker"]) <= set(data.ETFS)
print(f"CHECK ETF puts only: {r5['trades'][['ticker', 'strike', 'contracts']].values.tolist()}")

# 5) mid fills collect more per contract than bid fills
c6, i6, _ = core.build_candidates(chains, pd.Timestamp(exp), now, W["rate"], rv20, lwz, W["earnings"], {}, use, dict(fill=0.5))
assert (c6["px"].mean() > cand["px"].mean())
# 6) default expiration: Friday afternoon rolls to next week
assert data.default_expiration_index(W["expirations"], pd.Timestamp("2026-10-02 13:00", tz=data.NY)) in (0, 1)
exps = [pd.Timestamp("2026-10-02").date(), pd.Timestamp("2026-10-09").date()]
assert data.default_expiration_index(exps, pd.Timestamp("2026-10-02 13:00", tz=data.NY)) == 1
assert data.default_expiration_index(exps, pd.Timestamp("2026-09-30 10:00", tz=data.NY)) == 0
assert data.default_expiration_index(exps, pd.Timestamp("2026-10-03 10:00", tz=data.NY)) == 1
print("CHECK fills, expiration defaults")
print("ALL CORE CHECKS PASSED")
