"""Pure logic for the weekly options picker: price chains, build candidates, find the least-risk combination.

Same method as the backtests (Phases 2-8): forward price from put-call parity, Black-76 implied vol and delta,
scenarios = the last 3 years of real weekly co-moves scaled to each name's current volatility, and a
linear/integer program that reaches the premium target with the smallest average loss in the worst 5% of scenarios.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.special import ndtr

ETFS = ["QQQ", "SMH", "XLK"]
NY = "America/New_York"

DEFAULTS = dict(
    fill=0.0,                 # share of the bid-ask spread captured when selling: 0 = bid, 0.5 = mid
    min_premium=0.05,
    stock_put_delta=(-0.25, -0.08),
    etf_put_delta=(-0.40, -0.08),
    cc_rungs=(0, 1, 2),       # 0 = first strike below the price, 1 = first at/above, 2 = second above
    max_option_spread=0.25, max_atm_spread=0.15, min_bid=0.05,
    opt_commission=0.65, stock_commission=0.005, stock_slippage_bps=2.0,
    cvar_alpha=0.95, scen_weeks=156, runup_z=1.5,
    max_price_gap=0.10,       # drop a name if its option quotes imply a price >10% away from the stock's last trade
)


# ---------------------------------------------------------------- pricing
def black_price(F, K, T, DF, sig, is_call):
    vs = sig * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * vs * vs) / vs
    call = DF * (F * ndtr(d1) - K * ndtr(d1 - vs))
    return np.where(is_call, call, call - DF * (F - K)), d1


def implied_vol(price, F, K, T, DF, is_call, iters=40):
    price, F, K, T, DF = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (price, F, K, T, DF)))
    is_call = np.broadcast_to(np.asarray(is_call, dtype=bool), price.shape)
    intrinsic = DF * np.where(is_call, np.maximum(F - K, 0), np.maximum(K - F, 0))
    upper = DF * np.where(is_call, F, K)
    ok = np.isfinite(price) & np.isfinite(F) & (T > 0) & (F > 0) & (K > 0) & (price > intrinsic + 1e-6) & (price < upper)
    out = np.full(price.shape, np.nan)
    if not ok.any():
        return out
    p, f, k, t, df, c = price[ok], F[ok], K[ok], T[ok], DF[ok], is_call[ok]
    lo, hi = np.full(p.shape, 1e-4), np.full(p.shape, 8.0)
    s = np.clip(np.sqrt(2 * np.pi / t) * p / (df * f), 0.05, 3.0)
    with np.errstate(all="ignore"):
        for _ in range(iters):
            val, d1 = black_price(f, k, t, df, s, c)
            diff = val - p
            hi = np.where(diff > 0, s, hi)
            lo = np.where(diff <= 0, s, lo)
            vega = df * f * np.exp(-0.5 * d1 * d1) / np.sqrt(2 * np.pi) * np.sqrt(t)
            newton = s - diff / vega
            s = np.where(~np.isfinite(newton) | (newton <= lo) | (newton >= hi), 0.5 * (lo + hi), newton)
        val, _ = black_price(f, k, t, df, s, c)
    out[ok] = np.where(np.abs(val - p) <= np.maximum(1e-3, 1e-3 * p), s, np.nan)
    return out


def bs_delta(F, K, T, sig, is_call):
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(F / K) + 0.5 * sig * sig * T) / (sig * np.sqrt(T))
    n = ndtr(d1)
    return np.where(is_call, n, n - 1.0)


def price_chain(q: pd.DataFrame, T: float, rate: float, forward_strikes: int = 5):
    """q: one symbol's chain for one expiration (strike, right, bid, ask). Returns (priced rows, summary) or (None, None)."""
    q = q.copy()
    q["strike"] = pd.to_numeric(q["strike"], errors="coerce")
    q["bid"] = pd.to_numeric(q["bid"], errors="coerce")
    q["ask"] = pd.to_numeric(q["ask"], errors="coerce")
    q["is_call"] = q["right"].astype(str).str.upper().str.startswith("C")
    q = q.dropna(subset=["strike"])
    two_sided = (q["ask"] > 0) & (q["ask"] >= q["bid"]) & (q["bid"] >= 0)
    q["mid"] = np.where(two_sided, (q["bid"] + q["ask"]) / 2, np.nan)
    q["valid"] = two_sided & (q["bid"] > 0)
    q["spread_pct"] = (q["ask"] - q["bid"]) / q["mid"]
    DF = float(np.exp(-rate * T))
    c = q.loc[q["is_call"] & q["valid"], ["strike", "mid"]]
    p = q.loc[~q["is_call"] & q["valid"], ["strike", "mid"]]
    m = c.merge(p, on="strike", suffixes=("_c", "_p"))
    if m.empty:
        return None, None
    m["F_k"] = m["strike"] + (m["mid_c"] - m["mid_p"]) / DF
    m["gap"] = (m["mid_c"] - m["mid_p"]).abs()
    F = float(m.nsmallest(forward_strikes, "gap")["F_k"].median())
    q["iv"] = implied_vol(q["mid"], F, q["strike"], T, DF, q["is_call"])
    q["otm"] = np.where(q["is_call"], q["strike"] >= F, q["strike"] < F)
    siv = q.loc[q["otm"] & q["iv"].notna()].drop_duplicates("strike").set_index("strike")["iv"]
    q["strike_iv"] = q["strike"].map(siv).fillna(q["iv"])
    q["delta"] = bs_delta(F, q["strike"], T, q["strike_iv"], q["is_call"])
    otm = q[q["otm"] & q["iv"].notna()].sort_values("strike")
    atm = float(np.interp(F, otm["strike"], otm["iv"])) if len(otm) >= 2 and otm["strike"].iat[0] <= F <= otm["strike"].iat[-1] else np.nan
    near = q[q["otm"] & q["valid"]]
    near = near.iloc[(near["strike"] - F).abs().argsort()[:2]]
    return q, dict(F=F, DF=DF, S=F * DF, atm_iv=atm, atm_spread=float(near["spread_pct"].mean()) if len(near) else np.nan)


def trading_days_left(now: pd.Timestamp, expiration: pd.Timestamp, holidays=()) -> float:
    """Rest of today's session (if open) plus each full trading day through expiration."""
    now = pd.Timestamp(now).tz_convert(NY) if pd.Timestamp(now).tzinfo else pd.Timestamp(now).tz_localize(NY)
    days = pd.bdate_range(now.normalize().tz_localize(None) + pd.Timedelta(days=1), pd.Timestamp(expiration))
    days = [d for d in days if d.date() not in set(holidays)]
    minutes = now.hour * 60 + now.minute
    today = 0.0 if now.weekday() >= 5 or minutes >= 960 else min(1.0, (960 - max(minutes, 570)) / 390)
    return len(days) + today


def build_candidates(chains: dict, expiration: pd.Timestamp, now: pd.Timestamp, rate: float, rv20: dict,
                     last_week_z: dict, skip: set, holdings: dict, use: dict, cfg: dict, ref_px: dict | None = None):
    """chains: {ticker: DataFrame(strike, right, bid, ask)}. Returns (candidates, per-name info, skipped reasons)."""
    cfg = {**DEFAULTS, **cfg}
    exp_close = pd.Timestamp(expiration).tz_localize(NY) + pd.Timedelta(hours=16)
    now = pd.Timestamp(now).tz_convert(NY) if pd.Timestamp(now).tzinfo else pd.Timestamp(now).tz_localize(NY)
    T = max((exp_close - now).total_seconds() / (365 * 86400), 1e-4)
    td = max(trading_days_left(now, expiration), 0.05)
    rows, info, skipped = [], {}, {}
    for t, q in chains.items():
        if t in skip and t not in holdings:
            skipped[t] = "earnings before expiration"
            continue
        if q is None or not len(q):
            skipped[t] = "no option quotes"
            continue
        pq, s = price_chain(q, T, rate)
        if pq is None:
            skipped[t] = "couldn't price the chain"
            continue
        ref = (ref_px or {}).get(t)
        if ref is not None and np.isfinite(ref) and ref > 0 and abs(s["S"] / ref - 1) > cfg["max_price_gap"]:
            # a safety check: bad or mismatched quotes must never turn into trades
            skipped[t] = f"option quotes imply ${s['S']:,.2f} but the stock last traded at ${ref:,.2f}"
            continue
        info[t] = {**s, "rv20": rv20.get(t, np.nan), "T": T, "td": td}
        if not (s["atm_spread"] <= cfg["max_atm_spread"]):
            skipped[t] = f"wide spreads ({s['atm_spread']:.0%} at the money)"
            continue
        if t in skip:                        # held name with earnings: can't sell calls into earnings, so it gets sold
            skipped[t] = "earnings before expiration (held shares will be sold)"
            continue
        e = pq[pq["valid"] & (pq["spread_pct"] <= cfg["max_option_spread"]) & (pq["bid"] >= cfg["min_bid"])].copy()
        e["ticker"], e["S"] = t, s["S"]
        is_etf = t in ETFS
        lo, hi = cfg["etf_put_delta"] if is_etf else cfg["stock_put_delta"]
        if use.get("etf_puts" if is_etf else "stock_puts", True):
            p = e[~e["is_call"] & (e["delta"] >= lo) & (e["delta"] <= hi)].copy()
            p["bucket"] = (p["delta"] / 0.05).round()
            p["dist"] = (p["delta"] - p["bucket"] * 0.05).abs()
            rows.append(p.sort_values("dist").drop_duplicates("bucket").assign(kind="put"))
        calls = e[e["is_call"]].sort_values("strike")
        above = calls[calls["strike"] >= s["S"]].copy()
        above["rung"] = np.arange(1, len(above) + 1)
        below = calls[calls["strike"] < s["S"]].sort_values("strike", ascending=False).copy()
        below["rung"] = -np.arange(0, len(below))
        cc = pd.concat([above, below])
        cc = cc[cc["rung"].isin(cfg["cc_rungs"])]
        if t in holdings and holdings[t] >= 100:
            rows.append(cc.assign(kind="call_held"))
        if use.get("covered_calls", True) and t not in holdings:
            z = last_week_z.get(t, 0.0)
            if not (cfg["runup_z"] is not None and np.isfinite(z) and z > cfg["runup_z"]):
                rows.append(cc.assign(kind="call_new"))
    cand = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(cand):
        cand["px"] = cand["bid"] + cfg["fill"] * (cand["ask"] - cand["bid"])
        cand = cand[cand["px"] >= cfg["min_premium"]]
        is_call = cand["kind"] != "put"
        cand["tv"] = 100 * (cand["px"] - np.where(is_call, np.maximum(cand["S"] - cand["strike"], 0),
                                                   np.maximum(cand["strike"] - cand["S"], 0)))
        cand = cand[cand["tv"] > 0].reset_index(drop=True)
        cand["expiration"] = pd.Timestamp(expiration)
    return cand, info, skipped


# ---------------------------------------------------------------- scenarios and optimizer
def scenario_shocks(weekly_returns: pd.DataFrame, tickers, weeks=156, min_obs=26):
    hist = weekly_returns.tail(weeks)
    cols = [t for t in tickers if t in hist.columns and hist[t].notna().sum() >= min_obs]
    if not cols:
        return None, None
    h = hist[cols]
    sd = h.std()
    z = (h - h.mean()) / sd
    row_mean = z.mean(axis=1)
    z = z.apply(lambda col: col.fillna(row_mean)).dropna(how="all").fillna(0.0)
    return z, sd


def _solve(P, tv, need, gross, groups, cap_left, nmax, cash, objective, extra, integer, alpha, time_limit=20):
    S, n = P.shape
    k = 1.0 / ((1 - alpha) * S)
    nv = n + 1 + S
    rows = [sparse.hstack([sparse.csr_matrix(-P), sparse.csr_matrix(-np.ones((S, 1))), -sparse.identity(S)]),
            sparse.csr_matrix(np.r_[need, 0.0, np.zeros(S)][None, :]),
            sparse.coo_matrix((gross, (groups, np.arange(n))), shape=(len(cap_left), nv))]
    lo = [np.full(S, -np.inf), [-np.inf], np.full(len(cap_left), -np.inf)]
    hi = [np.zeros(S), [cash], cap_left]
    cvar_row = np.r_[np.zeros(n), 1.0, np.full(S, k)]
    tv_row = np.r_[tv, 0.0, np.zeros(S)]
    for which, a, b in extra:
        rows.append(sparse.csr_matrix((cvar_row if which == "cvar" else tv_row)[None, :]))
        lo.append([a]); hi.append([b])
    c = {"min_cvar": cvar_row, "max_tv": -tv_row}[objective]
    res = milp(c=c, constraints=[LinearConstraint(sparse.vstack(rows).tocsr(), np.concatenate(lo), np.concatenate(hi))],
               integrality=np.r_[np.full(n, 1 if integer else 0), np.zeros(1 + S)],
               bounds=Bounds(np.r_[np.zeros(n), -np.inf, np.zeros(S)], np.r_[nmax, np.inf, np.full(S, np.inf)]),
               options={"time_limit": time_limit, "mip_rel_gap": 0.01})
    return res.x[:n] if (res.x is not None and res.status in (0, 1)) else None


def _two_stage(P, tv, need, gross, groups, cap_left, nmax, cash, objective, extra, alpha):
    x_lp = _solve(P, tv, need, gross, groups, cap_left, nmax, cash, objective, extra, False, alpha)
    if x_lp is None:
        return None
    used = x_lp > 1e-6
    for idx in (np.flatnonzero(used), np.flatnonzero(np.isin(groups, groups[used])), np.arange(len(tv))):
        if len(idx) == 0:
            continue
        xs = _solve(P[:, idx], tv[idx], need[idx], gross[idx], groups[idx], cap_left, nmax[idx], cash, objective, extra, True, alpha)
        if xs is not None:
            x = np.zeros(len(tv))
            x[idx] = np.round(xs)
            return x.astype(int)
    return None


def betas_vs(weekly_returns: pd.DataFrame, market="QQQ", weeks=156):
    h = weekly_returns.tail(weeks)
    if market not in h:
        return {}
    m = h[market]
    out = {}
    for t in h.columns:
        pair = pd.concat([h[t], m], axis=1).dropna()
        out[t] = float(np.cov(pair.iloc[:, 0], pair.iloc[:, 1])[0, 1] / pair.iloc[:, 1].var()) if len(pair) > 26 else 1.0
    return out


def optimize(cand: pd.DataFrame, info: dict, holdings: dict, cash: float, target: float, weekly_returns: pd.DataFrame, cfg: dict):
    """Find the least-risk set of trades whose time value reaches `target`, within cash plus the value of held shares.

    holdings: {ticker: shares}. Held shares the plan doesn't sell a call on are sold at the current price.
    Returns a dict with the trade list, what to do with each holding, and the risk numbers."""
    cfg = {**DEFAULTS, **cfg}
    alpha = cfg["cvar_alpha"]
    held_value = sum(sh * info.get(t, {}).get("S", np.nan) for t, sh in holdings.items() if t in info)
    budget = cash + held_value
    out = dict(budget=budget, held_value=held_value, target=target)
    if cand is None or cand.empty:
        return {**out, "status": "no candidates"}
    tickers = sorted(set(cand["ticker"]))
    z, sd = scenario_shocks(weekly_returns, tickers, cfg["scen_weeks"])
    if z is None:
        return {**out, "status": "no price history for scenarios"}
    o = cand[cand["ticker"].isin(z.columns)].reset_index(drop=True)
    sig_w = {t: 0.5 * (i["atm_iv"] * np.sqrt(i["T"]) + i["rv20"] * np.sqrt(i["td"] / 252))
             if np.isfinite(i["atm_iv"]) and np.isfinite(i["rv20"]) else
             (i["atm_iv"] * np.sqrt(i["T"]) if np.isfinite(i["atm_iv"]) else i["rv20"] * np.sqrt(i["td"] / 252))
             for t, i in info.items()}
    o = o[o["ticker"].map(lambda t: np.isfinite(sig_w.get(t, np.nan)))].reset_index(drop=True)
    if o.empty:
        return {**out, "status": "no candidates with a volatility estimate"}
    kind = o["kind"].to_numpy()
    is_put, held_row, new_cc = kind == "put", kind == "call_held", kind == "call_new"
    S0, K, px = o["S"].to_numpy(), o["strike"].to_numpy(), o["px"].to_numpy()
    slip = cfg["stock_slippage_bps"] / 1e4
    stock_comm = max(1.0, cfg["stock_commission"] * 100)
    cost = cfg["opt_commission"] + np.where(new_cc, stock_comm + 100 * S0 * slip, 0.0)
    need = np.where(is_put, 100 * K, 100 * S0) + np.where(new_cc, 100 * S0 * slip + stock_comm, 0.0) + cfg["opt_commission"] - 100 * px
    gross = np.where(is_put, 100 * K, 100 * S0)
    Z = z[o["ticker"]].to_numpy()
    ST = S0 * np.exp(Z * o["ticker"].map(sig_w).to_numpy())
    P = np.where(is_put, 100 * (px - np.maximum(K - ST, 0)), 100 * (px + np.minimum(K, ST) - S0)) - cost
    grp = np.where(held_row, o["ticker"] + "#held", o["ticker"])
    names = sorted(set(grp))
    gid = {g: i for i, g in enumerate(names)}
    groups = pd.Series(grp).map(gid).to_numpy()
    cap_left = np.array([(holdings[g[:-5]] // 100) * 100 * info[g[:-5]]["S"] + 1e-6 if g.endswith("#held") else 1e12 for g in names])
    nmax = np.floor(cap_left[groups] / gross).clip(0, 200)
    keep = nmax > 0
    o, P, need, gross, groups, nmax = o[keep].reset_index(drop=True), P[:, keep], need[keep], gross[keep], groups[keep], nmax[keep]
    tv = o["tv"].to_numpy()
    args = (P, tv, need, gross, groups, cap_left, nmax, budget - 1.0)
    x = _two_stage(*args, "min_cvar", [("tv", target, np.inf)], alpha)
    status = "target reached"
    if x is None:
        best = _two_stage(*args, "max_tv", [], alpha)
        if best is None or best.sum() == 0:
            return {**out, "status": "nothing affordable reaches any premium"}
        tv_max = float(tv @ best)
        x = _two_stage(*args, "min_cvar", [("tv", 0.98 * tv_max, np.inf)], alpha)
        x = best if x is None else x
        status = f"target out of reach: the most this capital can collect is about ${tv_max:,.0f}"
    port = P @ x
    n_tail = max(1, int(np.ceil((1 - alpha) * len(port))))
    sel = o.loc[x > 0].copy()
    sel["contracts"] = x[x > 0]
    sel["premium"] = 100 * sel["px"] * sel["contracts"]
    sel["time_value"] = sel["tv"] * sel["contracts"]
    sel["capital_used"] = gross[x > 0] * sel["contracts"]          # put collateral, or the value of the shares
    sel["pct_from_price"] = sel["strike"] / sel["S"] - 1
    # what to do with each holding
    kept = sel[sel["kind"] == "call_held"].groupby("ticker")["contracts"].sum()
    hold_rows = []
    for t, sh in holdings.items():
        k_sh = min(sh, 100 * int(kept.get(t, 0)))
        price = info.get(t, {}).get("S", np.nan)
        hold_rows.append(dict(ticker=t, shares=sh, price=price, keep=k_sh, sell=sh - k_sh,
                              decision=("keep all and sell calls" if k_sh == sh else
                                        "sell all" if k_sh == 0 else f"keep {k_sh}, sell {sh - k_sh}")))
    # stress: the market falls 10% this week, each name by its usual sensitivity to QQQ
    beta = betas_vs(weekly_returns)
    drop = np.array([max(-0.95, -0.10 * beta.get(t, 1.0)) for t in o["ticker"]])
    s_k, k_k, px_k = o["S"].to_numpy(), o["strike"].to_numpy(), o["px"].to_numpy()
    ST_c = s_k * (1 + drop)
    crash = float((np.where(is_put[keep], 100 * (px_k - np.maximum(k_k - ST_c, 0)),
                            100 * (px_k + np.minimum(k_k, ST_c) - s_k)) - cost[keep]) @ x)
    return {**out, "status": status, "trades": sel, "holdings": pd.DataFrame(hold_rows),
            "time_value": float(tv @ x), "premium": float(sel["premium"].sum()),
            "capital_used": float(sel["capital_used"].sum()),
            "worst5_loss": float(-np.sort(port)[:n_tail].mean()), "chance_of_loss": float((port < 0).mean()),
            "expected": float(port.mean()), "worst_scenario": float(port.min()), "crash10": crash,
            "scenario_pnl": port, "n_scenarios": len(port)}
