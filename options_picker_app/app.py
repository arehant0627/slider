"""Weekly options picker: enter your cash and a weekly target, get the lowest-risk set of puts and covered calls
that collects it. Run with:  streamlit run app.py"""
import datetime as dt
import inspect

import numpy as np
import pandas as pd
import streamlit as st

import core
import data

st.set_page_config(page_title="Weekly options picker", layout="wide")

_ARROW_OFF = {"delta_arrow": "off"} if "delta_arrow" in inspect.signature(st.metric).parameters else {}


def usd(v):
    return f"−${abs(v):,.0f}" if v < 0 else f"${v:,.0f}"


def metric(col, label, value, sub=None, help=None):
    """A metric with an optional grey note underneath (no up/down arrow)."""
    kw = {"help": help}
    if sub is not None:
        kw.update(delta=sub, delta_color="off", **_ARROW_OFF)
    col.metric(label, value, **kw)
st.title("Weekly options picker")
st.caption("Finds the lowest-risk set of weekly puts and covered calls that collects your target, within your cash. "
           "Risk = the average loss in the worst 5% of weeks, from 3 years of real weekly moves scaled to today's "
           "volatility. A planning tool, not financial advice.")

# ---------------------------------------------------------------- data source
try:
    API_KEY = st.secrets.get("THETADATA_API_KEY", "")
except Exception:                                               # no secrets file at all
    API_KEY = ""
demo = st.sidebar.toggle("Demo data (no ThetaData key needed)", value=not API_KEY,
                         help="Made-up prices, for trying the app. Turn off to use live ThetaData quotes.")
if not demo and not API_KEY:
    st.sidebar.error("Add THETADATA_API_KEY to the app's secrets to use live data.")
    st.stop()


@st.cache_resource(show_spinner=False)
def client_for(key):
    return data.make_client(key)


@st.cache_data(ttl=3600, show_spinner=False)
def expirations_live(key):
    return data.upcoming_expirations(client_for(key))


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def universe_live():
    return data.nasdaq100()


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def prices_live(symbols):
    return data.price_history(list(symbols))


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def earnings_live(symbols, start, end):
    return data.earnings_between(list(symbols), start, end)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def rate_live():
    return data.tbill_rate()


@st.cache_resource(show_spinner=False)
def demo_world():
    return data.demo_market()


if demo:
    W = demo_world()
    exps = W["expirations"]
else:
    exps = expirations_live(API_KEY)
    if not exps:
        st.error("ThetaData returned no upcoming expirations for QQQ. Check the API key and your plan.")
        st.stop()

# ---------------------------------------------------------------- inputs
sb = st.sidebar
unit = sb.radio("Weekly target in", ["% of account", "$"], horizontal=True)
with sb.form("inputs"):
    cash = st.number_input("Cash to use ($)", min_value=1_000, max_value=50_000_000, value=100_000, step=5_000)
    if unit == "% of account":
        target_in = st.number_input("Weekly target (% of account)", min_value=0.05, max_value=10.0, value=0.5, step=0.05,
                                    format="%.2f")
    else:
        target_in = st.number_input("Weekly target ($)", min_value=10, max_value=5_000_000, value=500, step=50)
    exp = st.selectbox("Expiration", exps, index=data.default_expiration_index(exps),
                       format_func=lambda d: d.strftime("%a %b %d, %Y"),
                       help="Defaults to this Friday Monday-Thursday, and to next Friday from Friday noon on.")
    st.markdown("**What it can use**")
    use_stock_puts = st.checkbox("Puts on Nasdaq-100 stocks", True)
    use_etf_puts = st.checkbox("Puts on QQQ, SMH, XLK", True)
    use_cc = st.checkbox("Covered calls (buy 100 shares, sell a call)", True)
    fill_lbl = st.select_slider("Assume you sell at", ["bid", "25% into the spread", "mid"], value="bid")
    st.markdown("**Shares you already hold** (optional)")
    st.caption("The bot either keeps them and sells calls on them, or sells them, whichever is lower risk.")
    hold_in = st.data_editor(pd.DataFrame({"ticker": pd.Series([""], dtype=str), "shares": pd.Series([0], dtype=int)}),
                             num_rows="dynamic", width="stretch", key="holdings")
    go = st.form_submit_button("Find trades", type="primary", width="stretch")

holdings = {}
for r in hold_in.itertuples():
    t = str(r.ticker or "").strip().upper()
    if t and r.shares and int(r.shares) > 0:
        holdings[t] = holdings.get(t, 0) + int(r.shares)
fill = {"bid": 0.0, "25% into the spread": 0.25, "mid": 0.5}[fill_lbl]
use = dict(stock_puts=use_stock_puts, etf_puts=use_etf_puts, covered_calls=use_cc)

# ---------------------------------------------------------------- run
if go:
    now = data.now_ny()
    exp_ts = pd.Timestamp(exp)
    status = st.status("Finding trades…", expanded=True)
    if demo:
        names = [n for n in W["names"] if (n in data.ETFS or use_stock_puts or use_cc or n in holdings)]
        close, rate, earn, unknown = W["close"], W["rate"], W["earnings"], set()
    else:
        status.write("Nasdaq-100 list and prices…")
        stocks = universe_live() if (use_stock_puts or use_cc) else []
        names = sorted(set(stocks) | set(data.ETFS) | set(holdings))
        close = prices_live(tuple(sorted(set(names) | {"QQQ"})))
        rate = rate_live()
        status.write("Earnings dates…")
        earn, unknown = earnings_live(tuple(names), now.to_pydatetime(),
                                      (pd.Timestamp(exp).tz_localize(data.NY) + pd.Timedelta(hours=16)).to_pydatetime())
    weekly, rv20, lwz = data.features_from_prices(close, now)
    last_px = close.iloc[-1].to_dict()
    held_val_guess = sum(sh * last_px.get(t, 0) for t, sh in holdings.items())
    budget_guess = cash + held_val_guess
    too_big = {t for t in names if np.isfinite(last_px.get(t, np.nan)) and 100 * last_px[t] > budget_guess and t not in holdings}
    fetch = [t for t in names if t not in too_big and (t not in earn or t in holdings)]
    status.write(f"Option chains for {len(fetch)} names (2 at a time)…")
    if demo:
        chains = {t: W["chain"](t, exp) for t in fetch}
        src = {t: ("demo", now) for t in fetch}
    else:
        bar = status.progress(0.0)
        raw = data.fetch_chains(client_for(API_KEY), fetch, exp,
                                progress=lambda f, s: bar.progress(f, text=f"{s} ({f:.0%})"))
        chains = {t: r[0] for t, r in raw.items()}
        src = {t: (r[1], r[2]) for t, r in raw.items()}
    status.write("Pricing and optimizing…")
    cand, info, skipped = core.build_candidates(chains, exp_ts, now, rate, rv20, lwz, set(earn), holdings, use,
                                                dict(fill=fill))
    for t in too_big:
        skipped[t] = "one contract needs more than your cash"
    for t in set(earn) - set(holdings):
        skipped.setdefault(t, "earnings before expiration")
    held_value = sum(sh * info[t]["S"] for t, sh in holdings.items() if t in info)
    account = cash + held_value
    target = (target_in / 100) * account if unit == "% of account" else float(target_in)
    res = core.optimize(cand, info, holdings, cash, target, weekly, {})
    status.update(label="Done", state="complete", expanded=False)
    st.session_state["result"] = dict(res=res, target=target, account=account, cash=cash, unit=unit, target_in=target_in,
                                      exp=exp, skipped=skipped, src=src, unknown=unknown, n_cand=len(cand), demo=demo,
                                      holdings=holdings)

# ---------------------------------------------------------------- results
R = st.session_state.get("result")
if R is None:
    st.info("Set your cash and weekly target in the sidebar, then **Find trades**. With live data, fetching the chains "
            "for ~100 names takes a minute or two.")
    st.stop()

res = R["res"]
if R["demo"]:
    st.warning("Demo data: made-up prices, for trying the app.")
tgt_lbl = f"{R['target_in']:.2f}% of ${R['account']:,.0f}" if R["unit"] == "% of account" else "set in $"
if "trades" not in res:
    st.error(f"No plan: {res['status']}.")
    st.stop()
if res["status"] == "target reached":
    st.success(f"Target reached with the lowest-risk combination found, expiring {R['exp']:%a %b %d}.")
else:
    st.warning(f"{res['status'][0].upper() + res['status'][1:]}. Showing the lowest-risk way to get close to it.")

c = st.columns(4)
metric(c[0], "Weekly target", usd(R["target"]), tgt_lbl)
metric(c[1], "Time value collected", usd(res["time_value"]), f"premium {usd(res['premium'])}",
       help="Premium minus any in-the-money amount: what you keep if nothing moves. This is what's matched to the target.")
metric(c[2], "Capital used", usd(res["capital_used"]), f"of {usd(res['budget'])}")
metric(c[3], "Cash left idle", usd(max(res["budget"] - res["capital_used"], 0)), "earns interest at your broker")
c = st.columns(4)
metric(c[0], "Average loss, worst 5% of weeks", usd(res["worst5_loss"]), f"{res['worst5_loss'] / R['account']:.1%} of account",
       help="A model estimate. In the backtests, real bad weeks ran 1.2-1.6x this.")
metric(c[1], "Chance of a losing week", f"{res['chance_of_loss']:.0%}", "model estimate")
metric(c[2], "If the market drops 10% this week", usd(res["crash10"]), f"{res['crash10'] / R['account']:.1%} of account",
       help="Each name falls by its usual sensitivity to QQQ times 10%.")
metric(c[3], "Model's average outcome", usd(res["expected"]), "close to zero by design",
       help="Average over the scenarios. Scenarios are scaled to current volatility, so this sits near zero; "
            "the backtest table is the better guide to what's actually kept.")

t = res["trades"].copy()
act = {"put": "Sell puts", "call_new": "Buy shares + sell calls", "call_held": "Sell calls on your shares"}
t["action"] = t["kind"].map(act)
t["expiration"] = pd.to_datetime(t["expiration"]).dt.strftime("%Y-%m-%d")
show = t[["action", "ticker", "strike", "expiration", "contracts", "px", "premium", "time_value", "capital_used", "delta",
          "pct_from_price", "S"]].rename(columns={
    "px": "limit price", "time_value": "time value", "capital_used": "capital", "pct_from_price": "strike vs price",
    "S": "price now"}).sort_values(["action", "capital"], ascending=[True, False])
st.subheader("Trades")
st.dataframe(show.style.format({"strike": "{:,.2f}", "limit price": "{:,.2f}", "premium": "${:,.0f}",
                                "time value": "${:,.0f}", "capital": "${:,.0f}", "delta": "{:.2f}",
                                "strike vs price": "{:+.1%}", "price now": "{:,.2f}"}),
             width="stretch", hide_index=True)
st.download_button("Download trades (CSV)", show.to_csv(index=False), file_name=f"trades_{R['exp']:%Y%m%d}.csv")

if R["holdings"]:
    st.subheader("Your shares")
    h = res["holdings"]
    st.dataframe(h.style.format({"price": "{:,.2f}"}, na_rep="no quote"), width="stretch", hide_index=True)
    st.caption("Shares with no call on them are sold: holding them uncovered adds risk without adding premium.")

st.subheader("Range of outcomes for this week (model scenarios)")
pnl = res["scenario_pnl"]
bins = np.histogram_bin_edges(pnl, bins=30)
hist = pd.DataFrame({"profit or loss ($)": np.round((bins[:-1] + bins[1:]) / 2), "scenarios": np.histogram(pnl, bins=bins)[0]})
st.bar_chart(hist, x="profit or loss ($)", y="scenarios")
st.caption(f"{res['n_scenarios']} scenarios: every week of the last 3 years, with each name's move rescaled to its current volatility.")

bt = data.backtest_table()
st.subheader("How this target did in the backtest")
if bt is None:
    st.info("Put `target_table.csv` (saved by the Phase 8 notebook in `data/derived/backtests_phase8/`) next to app.py "
            "to see how each weekly target held up in 2020-2026.")
else:
    pct_target = 100 * R["target"] / R["account"]
    near = bt.iloc[(bt["target"] * 100 - pct_target).abs().argsort()[:2]]
    cols = [c for c in ["run", "2024-26 avg/week", "PASS", "2026 so far", "weeks profit ≥ target", "worst week", "max drawdown"] if c in near]
    st.caption(f"Your target is {pct_target:.2f}% a week; the closest backtested targets:")
    st.dataframe(near[cols], width="stretch", hide_index=True)

with st.expander(f"Data and names left out ({len(R['skipped'])})"):
    srcs = pd.Series({k: v[0] for k, v in R["src"].items()}).value_counts()
    st.write("Quote sources: " + ", ".join(f"{k}: {v}" for k, v in srcs.items()))
    if R["unknown"]:
        st.write("Couldn't check earnings dates for: " + ", ".join(sorted(R["unknown"])) + " (they were kept in).")
    if R["skipped"]:
        st.dataframe(pd.DataFrame(sorted(R["skipped"].items()), columns=["ticker", "why left out"]), hide_index=True)
