# Weekly options picker

Enter your cash and a weekly target, as a % of the account or in $. The app finds the **lowest-risk set of weekly puts and covered calls** that collects that target, using this week's option chains.

**What it can trade:**
- puts on Nasdaq-100 stocks,
- puts on QQQ, SMH and XLK,
- covered calls (buy 100 shares and sell a call).

**Rules:** no margin, earnings weeks skipped, and every trade fits inside your cash.

**If you already hold shares** (from an assigned put, or a covered call that wasn't called away), list them. For each one the app either keeps the shares and sells calls on them at any strike, including below what you paid, or sells them. Whichever is lower risk for reaching the target wins.

"Lowest risk" means the **smallest average loss in the worst 5% of weeks**. That's estimated from every week of the last 3 years of real price moves, with each name's move rescaled to its current volatility. It's the same method as the backtest notebooks (Phases 3–8).

## What you get

- **The trade list:** action, ticker, strike, expiration, contracts, limit price, premium, time value, capital used, delta, and how far the strike is from the price. Downloadable as CSV.
- **For each holding:** keep and sell calls, or sell.
- **Risk:**
  - the average loss in the worst 5% of weeks,
  - the chance of a losing week,
  - the loss if the market drops 10% this week (each name moving by its usual sensitivity to QQQ),
  - a chart of the week's range of outcomes.
- **The backtest:** how the closest target did, if `target_table.csv` is present (see below).

If the target can't be reached with your cash, it says the most you could collect and shows the lowest-risk way to get close.

## Run it locally

```bash
pip install -r requirements.txt          # Python 3.12+ (the thetadata library requires it)
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then paste your ThetaData API key into it
streamlit run app.py
```

Without a key, the app starts in **demo mode** with made-up prices, so you can try the screens.

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repo. `.gitignore` keeps `secrets.toml` out.
2. At share.streamlit.io, choose **New app**, pick the repo and `app.py`, and under **Advanced settings** choose **Python 3.12**.
3. Paste `THETADATA_API_KEY = "…"` into **Secrets**.

## Data sources

| What | From | Notes |
|---|---|---|
| Option chains | ThetaData (your Value plan) | Live snapshot during market hours. When the market is closed, the last quotes of the previous session. Fetched 2 names at a time, so ~100 names take a minute or two; cached for 10 minutes. |
| Nasdaq-100 list | Wikipedia | Or put a `universe.csv` (one ticker per line) next to `app.py` to control the list yourself. |
| Prices, realized vol, scenario history | Yahoo Finance | 4 years of daily closes. |
| Earnings dates | Yahoo Finance | Names it can't check are kept in and listed under "Data and names left out". Double-check those. |
| 3-month T-bill rate | FRED | Used for discounting. |

## The backtest table

Copy `target_table.csv` from Drive (`options_bot/data/derived/backtests_phase8/`, written by the Phase 8 notebook) into this folder. The app will then show how the closest weekly target held up in 2020–2026 and in 2026 so far.

## Things to know

- **The worst-5% figure is a model estimate.** In the backtests, real bad weeks ran **1.2–1.6×** the model's estimate. Treat it as a floor, not a ceiling.
- **Premium is not profit.** The backtests (Phases 3–8) showed most of the premium from high targets going back out in losing weeks. Check the backtest table for your target before trusting it.
- **Fills default to the bid.** The "assume you sell at" setting can move that toward mid.
- **"Time value" is what gets matched to the target:** the premium minus any in-the-money amount on a call below the price. That's what you keep if nothing moves.
- **Place the orders yourself** after checking them. This is a planning tool, not financial advice, and it doesn't place trades.

## Tests

`tests/` holds headless checks:
- `test_core.py`: the optimizer on the demo market.
- `test_data.py`: the ThetaData layer against a fake client.
- `test_app.py`: the whole app in demo mode, using Streamlit's AppTest.

Run each with `python tests/<name>.py`.
