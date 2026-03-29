# LIVE TRADING SETUP GUIDE

## Step 1 — Binance Futures API
1. Go to binance.com → Profile → API Management
2. Create new API key → Label it "CryptoBot"
3. Enable: "Enable Futures" ✅
4. Disable: "Enable Withdrawals" ❌ (never enable this)
5. Whitelist your IP address for extra safety

## Step 2 — Add API Keys
1. Copy live/.env.example → live/.env
2. Paste your API key and secret into live/.env
3. Run this in your terminal to load them:

   Windows (PowerShell):
   $env:BINANCE_API_KEY="your_key"
   $env:BINANCE_SECRET="your_secret"

   Or install python-dotenv and load automatically:
   pip install python-dotenv

## Step 3 — Test on Testnet First
In live/trader.py confirm:
   TESTNET = True   ← keep this True for paper trading
   CAPITAL = 100    ← USDT per trade

Binance Testnet: https://testnet.binancefuture.com
Create testnet API keys there for testing.

## Step 4 — Train Model
   python run.py

## Step 5 — Run Bot
   python live/trader.py

## Step 6 — Monitor
- Logs saved to live/trade_log.csv
- Check every day for first 2 weeks
- Only switch TESTNET=False after 20+ paper trades match backtest expectations

## Risk Rules (follow these)
- Start with CAPITAL = 50-100 USDT per trade
- Never risk more than 5% of total account per trade
- Stop bot if drawdown exceeds 10%
- Review trade_log.csv weekly

## File Structure
Algo/
├── live/
│   ├── trader.py        ← main live bot
│   ├── .env             ← your API keys (never share)
│   ├── .env.example     ← template
│   ├── trade_log.csv    ← auto-generated trade history
│   └── SETUP.md         ← this file
