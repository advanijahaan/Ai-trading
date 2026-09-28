# Ai-Duolingo-
Duolingo for free

## AI day-trading bot (Alpaca paper trading)

A bot that trades with **fake money** on Alpaca's paper account: **all major US stocks and ETFs, plus options**.
It bets on prices going **up and down**. It learns which
strategy works for each one and switches on its own.

**What it trades:**
- **Stocks:** every morning it builds a list of the **500 most-traded US stocks and ETFs** priced over $10, using
  Alpaca's own data, so the list is always current. During the day it also adds the most-traded stocks and the day's biggest gainers and losers, the "stocks in play"
  day traders focus on.
  Trading hours: Monday to Friday, 9:30 AM to 4:00 PM Eastern.
  With that many stocks, there are often more buy signals than free slots, so it takes the ones whose strategy
  has the best record first.
- **World markets:** Alpaca can only trade on US exchanges, so the bot reaches other countries through funds and
  companies listed in the US. It trades these during US market hours:
  - Countries and regions: Europe, Japan, China, India, Brazil, UK, Canada, Korea, Taiwan, emerging markets.
  - Big foreign companies: TSMC, ASML, Alibaba, Novo Nordisk, SAP, Shell, BP, Shopify, MercadoLibre and more.
  - Gold, silver, oil, natural gas and US Treasury bonds.
- **Options:** on SPY, QQQ, IWM, AAPL, MSFT, NVDA, AMD, TSLA, META, AMZN, GOOGL and AVGO, the bot buys **calls**
  when it expects a rise and **puts** when it expects a fall, instead of the shares. It uses contracts expiring
  in 1 to 3 weeks, near the current price. The most it can lose is what it paid. It sells if the option drops
  50% or doubles, and sells all options before the close.
- **Short selling:** on other stocks Alpaca allows, it can bet on a fall by shorting.
- **Crypto:** off. If it still holds crypto from before, it sells it. To turn crypto back on, add
  `BOT_CRYPTO=BTC/USD,ETH/USD` to `.env`.

**When it trades (New York time):**

| Session | Hours | What the bot does |
|---|---|---|
| Overnight | 8 PM – 4 AM, Sunday to Thursday nights | **Trades** the 100 most-traded stocks that allow overnight trading (free data). Buys only, limit orders, stops watched by the bot. Everything is sold by 3:50 AM. |
| Pre-market | 4 – 9:30 AM | Protects open positions. Also **trades** if you add paid data: `BOT_EXTENDED_FEED=sip` |
| Regular | 9:30 AM – 4 PM | Full trading |
| After-hours | 4 – 8 PM | Protects open positions. Also **trades** with `BOT_EXTENDED_FEED=sip` |
| Weekend | Friday 8 PM – Sunday 8 PM | Nothing: no market is open |

Outside regular hours, trading is thin and prices jump more, so the bot learns each stock's overnight behaviour
separately from its daytime behaviour. Turn it off with `BOT_EXTENDED=off`.
Pre-market and after-hours trading needs Alpaca's paid "Algo Trader Plus" data, because the free data has
almost no live prices then.

### How to see what it's making (phone or computer)

1. Go to **https://app.alpaca.markets** in any web browser (phone or computer) and log in.
2. At the top left, make sure your **Paper** account is selected. It shows a "Paper" label and starts with $100,000.
3. You'll see:
   - **Home:** your total money (equity) and today's gain or loss.
   - **Positions:** the stocks the bot owns right now and how much each is up or down.
   - **Orders:** every buy and sell the bot made. Click an order to open its details. The **Client Order ID**
     starts with the strategy the AI used, e.g. `breakout-AMD-1a2b3c4d`.

It sells everything a few minutes before the market closes, except trades from the `overnight_hold` strategy,
which are held until just after the next open.

### How to run it on your computer (one-time setup, ~5 minutes)

1. **Install Python** from https://www.python.org/downloads/ (on Windows, tick "Add Python to PATH").
2. **Download this project.** On GitHub click the green **Code** button → **Download ZIP**, then unzip it.
3. **Open a terminal in the project folder.**
   - Windows: open the folder, click the address bar, type `cmd`, press Enter.
   - Mac: right-click the folder → **New Terminal at Folder**.
4. **Install what it needs:** `pip install -r requirements.txt`
5. **Add your keys.** Make a copy of `.env.example` and name it `.env`. Open it in Notepad or TextEdit and paste your
   paper **API Key** and **Secret Key**. Get them at https://app.alpaca.markets (Paper account) → **API Keys** on the home page.
   Never share `.env` or upload it anywhere.
6. **Test it** (places no orders): `python -m trading_bot.bot --dry-run --once`
7. **Start it:** `python -m trading_bot.bot`
   Leave the window open for as long as you want it to trade. Close it or press **Ctrl+C** to stop.
   Stock trades keep their automatic stop-loss and take-profit orders after you stop. Crypto and options don't
   support those orders, so the bot watches their stop-losses itself. If you stop the bot while it owns crypto
   or options, sell them in the Alpaca app or start the bot again.

To see what the AI has learned and which strategy each stock and coin is using: `python -m trading_bot.bot --report`

### Keep it running 24/7 (free, no credit card)

The bot has to run on a computer that stays on. Use one you already own: a desktop, or a laptop left plugged in.
After a one-time setup it starts when you log in, restarts within 5 seconds if it crashes, and keeps the computer
from sleeping. On a Mac, do steps 1–5 of "How to run it on your computer" above first.

**Windows (easiest):** download
[this ZIP](https://github.com/advanijahaan/Ai-trading/archive/refs/heads/claude/jolly-mendel-hoavwy.zip),
right-click it → **Extract All**, open the extracted folder and **double-click `SETUP.bat`**. It installs Python if
needed, asks for your keys, checks they work and starts the bot. You can skip the steps above.

**Mac:** open Terminal in the project folder and run `bash deploy/mac/install-autostart.sh`.

Then keep the computer **plugged in and switched on**. Closing a laptop lid usually still makes it sleep, so leave
the lid open. Logging out stops the bot until you log back in. Setting Windows or macOS to log in automatically
makes it restart after a power cut too.

**Getting updates:** double-click **`UPDATE.bat`** in the bot folder. It downloads the latest version, keeps your
keys and everything the bot has learned, and restarts it.

Check on it from your phone at any time in the Alpaca app or website. The log is in `bot.log` in the project folder.

⚠️ **Run only one copy of the bot at a time.** Two copies on the same Alpaca account would both place orders.

<details><summary>Other ways to host it (a Linux server, Docker)</summary>

- Any always-on Ubuntu/Debian server: `bash deploy/install.sh` installs it as a service that restarts itself.
- Docker hosts: use the `Dockerfile` with `--restart always`, as shown at the top of that file.
</details>

### How the AI works

The bot has 7 strategies, each in an "up" version and a "down" (`_short`) version. That makes 14 in total.
The last four come from published research on US stocks:

| Strategy | Buys when… | Sells when… |
|---|---|---|
| `trend` | the short-term average price crosses above the longer-term one | it crosses back below |
| `mean_reversion` | a stock that dropped too far starts bouncing back (RSI climbs back above 30) | price gets back to its average |
| `breakout` | price breaks above its recent high on heavy volume | price falls below its recent low |
| `orb` (opening range breakout) | on a stock trading far more than usual at the open (a "stock in play"), price breaks above the first 5 minutes' high after an up start ([Zarattini, Barbon & Aziz](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4729284)) | stop at the first 5 minutes' low, otherwise at the close |
| `vwap_trend` | price crosses above VWAP, the day's volume-weighted average price that professional desks trade against ([Zarattini & Aziz](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4631351)) | price falls back below VWAP |
| `overnight_hold` | at 3:50 PM, to hold overnight. Historically, most US stock gains came between the close and the next open ([Cliff, Cooper & Gulen](https://www.ssrn.com/abstract=1004081)). Overnight holds never count as day trades. Buy-only, at most 3 at a time and 10% of the account each. | 9:35 AM the next morning |
| `intraday_momentum` | at 3:30 PM, if the stock rose in the first half hour (previous close to 10:00) ([Gao, Han, Li & Zhou, 2018](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2440866)) | at the close |

The `_short` versions (`trend_short`, `orb_short` and so on) are the mirror images. They bet on
a fall by shorting the stock or buying a put. Crypto only uses the "up" versions because Alpaca doesn't allow
shorting crypto.

For each stock (every 5 minutes) and each coin (every hour), the bot:
1. **Replays** all 3 strategies on recent prices to see which would have made money after fees. It uses the last
   ~2 weeks for stocks and the last ~30 days for crypto.
2. **Remembers** how its real trades turned out, saved in `bot_state.json`. Real trades count double.
3. **Switches** each stock or coin to the strategy that's working best. If none are making money, it **sits that one out**.
   Stocks learn from stocks and crypto learns from crypto.
   The log shows lines like `AMD: AI switched strategy trend -> breakout`.

It doesn't trust a strategy because of a few lucky trades. A strategy needs a steady record before the bot uses it.

**It adjusts itself to the size of your account.** It checks the balance every minute, so this works the same
whether you start with $10 or $100,000:

| Balance | What it does |
|---|---|
| under $100 | up to 2 positions, each up to half the account. Buys **slices of shares** ($1 minimum) when a whole share costs too much. |
| $100 – $2,000 | up to 4 positions, each up to a quarter of the account |
| $2,000 and up | normal mode: up to 8 positions, plus short selling and options |
| under $25,000 | stays within the US **day-trading limit** (3 same-day round trips in 5 business days), so a sell is never blocked. Crypto isn't affected. Turn this off with `BOT_PDT=off` if the rule doesn't apply to your account. |

**It reads the news.** Every few minutes the bot downloads new articles from Alpaca's free news feed (Benzinga).
Research finds that price moves caused by news tend to keep going, while moves without news tend to reverse
([Alpha Architect summary](https://alphaarchitect.com/what-drives-momentum-and-reversal/)). So:
- the "ride the move" strategies (`trend`, `breakout`, `orb`, `vwap_trend`, `intraday_momentum`) only buy a stock
  that has had news in the last 24 hours;
- `mean_reversion` only buys a stock that has **not** been in the news, so it doesn't catch a falling knife.

The AI's replays apply the same rule, using the news history. Turn it off with `BOT_NEWS=off`.

**How trades end:**
- Every trade starts with a stop-loss.
- Once a trade is up by the amount it risked, the stop follows the price up (1x the risk below the best price so
  far), so a winner can't turn into a loser.
- Trades are closed after **2 hours**, except `orb`, `intraday_momentum` and `overnight_hold`, which have their own
  timing.
- Outside market hours stop orders don't work. So if an overnight hold drops below its stop in pre-market,
  after-hours or Alpaca's overnight session, the bot sells it with an extended-hours limit order.
- The AI's replays use exactly these rules, so what it learns matches how trades really run.

**Safety limits:** every buy comes with a stop-loss and a take-profit. Each trade risks about 1% of the account.
Options risk about 1% of the account each, counting a 50% drop as the full loss.
No single stock or coin gets more than 20% of the money, and the bot holds at most 8 at once. If the account is down 3% on
the day, it sells everything and stops until the next day. It won't connect to a real-money account unless you set
`ALPACA_ALLOW_LIVE=1`.

Settings live in `trading_bot/config.py`. For example, lowering `min_score` makes the bot trade more often, with less proof
that a strategy works. You can also add these lines to `.env`:
- `BOT_SYMBOLS=SPY,AAPL`: stocks to always watch
- `BOT_SCAN_STOCKS=off`: stop adding the most-traded stocks
- `BOT_WORLD=off`: no world markets
- `BOT_OPTIONS=off`: trade shares instead of options
- `BOT_SHORTS=off`: only bet on prices going up
- `BOT_UNIVERSE_SIZE=200`: watch fewer stocks (useful on a slow computer)
- `BOT_CRYPTO=BTC/USD,ETH/USD`: turn crypto on with these coins

This is a learning project with fake money. How it does on paper doesn't predict real results.

Developers: run the tests with `python -m unittest discover -s tests`.
