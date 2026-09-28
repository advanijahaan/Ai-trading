"""Configuration loaded from environment variables.

API keys are never hard-coded: set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY
(from the Alpaca paper dashboard) in your shell or a .env file.
"""
import os
from dataclasses import dataclass, field

PAPER_BASE_URL = "https://paper-api.alpaca.markets/v2"
DATA_BASE_URL = "https://data.alpaca.markets/v2"


def _load_dotenv(path=".env"):
    """Minimal .env loader so no extra dependency is needed."""
    if not os.path.exists(path):
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _env_list(name, default):
    raw = os.getenv(name)
    if not raw:
        return default
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


@dataclass
class Config:
    api_key: str
    api_secret: str
    base_url: str = PAPER_BASE_URL
    data_url: str = DATA_BASE_URL
    data_feed: str = "iex"  # free accounts only get the IEX feed

    # Stocks: the day's universe_size most-traded US stocks/ETFs, plus these, plus intraday movers (scan)
    universe_size: int = 500
    replay_minutes: int = 45         # re-learn each stock this often (staggered)
    max_replays_per_loop: int = 60   # bounds each loop's learning time
    symbols: list = field(default_factory=lambda: ["SPY", "QQQ"])
    # World markets through US-listed funds and foreign companies (Alpaca can't reach foreign exchanges)
    world_symbols: list = field(default_factory=lambda: [
        # country / region ETFs
        "EFA", "EEM", "VGK", "EWJ", "FXI", "KWEB", "MCHI", "INDA", "EWZ", "EWU", "EWC", "EWY", "EWT",
        # big foreign companies listed in the US
        "TSM", "ASML", "BABA", "PDD", "NVO", "SAP", "SHEL", "BP", "SHOP", "MELI", "NU",
        # commodities and bonds
        "GLD", "SLV", "USO", "UNG", "TLT"])
    scan_stocks: bool = True
    scan_top: int = 50               # how many most-active stocks to look at
    max_scanned: int = 40            # how many of them to trade after filtering
    min_price: float = 10.0          # skip penny stocks
    min_dollar_volume: float = 20e6  # skip thinly traded stocks
    rescan_minutes: int = 30
    movers_top: int = 20             # also watch the day's top gainers and losers
    # Crypto trades 24/7. Off by default; turn on with e.g. BOT_CRYPTO=BTC/USD,ETH/USD,SOL/USD
    crypto_symbols: list = field(default_factory=list)
    crypto_timeframe: str = "1Hour"  # 5-minute crypto moves are too small to beat the fees
    crypto_learn_days: int = 30
    crypto_cost_pct: float = 0.005   # Alpaca crypto fees are ~0.25% per side
    min_order_dollars: float = 1.0   # Alpaca's minimum for fractional shares
    # Short selling: bet on falling prices (stocks Alpaca can borrow; never crypto)
    allow_shorts: bool = True
    # Options: on these liquid names, signals buy calls (bullish) or puts (bearish) instead of shares
    options_underlyings: list = field(default_factory=lambda: [
        "SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "META", "AMZN", "GOOGL", "AVGO"])
    option_min_days: int = 7          # skip contracts expiring sooner (time decay is brutal)
    option_max_days: int = 21
    option_target_delta: float = 0.5  # at-the-money-ish
    option_max_spread: float = 0.15   # skip if (ask - bid) / mid is wider than this
    option_stop_pct: float = 0.5      # sell if the option loses half its value
    option_take_profit_pct: float = 1.0  # sell if it doubles
    timeframe: str = "5Min"
    poll_seconds: int = 60

    # Strategy
    fast_ema: int = 9
    slow_ema: int = 21
    rsi_period: int = 14
    rsi_max_entry: float = 70.0
    atr_period: int = 14
    stop_atr_mult: float = 1.5
    take_profit_atr_mult: float = 3.0
    # mean reversion
    mr_sma: int = 20
    mr_oversold: float = 30.0
    mr_exit_rsi: float = 55.0
    mr_target_atr_mult: float = 2.0
    # breakout
    breakout_lookback: int = 20
    breakout_exit_lookback: int = 10
    breakout_volume_mult: float = 1.5
    breakout_stop_atr_mult: float = 2.0
    breakout_target_atr_mult: float = 4.0
    # opening range breakout (stocks in play)
    orb_min_rel_volume: float = 2.0      # first 5 minutes' volume vs. the usual first 5 minutes
    orb_entry_window_minutes: int = 90   # only take breakouts before 11:00
    orb_target_r: float = 10.0           # effectively "hold until the close"
    # VWAP trend
    vwap_skip_minutes: int = 15          # VWAP is noisy right after the open
    vwap_target_atr_mult: float = 4.0
    # intraday momentum
    im_entry_minute: int = 15 * 60 + 25  # the bar that closes at 15:30 New York time
    im_min_move: float = 0.0
    # overnight hold
    overnight_entry_minute: int = 15 * 60 + 45  # the bar that closes at 15:50 New York time
    overnight_exit_after_minutes: int = 5       # sell once the first 5 minutes after the open have closed
    overnight_stop_pct: float = 0.02            # also sells in pre-market/after-hours if it falls this far
    overnight_max_positions: int = 3            # prices can gap past stops overnight: keep it small
    overnight_max_position_pct: float = 0.10    # each at most 10% of the account (so at most 30% overnight)

    # News (Alpaca's free Benzinga feed): momentum strategies need news in the last day, the bounce-back
    # strategy needs none (news-driven moves continue, moves without news reverse)
    news_filter: bool = True
    news_window_hours: int = 24

    # Extended hours. The overnight session (8 PM - 4 AM) trades on free data; pre-market (4 - 9:30 AM)
    # and after-hours (4 - 8 PM) need live extended-hours prices, which on Alpaca means the paid SIP feed:
    # set BOT_EXTENDED_FEED=sip after subscribing to turn those sessions on.
    extended_trading: bool = True
    extended_feed: str = ""
    ext_universe_size: int = 100       # the most-traded stocks that can trade outside regular hours
    ext_cost_pct: float = 0.001        # wider spreads outside regular hours
    ext_flatten_minutes: int = 10      # sell everything this long before the session ends
    ext_no_entry_minutes: int = 30     # no new trades this close to the session end
    ext_limit_pct: float = 0.001       # limit orders reach this far past the quote so they fill

    # Exits for every trade: trail the stop once ahead, and don't sit in a trade all day
    trail_start_r: float = 1.0       # once up by 1x the amount risked...
    trail_r: float = 1.0             # ...keep the stop 1x the risk below the best price since entry
    max_hold_minutes: int = 120      # sell after 2 hours (not for orb / intraday_momentum / overnight_hold)

    # Learning: which strategy each symbol uses
    learn_days: int = 10          # calendar days of history to replay each strategy on
    prior_strength: float = 10.0  # trades' worth of "assume average" before trusting a symbol's own record
    live_weight: float = 2.0      # a real trade counts as much as this many replayed ones
    min_score: float = 0.10       # sit a symbol out unless the best strategy expects > this many R per trade
    cost_pct: float = 0.0005      # assumed round-trip slippage when replaying
    state_file: str = "bot_state.json"

    # Risk
    risk_per_trade: float = 0.01      # risk 1% of equity per trade
    max_position_pct: float = 0.20    # never put more than 20% of equity in one name
    max_open_positions: int = 8
    daily_loss_limit: float = 0.03    # stop trading after -3% on the day
    # Account-size modes (see risk.account_limits); the bot switches automatically as the balance changes
    margin_min_equity: float = 2000.0   # Alpaca needs $2,000 for shorting (and we require it for options)
    pdt_protection: bool = True         # respect the US pattern-day-trader limit below pdt_equity
    pdt_equity: float = 25000.0
    pdt_max_day_trades: int = 3         # per rolling 5 business days
    no_new_entries_minutes: int = 25  # before close (15:35), so the 15:30 momentum entry fits
    flatten_minutes: int = 10         # close everything this long before close

    @classmethod
    def from_env(cls):
        _load_dotenv()
        key = os.getenv("ALPACA_API_KEY_ID") or os.getenv("APCA_API_KEY_ID")
        secret = os.getenv("ALPACA_API_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
        if not key or not secret:
            raise SystemExit(
                "Missing API keys. Set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY "
                "(see .env.example)."
            )
        base_url = os.getenv("ALPACA_BASE_URL", PAPER_BASE_URL).rstrip("/")
        if not base_url.endswith("/v2"):
            base_url += "/v2"
        if "paper-api" not in base_url and os.getenv("ALPACA_ALLOW_LIVE") != "1":
            raise SystemExit(
                f"Refusing to trade against {base_url}: this bot is for paper trading. "
                "Set ALPACA_ALLOW_LIVE=1 only if you really mean to use real money."
            )
        cfg = cls(api_key=key, api_secret=secret, base_url=base_url)
        cfg.symbols = _env_list("BOT_SYMBOLS", cfg.symbols)
        if os.getenv("BOT_WORLD", "").lower() in ("0", "off", "false", "no"):
            cfg.world_symbols = []
        if os.getenv("BOT_SCAN_STOCKS", "").lower() in ("0", "off", "false", "no"):
            cfg.scan_stocks = False
        if os.getenv("BOT_OPTIONS", "").lower() in ("0", "off", "false", "no"):
            cfg.options_underlyings = []
        if os.getenv("BOT_NEWS", "").lower() in ("0", "off", "false", "no"):
            cfg.news_filter = False
        if os.getenv("BOT_EXTENDED", "").lower() in ("0", "off", "false", "no"):
            cfg.extended_trading = False
        cfg.extended_feed = os.getenv("BOT_EXTENDED_FEED", cfg.extended_feed).lower()
        if os.getenv("BOT_PDT", "").lower() in ("0", "off", "false", "no"):
            cfg.pdt_protection = False
        if os.getenv("BOT_SHORTS", "").lower() in ("0", "off", "false", "no"):
            cfg.allow_shorts = False
        crypto = os.getenv("BOT_CRYPTO", "")
        if crypto and crypto.lower() not in ("0", "off", "false", "no"):
            cfg.crypto_symbols = _env_list("BOT_CRYPTO", cfg.crypto_symbols)
        if os.getenv("BOT_UNIVERSE_SIZE"):
            cfg.universe_size = int(os.getenv("BOT_UNIVERSE_SIZE"))
        cfg.data_feed = os.getenv("ALPACA_DATA_FEED", cfg.data_feed)
        cfg.state_file = os.getenv("BOT_STATE_FILE", cfg.state_file)
        return cfg
