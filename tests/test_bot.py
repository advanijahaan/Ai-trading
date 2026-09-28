import os
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from unittest import mock

from trading_bot.bot import TradingBot, completed_bars
from trading_bot.config import Config
from trading_bot.extended import extended_session, in_session
from trading_bot.risk import account_limits, daily_loss_hit, option_contracts, pick_option, position_size
import tempfile

from trading_bot.learner import Learner, backtest, trail_stop
from trading_bot.strategy import (STRATEGIES, Signal, annotate_sessions, atr, breakout, ema, intraday_momentum,
                                   mean_reversion, mirror, orb, overnight_hold, rsi, trend, vwap_trend)

T0 = datetime(2026, 9, 24, 14, 0, tzinfo=timezone.utc)


def make_bars(closes):
    return [
        {"t": (T0 + timedelta(minutes=5 * i)).isoformat().replace("+00:00", "Z"),
         "o": c, "h": c + 0.5, "l": c - 0.5, "c": c, "v": 1000}
        for i, c in enumerate(closes)
    ]


STATE_DIR = tempfile.mkdtemp()


def cfg(**kw):
    c = Config(api_key="k", api_secret="s")
    c.symbols = ["AAPL"]
    c.scan_stocks = False
    c.universe_size = 0
    c.world_symbols = []
    c.options_underlyings = []
    c.crypto_symbols = []
    c.crypto_timeframe = "5Min"
    c.state_file = os.path.join(STATE_DIR, f"state-{uuid.uuid4().hex}.json")
    for k, v in kw.items():
        setattr(c, k, v)
    return c


# Downtrend then a sharp turn up -> bullish crossover on the final bar.
DOWN_THEN_UP = [100 - 0.3 * i for i in range(30)] + [91 + i for i in range(15)]


def crossover_series():
    c = cfg()
    # find the first prefix whose last bar is the crossover
    for n in range(c.slow_ema + 2, len(DOWN_THEN_UP) + 1):
        if trend(make_bars(DOWN_THEN_UP[:n]), c).action == "buy":
            return DOWN_THEN_UP[:n]
    raise AssertionError("fixture never crosses")


class IndicatorTests(unittest.TestCase):
    def test_ema_constant(self):
        self.assertAlmostEqual(ema([5.0] * 20, 9)[-1], 5.0)
        self.assertEqual(ema([1, 2], 9), [])

    def test_rsi_extremes(self):
        self.assertEqual(rsi(list(range(30)), 14), 100.0)
        self.assertLess(rsi(list(range(30, 0, -1)), 14), 1)

    def test_atr_constant_range(self):
        closes = [10.0] * 30
        self.assertAlmostEqual(atr([c + 1 for c in closes], [c - 1 for c in closes], closes), 2.0)


class StrategyTests(unittest.TestCase):
    def test_not_enough_bars(self):
        self.assertEqual(trend(make_bars([1, 2, 3]), cfg()).action, "hold")

    def test_buy_on_bullish_cross(self):
        sig = trend(make_bars(crossover_series()), cfg())
        self.assertEqual(sig.action, "buy")
        self.assertLess(sig.stop, sig.price)
        self.assertGreater(sig.target, sig.price)

    def test_sell_on_bearish_cross(self):
        up_then_down = [100 + 0.3 * i for i in range(30)] + [109 - i for i in range(15)]
        actions = [trend(make_bars(up_then_down[:n]), cfg()).action
                   for n in range(25, len(up_then_down) + 1)]
        self.assertIn("sell", actions)
        self.assertNotIn("buy", actions)

    def test_rsi_filter_blocks_entry(self):
        sig = trend(make_bars(crossover_series()), cfg(rsi_max_entry=1))
        self.assertEqual(sig.action, "hold")
        self.assertIn("overbought", sig.reason)


class OtherStrategyTests(unittest.TestCase):
    def test_mean_reversion_buys_bounce(self):
        closes = [100.0] * 20 + [100 - 1.0 * i for i in range(1, 16)]
        actions = []
        for step in range(8):
            closes.append(closes[-1] + 0.6)
            actions.append(mean_reversion(make_bars(closes), cfg()).action)
        self.assertIn("buy", actions)

    def test_mean_reversion_exits_at_average(self):
        closes = [100.0] * 20 + [95.0] * 5 + [101.0]
        self.assertEqual(mean_reversion(make_bars(closes), cfg()).action, "sell")

    def test_breakout_needs_volume(self):
        bars = make_bars([100.0] * 25 + [103.0])
        self.assertEqual(breakout(bars, cfg()).action, "hold")
        bars[-1]["v"] = 5000
        sig = breakout(bars, cfg())
        self.assertEqual(sig.action, "buy")
        self.assertLess(sig.stop, sig.price)


def scripted(entries):
    """Strategy that buys at the given bar counts with stop -2 / target +4."""
    def fn(bars, c):
        price = bars[-1]["c"]
        if len(bars) in entries:
            return Signal("buy", price=price, stop=price - 2, target=price + 4)
        return Signal("hold", price=price)
    return fn


class BacktestTests(unittest.TestCase):
    def test_target_and_stop(self):
        c = cfg(cost_pct=0)
        closes = [100.0] * 40
        bars = make_bars(closes)
        bars[32]["h"] = 105  # target hit after entry at bar 31 (window length 32)
        bars[36]["l"] = 97   # stop hit after entry at bar 35
        self.assertEqual(backtest(scripted({32, 36}), bars, c), [2.0, -1.0])

    def test_stop_wins_when_both_touched(self):
        bars = make_bars([100.0] * 40)
        bars[32]["h"], bars[32]["l"] = 105, 97
        self.assertEqual(backtest(scripted({32}), bars, cfg(cost_pct=0)), [-1.0])

    def test_exit_at_end_of_day(self):
        bars = make_bars([100.0] * 40)
        for b in bars[34:]:
            b["t"] = "2026-09-25" + b["t"][10:]
        bars[33]["c"] = 101.0
        self.assertEqual(backtest(scripted({32}), bars, cfg(cost_pct=0)), [0.5])


class ExitRuleTests(unittest.TestCase):
    def test_trail_stop(self):
        c = cfg()
        self.assertEqual(trail_stop(98, 100, 2, 101, c), 98)      # not ahead enough yet
        self.assertEqual(trail_stop(98, 100, 2, 102, c), 100)     # +1R: stop to break-even
        self.assertEqual(trail_stop(100, 100, 2, 105, c), 103)    # keeps 1R behind the best price

    def test_backtest_trailing_stop_keeps_winner(self):
        bars = make_bars([100.0] * 40)  # entry at bar 31's close: 100, stop 98 (risk 2), target 104
        for i, (o, h, l, c) in zip(range(32, 36), [(100, 101, 100, 101), (101, 103, 101, 103),
                                                    (103, 103.5, 102.5, 103), (103, 103, 101, 101)]):
            bars[i].update(o=o, h=h, l=l, c=c)
        # best price 103.5 drags the stop to 101.5, so the pullback sells at +0.75R instead of riding to 98
        self.assertEqual(backtest(scripted({32}), bars, cfg(cost_pct=0, max_hold_minutes=0)), [0.75])

    def test_backtest_time_limit(self):
        bars = make_bars([100.0] * 80)
        bars[40]["c"] = 101.0
        # entry at bar 31; a 45-minute limit is 9 five-minute bars, so it sells at bar 40's close
        self.assertEqual(backtest(scripted({32}), bars, cfg(cost_pct=0, max_hold_minutes=45)), [0.5])

    def test_overnight_hold_buys_before_close_and_sells_after_open(self):
        day = [(100, 100.3, 99.7, 100, 1000)] * 78
        bars = session_days(4, day)
        entry = [b for b in bars if b["_min"] == 15 * 60 + 45][0]
        idx = bars.index(entry)
        self.assertEqual(overnight_hold(bars[:idx + 1], cfg()).action, "buy")
        first_bar = next(b for b in bars[idx:] if b["_min"] == 9 * 60 + 30)
        self.assertEqual(overnight_hold(bars[:bars.index(first_bar) + 1], cfg()).action, "sell")

    def test_backtest_holds_overnight_through_the_gap(self):
        day = [(100, 100.3, 99.7, 100, 1000)] * 78
        bars = session_days(4, day) + session_days(1, [(101, 101.3, 100.9, 101, 1000)] * 3)[-3:]
        for b in bars[-3:]:
            b["t"] = "2026-09-18" + b["t"][10:]
            b.pop("_min", None)
        annotate_sessions(bars)
        results = backtest(overnight_hold, bars, cfg(cost_pct=0))
        self.assertAlmostEqual(results[-1], 0.5)  # bought 100 at 15:50, sold 101 next morning, risk 2%


NY = timezone(timedelta(hours=-4))  # New York in September


class ExtendedSessionTests(unittest.TestCase):
    def test_session_calendar(self):
        def at(day, hour, minute=0):  # 2026-09-27 is a Sunday
            return datetime(2026, 9, day, hour, minute, tzinfo=NY)
        self.assertEqual(extended_session(at(27, 21)), ("overnight", at(28, 4)))  # Sunday night
        self.assertEqual(extended_session(at(28, 2))[0], "overnight")              # Monday 2 AM
        self.assertEqual(extended_session(at(28, 5))[0], "pre")
        self.assertEqual(extended_session(at(28, 12))[0], None)                    # regular hours
        self.assertEqual(extended_session(at(28, 17))[0], "after")
        self.assertEqual(extended_session(at(2, 21) + timedelta(days=30))[0], None)  # Friday 9 PM: closed
        self.assertEqual(extended_session(at(26, 12))[0], None)                    # Saturday
        self.assertTrue(in_session(23 * 60, "overnight") and in_session(60, "overnight"))
        self.assertFalse(in_session(12 * 60, "overnight"))


def overnight_bars(n=60):
    """Overnight-session 5-minute bars (Monday 9 PM New York onwards)."""
    start = datetime(2026, 9, 29, 1, 0, tzinfo=timezone.utc)
    return [{"t": (start + timedelta(minutes=5 * i)).isoformat().replace("+00:00", "Z"),
             "o": 100.0, "h": 100.2, "l": 99.8, "c": 100.0, "v": 500} for i in range(n)]


class OvernightTradingTests(unittest.TestCase):
    def setUp(self):
        self.c = cfg(universe_size=5, symbols=[])
        self.client = FakeClient(overnight_bars(), is_open=False, tradable=["AAPL"],
                                 daily_stats={"AAPL": (100, 1e9)})
        self.client.now = datetime(2026, 9, 29, 6, 1, tzinfo=timezone.utc)  # Tuesday 2:01 AM New York
        self.client.quotes = {"AAPL": (100.0, 100.1)}
        # every strategy "works" in replays, so the first one (trend) is chosen
        learner = Learner(self.c, backtester=lambda fn, bars, c, **kw: [1.0] * 10)
        self.bot = TradingBot(self.c, learner=learner, client=self.client)

    def run_two_loops(self):
        buy = lambda bars, c: Signal("buy", price=bars[-1]["c"], stop=bars[-1]["c"] - 1, target=bars[-1]["c"] + 2)
        with mock.patch.dict(STRATEGIES, {"trend": buy}):
            self.bot.run_once()                       # starts a live bar from the quote
            self.client.now += timedelta(minutes=5)
            self.bot.run_once()                       # that bar closes -> learn -> signal

    def test_buys_with_a_limit_order_overnight(self):
        self.run_two_loops()
        symbol, qty, limit = self.client.extended_entries[0]
        self.assertEqual(symbol, "AAPL")
        self.assertAlmostEqual(limit, 100.1 * 1.001)
        trade = self.bot.open_trades["AAPL"]
        self.assertEqual((trade["session"], trade["learn_key"], trade["managed"]), ("overnight", "AAPL@overnight", True))
        self.assertIn("boats", self.client.bar_feeds)
        self.assertIn("overnight", self.client.quote_feeds)
        self.assertEqual(self.client.orders, [])      # no regular (bracket) orders at night

    def test_sells_with_a_limit_order_when_the_stop_breaks(self):
        self.bot.open_trades["AAPL"] = {"strategy": "trend", "side": "long", "entry": 100.0, "stop": 99.0,
                                        "target": 102.0, "risk": 1.0, "opened_at": "2026-09-29T06:00:00Z",
                                        "managed": True, "session": "overnight", "learn_key": "AAPL@overnight"}
        self.client.positions = [{"symbol": "AAPL", "qty": "10", "avg_entry_price": "100", "current_price": "98.9"}]
        self.client.quotes = {"AAPL": (98.8, 98.9)}
        self.bot.run_once()
        self.bot.run_once()
        self.assertEqual(len(self.client.extended_exits), 1)  # once, then waits for the fill
        self.assertEqual(self.client.closed, [])              # never a market order at night

    def test_everything_sold_before_the_session_ends(self):
        self.client.now = datetime(2026, 9, 29, 7, 55, tzinfo=timezone.utc)  # 3:55 AM New York
        self.bot.open_trades["AAPL"] = {"strategy": "trend", "side": "long", "entry": 100.0, "stop": 90.0,
                                        "target": 120.0, "risk": 10.0, "opened_at": "2026-09-29T06:00:00Z",
                                        "managed": True, "session": "overnight"}
        self.client.positions = [{"symbol": "AAPL", "qty": "10", "avg_entry_price": "100", "current_price": "100"}]
        self.bot.run_once()
        self.assertEqual(self.client.extended_exits[0][:3], ("AAPL", "10", "sell"))

    def test_pre_market_needs_paid_data(self):
        self.client.now = datetime(2026, 9, 29, 10, 1, tzinfo=timezone.utc)  # 6:01 AM New York
        self.run_two_loops()
        self.assertFalse(hasattr(self.client, "extended_entries"))
        self.c.extended_feed = "sip"
        self.client.now += timedelta(minutes=5)
        self.run_two_loops()
        self.assertEqual(len(self.client.extended_entries), 1)
        self.assertIn("sip", self.client.quote_feeds)


class LearnerTests(unittest.TestCase):
    def test_picks_best_and_shrinks_small_samples(self):
        results = {"trend": [0.5] * 20, "mean_reversion": [3.0], "breakout": [-1.0] * 5}
        learner = Learner(cfg(), backtester=lambda fn, bars, c, **kw: results.get(fn.__name__, [-1.0] * 5))
        learner.update("AAPL", [])
        choice, scores = learner.choose("AAPL")
        self.assertEqual(choice, "trend")  # one lucky 3R trade shouldn't beat a steady record
        self.assertLess(scores["breakout"], 0)

    def test_live_results_persist_and_change_choice(self):
        c = cfg()
        learner = Learner(c, backtester=lambda fn, bars, c, **kw: [0.2] * 10 if fn is trend else [0.1] * 10)
        learner.update("AAPL", [])
        self.assertEqual(learner.choose("AAPL")[0], "trend")
        for _ in range(10):
            learner.record_trade("AAPL", "trend", -1.0)
        reloaded = Learner(c, backtester=learner.backtester)
        reloaded.update("AAPL", [])
        self.assertNotEqual(reloaded.choose("AAPL")[0], "trend")


def session_days(days, today=None):
    """Regular-session 5-minute bars (9:30-16:00 New York = 13:30-20:00 UTC in September), flat at 100
    with 1,000 shares a bar, for `days` past days, then `today` (a list of (open, high, low, close, volume))."""
    bars = []
    for d in range(days):
        for i in range(78):
            t = datetime(2026, 9, 14 + d, 13, 30, tzinfo=timezone.utc) + timedelta(minutes=5 * i)
            bars.append({"t": t.isoformat().replace("+00:00", "Z"), "o": 100.0, "h": 100.3, "l": 99.7,
                         "c": 100.0, "v": 1000})
    for i, (o, h, l, c, v) in enumerate(today or []):
        t = datetime(2026, 9, 14 + days, 13, 30, tzinfo=timezone.utc) + timedelta(minutes=5 * i)
        bars.append({"t": t.isoformat().replace("+00:00", "Z"), "o": o, "h": h, "l": l, "c": c, "v": v})
    return annotate_sessions(bars)


class SessionStrategyTests(unittest.TestCase):
    def test_annotation(self):
        bars = session_days(4, [(100, 101, 99.5, 100.8, 5000), (100.8, 101.5, 100.7, 101.4, 1500)])
        b = bars[-1]
        self.assertEqual((b["_min"], b["_orh"], b["_orl"]), (9 * 60 + 35, 101, 99.5))
        self.assertAlmostEqual(b["_rv"], 5.0)
        self.assertEqual(b["_pc"], 100.0)
        self.assertEqual(bars[0]["_min"], 9 * 60 + 30)

    def test_orb_buys_breakout_on_stock_in_play(self):
        bars = session_days(4, [(100, 101, 99.5, 100.8, 5000), (100.8, 101.5, 100.7, 101.4, 1500)])
        sig = orb(bars, cfg())
        self.assertEqual(sig.action, "buy")
        self.assertEqual(sig.stop, 99.5)

    def test_orb_ignores_normal_volume_and_short_twin_catches_breakdowns(self):
        quiet = session_days(4, [(100, 101, 99.5, 100.8, 1000), (100.8, 101.5, 100.7, 101.4, 1500)])
        self.assertEqual(orb(quiet, cfg()).action, "hold")
        down = session_days(4, [(100, 100.5, 99, 99.2, 5000), (99.2, 99.3, 98.5, 98.6, 1500)])
        sig = STRATEGIES["orb_short"](down, cfg())
        self.assertEqual((sig.action, sig.side), ("buy", "short"))
        self.assertAlmostEqual(sig.stop, 100.5)

    def test_vwap_cross(self):
        today = [(100, 100.2, 98.8, 99.0, 3000)] * 3 + [(99.0, 99.1, 98.9, 99.0, 1000)] * 2
        below = session_days(4, today)
        self.assertEqual(vwap_trend(below, cfg()).action, "sell")
        crossed = session_days(4, today + [(99.0, 101.2, 99.0, 101.0, 4000)])
        self.assertEqual(vwap_trend(crossed, cfg()).action, "buy")

    def test_intraday_momentum_at_1530(self):
        def day(ten_oclock_close):
            bars = [(100, 100.3, 99.7, 100, 1000)] * 78
            bars[5] = (100, 101.3, 99.7, ten_oclock_close, 1000)  # the 9:55 bar closes at 10:00
            return bars[:72]  # up to the bar that closes at 15:30
        self.assertEqual(intraday_momentum(session_days(4, day(101)), cfg()).action, "buy")
        self.assertEqual(intraday_momentum(session_days(4, day(99)), cfg()).action, "hold")
        self.assertEqual(STRATEGIES["intraday_momentum_short"](session_days(4, day(99)), cfg()).action, "buy")

    def test_session_strategies_skip_unannotated_bars(self):
        bars = make_bars([100.0] * 40)
        for fn in (orb, vwap_trend, intraday_momentum):
            self.assertEqual(fn(bars, cfg()).action, "hold")


class ShortAndOptionTests(unittest.TestCase):
    def test_mirror_keeps_last_close_and_flips_direction(self):
        bars = make_bars([100.0, 110.0, 120.0])
        m = mirror(bars)
        self.assertAlmostEqual(m[-1]["c"], 120.0)
        self.assertGreater(m[0]["c"], m[-1]["c"])
        self.assertTrue(all(b["h"] >= b["l"] for b in m))

    def test_short_signal_has_stop_above(self):
        sig = STRATEGIES["trend_short"](make_bars(bearish_series()), cfg())
        self.assertEqual((sig.action, sig.side), ("buy", "short"))
        self.assertGreater(sig.stop, sig.price)
        self.assertLess(sig.target, sig.price)

    def test_short_size(self):
        self.assertEqual(position_size(100_000, 1e9, 100, 110, cfg(), side="short"), 100)

    def test_pick_option(self):
        opts = [option("call", "far", 0.8, 5.0, 5.1), option("call", "atm", 0.49, 2.0, 2.1),
                option("call", "wide", 0.5, 1.0, 2.0), option("call", "atm_later", 0.5, 2.5, 2.6, exp="2026-10-16")]
        self.assertEqual(pick_option(opts, cfg())["symbol"], "atm")
        self.assertIsNone(pick_option([opts[2]], cfg()))

    def test_option_contracts(self):
        # $1000 risk / ($210 * 50%) = 9.5 -> 9; cap 20% = $20k / $210 = 95
        self.assertEqual(option_contracts(100_000, 1e9, 2.1, cfg()), 9)
        self.assertEqual(option_contracts(100_000, 300, 2.1, cfg()), 1)


class RiskTests(unittest.TestCase):
    def test_account_limits_grow_with_balance(self):
        c = cfg(options_underlyings=["SPY"])
        tiny, small, mid, big = (account_limits(e, c) for e in (10, 500, 5_000, 30_000))
        self.assertEqual((tiny.max_positions, tiny.max_position_pct), (2, 0.5))
        self.assertEqual((small.max_positions, small.shorts, small.options), (4, False, False))
        self.assertEqual((mid.max_positions, mid.shorts, mid.options, mid.pdt_limited), (8, True, True, True))
        self.assertFalse(big.pdt_limited)
        self.assertIn("tiny account", tiny.describe())

    def test_size_by_risk(self):
        # risk $1000 (1% of 100k), $2/share -> 500 shares, cap 20% = 200 shares at $100
        self.assertEqual(position_size(100_000, 1e9, 100, 98, cfg()), 200)
        self.assertEqual(position_size(100_000, 1e9, 100, 90, cfg()), 100)

    def test_fractional_crypto_size(self):
        qty = position_size(100_000, 1e9, 60_000, 59_000, cfg(), fractional=True)
        self.assertAlmostEqual(qty, 0.333333)
        self.assertEqual(position_size(100_000, 0.5, 60_000, 59_000, cfg(), fractional=True), 0)  # < $1

    def test_size_limited_by_cash(self):
        self.assertEqual(position_size(100_000, 500, 100, 90, cfg()), 5)

    def test_invalid_stop(self):
        self.assertEqual(position_size(100_000, 1e9, 100, 101, cfg()), 0)

    def test_daily_loss(self):
        self.assertTrue(daily_loss_hit({"equity": "96000", "last_equity": "100000"}, cfg()))
        self.assertFalse(daily_loss_hit({"equity": "99000", "last_equity": "100000"}, cfg()))


class ConfigTests(unittest.TestCase):
    def test_refuses_live_url(self):
        env = {"ALPACA_API_KEY_ID": "k", "ALPACA_API_SECRET_KEY": "s",
               "ALPACA_BASE_URL": "https://api.alpaca.markets"}
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(SystemExit):
                Config.from_env()

    def test_paper_url_normalised(self):
        env = {"ALPACA_API_KEY_ID": "k", "ALPACA_API_SECRET_KEY": "s",
               "ALPACA_BASE_URL": "https://paper-api.alpaca.markets"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(Config.from_env().base_url, "https://paper-api.alpaca.markets/v2")


class FakeClient:
    def __init__(self, bars, positions=(), minutes_to_close=120, equity="100000", last_equity="100000",
                 is_open=True, actives=(), shortable=True, options=(), daytrade_count=0, asset=None,
                 bars_by_symbol=None, tradable=(), daily_stats=None):
        self.bars = bars
        self.positions = list(positions)
        self.now = datetime.fromisoformat(bars[-1]["t"].replace("Z", "+00:00")) + timedelta(minutes=5)
        self.close = self.now + timedelta(minutes=minutes_to_close)
        self.equity, self.last_equity = equity, last_equity
        self.orders, self.closed, self.closed_all = [], [], 0
        self.sell_fill = None
        self.open_orders = []
        self.is_open, self.actives = is_open, list(actives)
        self.bar_requests = []
        self.shortable, self.options = shortable, list(options)
        self.daytrade_count, self.asset = daytrade_count, asset or {}
        self.bars_by_symbol = bars_by_symbol or {}
        self.tradable, self.daily_stats = list(tradable), daily_stats or {}
        self.universe_calls = 0
        self.option_orders, self.canceled, self.exit_sides = [], [], []

    def get_clock(self):
        return {"is_open": self.is_open, "timestamp": self.now.isoformat(),
                "next_close": self.close.isoformat(), "next_open": ""}

    def get_account(self):
        return {"equity": self.equity, "last_equity": self.last_equity,
                "buying_power": str(2 * float(self.equity)), "non_marginable_buying_power": self.equity,
                "daytrade_count": self.daytrade_count}

    def get_most_active_stocks(self, top):
        return self.actives

    def get_movers(self, top=20):
        return getattr(self, "movers", [])

    def get_overnight_tradable(self):
        return set(getattr(self, "overnight_symbols", self.tradable))

    def get_latest_quotes(self, symbols, feed):
        self.quote_feeds = getattr(self, "quote_feeds", []) + [feed]
        return {x: q for x, q in getattr(self, "quotes", {}).items() if x in symbols}

    def submit_extended_entry(self, symbol, qty, limit_price, client_order_id):
        self.extended_entries = getattr(self, "extended_entries", []) + [(symbol, qty, limit_price)]
        self.open_orders.append({"id": "x1", "symbol": symbol, "side": "buy", "client_order_id": client_order_id})
        return {"id": "x1"}

    def get_positions(self):
        return self.positions

    def get_open_orders(self):
        return self.open_orders

    def get_bars(self, symbol, timeframe, lookback_days=5):
        self.bar_requests.append(symbol)
        return self.bars_by_symbol.get(symbol, self.bars)

    def get_stock_bars(self, symbols, timeframe, lookback_days=5, feed=None):
        self.bar_requests.extend(symbols)
        self.bar_feeds = getattr(self, "bar_feeds", []) + [feed]
        return {x: [dict(b) for b in self.bars_by_symbol.get(x, self.bars)] for x in symbols}

    def get_tradable_stocks(self):
        self.universe_calls += 1
        return self.tradable

    def get_daily_stats(self, symbols):
        return {x: self.daily_stats[x] for x in symbols if x in self.daily_stats}

    def get_last_exit_fill(self, symbol, after, exit_side="sell"):
        self.exit_sides.append(exit_side)
        return self.sell_fill

    def get_asset(self, symbol):
        return {"shortable": self.shortable, "easy_to_borrow": self.shortable, "fractionable": True, **self.asset}

    def get_option_candidates(self, underlying, kind, price, exp_from, exp_to):
        return [o for o in self.options if o["kind"] == kind]

    def submit_option_buy(self, contract, qty, limit_price, client_order_id=None):
        self.option_orders.append((contract, qty, limit_price))
        self.open_orders.append({"id": "o1", "symbol": contract, "side": "buy", "client_order_id": client_order_id})

    def cancel_order(self, order_id):
        self.canceled.append(order_id)

    def submit_extended_exit(self, symbol, qty, side, limit_price):
        self.extended_exits = getattr(self, "extended_exits", []) + [(symbol, qty, side, limit_price)]
        return {"id": "ext1"}

    def submit_crypto_stop(self, symbol, qty, stop_price):
        self.stops = getattr(self, "stops", []) + [(symbol, qty, stop_price)]
        return {"id": f"stop{len(self.stops)}"}

    def submit_entry(self, symbol, qty, side, tp, sl, client_order_id=None, fractional=False, overnight=False):
        self.orders.append((symbol, qty, tp, sl))
        self.overnight_flags = getattr(self, "overnight_flags", []) + [overnight]
        self.fractional_flags = getattr(self, "fractional_flags", []) + [fractional]
        self.sides = getattr(self, "sides", []) + [side]
        self.open_orders.append({"id": "e1", "symbol": symbol, "side": side, "client_order_id": client_order_id})
        self.order_ids = getattr(self, "order_ids", []) + [client_order_id]

    def close_position(self, symbol):
        self.closed.append(symbol)
        self.positions = [p for p in self.positions if p["symbol"] != symbol.replace("/", "")]

    def close_all_positions(self):
        self.closed_all += 1


def trend_learner(c=None, short=False):
    """Learner whose replay always says trend (or trend_short) wins, so bot tests are deterministic.

    Short strategies are replayed as their long twin on a flipped chart, so tell them apart by the prices.
    """
    def fake_backtest(fn, bars, c, **kw):
        flipped = bars and bars[0]["c"] > 100
        if fn is trend and flipped == short:
            return [1.0] * 10
        return [-1.0] * 10
    return Learner(c or cfg(), backtester=fake_backtest)


UP_THEN_DOWN = [100 + 0.3 * i for i in range(30)] + [109 - i for i in range(15)]


def bearish_series():
    for n in range(25, len(UP_THEN_DOWN) + 1):
        if STRATEGIES["trend_short"](make_bars(UP_THEN_DOWN[:n]), cfg()).action == "buy":
            return UP_THEN_DOWN[:n]
    raise AssertionError("fixture never crosses down")


def option(kind, symbol, delta, bid, ask, exp="2026-10-09"):
    return {"kind": kind, "symbol": symbol, "delta": delta, "bid": bid, "ask": ask, "expiration": exp, "strike": 100}


class BotTests(unittest.TestCase):
    def test_places_bracket_buy_once_per_bar(self):
        client = FakeClient(make_bars(crossover_series()))
        bot = TradingBot(cfg(), learner=trend_learner(), client=client)
        bot.run_once()
        bot.last_bar_seen.clear()  # force a re-fetch of the same bar
        bot.run_once()  # same bar again: must not double-buy
        self.assertEqual(len(client.orders), 1)
        symbol, qty, tp, sl = client.orders[0]
        self.assertEqual(symbol, "AAPL")
        self.assertGreater(qty, 0)
        self.assertLess(sl, tp)
        self.assertTrue(client.order_ids[0].startswith("trend-AAPL-"))
        self.assertEqual(bot.learner.state["active"]["AAPL"], "trend")
        self.assertIn("AAPL", bot.open_trades)

    def test_learns_from_closed_trade(self):
        client = FakeClient(make_bars(crossover_series()))
        bot = TradingBot(cfg(), learner=trend_learner(), client=client)
        bot.open_trades["AAPL"] = {"strategy": "trend", "entry": 100.0, "stop": 98.0, "opened_at": "x"}
        client.sell_fill = 104.0
        bot.run_once()
        self.assertEqual(bot.learner.state["live_results"]["trend"]["AAPL"], [2.0])

    def test_sits_out_when_nothing_works(self):
        c = cfg()
        learner = Learner(c, backtester=lambda fn, bars, c, **kw: [-1.0] * 10)
        client = FakeClient(make_bars(crossover_series()))
        TradingBot(c, learner=learner, client=client).run_once()
        self.assertEqual(client.orders, [])
        self.assertIsNone(learner.state["active"]["AAPL"])

    def test_dry_run_places_nothing(self):
        client = FakeClient(make_bars(crossover_series()))
        TradingBot(cfg(), learner=trend_learner(), client=client, dry_run=True).run_once()
        self.assertEqual(client.orders, [])

    def test_no_entries_near_close(self):
        client = FakeClient(make_bars(crossover_series()), minutes_to_close=20)
        TradingBot(cfg(), learner=trend_learner(), client=client).run_once()
        self.assertEqual(client.orders, [])

    def test_flattens_at_end_of_day(self):
        client = FakeClient(make_bars(crossover_series()), positions=[{"symbol": "AAPL", "asset_class": "us_equity", "avg_entry_price": "100"},
                                       {"symbol": "BTCUSD", "asset_class": "crypto", "avg_entry_price": "100"}],
                            minutes_to_close=5)
        c = cfg(crypto_symbols=["BTC/USD"])
        TradingBot(c, learner=trend_learner(c), client=client).run_once()
        self.assertEqual(client.closed, ["AAPL"])  # crypto keeps trading overnight

    def test_daily_loss_halts(self):
        client = FakeClient(make_bars(crossover_series()), positions=[{"symbol": "AAPL", "avg_entry_price": "100"}],
                            equity="95000", last_equity="100000")
        TradingBot(cfg(), learner=trend_learner(), client=client).run_once()
        self.assertEqual(client.closed_all, 1)
        self.assertEqual(client.orders, [])

    def test_crypto_trades_when_stock_market_closed(self):
        c = cfg(crypto_symbols=["BTC/USD"])
        client = FakeClient(make_bars(crossover_series()), is_open=False)
        bot = TradingBot(c, learner=trend_learner(c), client=client)
        bot.run_once()
        self.assertEqual(client.bar_requests, ["BTC/USD"])  # stocks skipped while closed
        self.assertEqual(len(client.orders), 1)
        symbol, qty, tp, sl = client.orders[0]
        self.assertEqual(symbol, "BTC/USD")
        self.assertNotEqual(qty, int(qty))  # fractional coins
        self.assertIn("target", bot.open_trades["BTC/USD"])

    def test_crypto_stop_parked_at_alpaca_once_filled(self):
        c = cfg(crypto_symbols=["BTC/USD"])
        client = FakeClient(make_bars(crossover_series()), is_open=False,
                            positions=[{"symbol": "BTCUSD", "asset_class": "crypto", "qty": "0.5",
                                        "avg_entry_price": "100", "current_price": "100"}])
        bot = TradingBot(c, learner=trend_learner(c), client=client)
        bot.open_trades["BTC/USD"] = {"strategy": "trend", "entry": 100.0, "stop": 98.0, "target": 104.0,
                                      "opened_at": "x"}
        bot.run_once()
        bot.run_once()
        self.assertEqual(client.stops, [("BTC/USD", "0.5", 98.0)])  # placed once, not every loop
        self.assertEqual(bot.open_trades["BTC/USD"]["stop_order_id"], "stop1")

    def test_crypto_stop_loss_enforced_by_bot(self):
        c = cfg(crypto_symbols=["BTC/USD"])
        client = FakeClient(make_bars(crossover_series()), is_open=False,
                            positions=[{"symbol": "BTCUSD", "asset_class": "crypto", "qty": "0.5",
                                        "avg_entry_price": "100", "current_price": "97"}])
        bot = TradingBot(c, learner=trend_learner(c), client=client)
        bot.open_trades["BTC/USD"] = {"strategy": "trend", "entry": 100.0, "stop": 98.0, "target": 104.0,
                                      "opened_at": "x"}
        bot.run_once()
        self.assertEqual(client.closed, ["BTC/USD"])

    def test_universe_has_world_and_filtered_scan(self):
        c = cfg(scan_stocks=True, world_symbols=["EWJ"])
        client = FakeClient(make_bars(crossover_series()),
                            actives=[("NVDA", 225.0, 5e8), ("PENNY", 0.5, 5e8), ("THIN", 50.0, 1e5)])
        bot = TradingBot(c, learner=trend_learner(c), client=client)
        self.assertEqual(bot.stock_universe(client.now), ["AAPL", "EWJ", "NVDA"])

    def test_only_fetches_when_new_bar_due(self):
        client = FakeClient(make_bars(crossover_series()))
        bot = TradingBot(cfg(), learner=trend_learner(), client=client)
        bot.run_once()
        bot.run_once()
        self.assertEqual(client.bar_requests, ["AAPL"])

    def test_short_sells_borrowable_stock(self):
        client = FakeClient(make_bars(bearish_series()))
        bot = TradingBot(cfg(), learner=trend_learner(short=True), client=client)
        bot.run_once()
        self.assertEqual(client.sides, ["sell"])
        symbol, qty, tp, sl = client.orders[0]
        self.assertGreater(sl, tp)  # stop above, target below
        self.assertEqual(bot.open_trades["AAPL"]["side"], "short")

    def test_no_short_when_not_borrowable(self):
        client = FakeClient(make_bars(bearish_series()), shortable=False)
        bot = TradingBot(cfg(), learner=trend_learner(short=True), client=client)
        bot.run_once()
        self.assertEqual(client.orders, [])
        self.assertNotIn("trend_short", bot.learner.allowed["AAPL"])

    def test_learns_from_closed_short(self):
        client = FakeClient(make_bars(bearish_series()))
        bot = TradingBot(cfg(), learner=trend_learner(short=True), client=client)
        bot.open_trades["AAPL"] = {"strategy": "trend_short", "side": "short", "entry": 100.0, "stop": 102.0,
                                   "opened_at": "x"}
        client.sell_fill = 96.0
        bot.run_once()
        self.assertEqual(client.exit_sides[0], "buy")
        self.assertEqual(bot.learner.state["live_results"]["trend_short"]["AAPL"], [2.0])

    def test_bullish_signal_buys_call_on_options_name(self):
        c = cfg(options_underlyings=["AAPL"])
        client = FakeClient(make_bars(crossover_series()),
                            options=[option("call", "AAPL261009C00100000", 0.52, 2.0, 2.1),
                                     option("put", "AAPL261009P00100000", -0.5, 2.0, 2.1)])
        bot = TradingBot(c, learner=trend_learner(c), client=client)
        bot.run_once()
        self.assertEqual(client.orders, [])  # no shares
        contract, qty, limit = client.option_orders[0]
        self.assertEqual(contract, "AAPL261009C00100000")
        self.assertEqual(limit, 2.1)
        trade = bot.open_trades[contract]
        self.assertEqual((trade["underlying"], trade["stop"], trade["target"]), ("AAPL", 1.05, 4.2))

    def test_bearish_signal_buys_put(self):
        c = cfg(options_underlyings=["AAPL"])
        client = FakeClient(make_bars(bearish_series()), shortable=False,
                            options=[option("put", "AAPL261009P00100000", -0.48, 2.0, 2.1)])
        TradingBot(c, learner=trend_learner(c, short=True), client=client).run_once()
        self.assertEqual(client.option_orders[0][0], "AAPL261009P00100000")
        self.assertEqual(client.orders, [])

    def test_falls_back_to_shares_without_liquid_option(self):
        c = cfg(options_underlyings=["AAPL"])
        client = FakeClient(make_bars(crossover_series()), options=[option("call", "WIDE", 0.5, 1.0, 2.0)])
        TradingBot(c, learner=trend_learner(c), client=client).run_once()
        self.assertEqual(client.option_orders, [])
        self.assertEqual(len(client.orders), 1)

    def test_option_stop_enforced_by_bot(self):
        occ = "AAPL261009C00100000"
        client = FakeClient(make_bars(crossover_series()),
                            positions=[{"symbol": occ, "asset_class": "us_option", "avg_entry_price": "2.0",
                                        "current_price": "0.9"}])
        bot = TradingBot(cfg(), learner=trend_learner(), client=client)
        bot.open_trades[occ] = {"strategy": "trend", "side": "long", "underlying": "AAPL", "entry": 2.0,
                                "stop": 1.0, "target": 4.0, "opened_at": "x", "managed": True}
        bot.run_once()
        self.assertEqual(client.closed, [occ])

    def test_end_of_day_cancels_unfilled_entries(self):
        client = FakeClient(make_bars(crossover_series()), minutes_to_close=5)
        client.open_orders = [{"id": "abc", "symbol": "AAPL261009C00100000", "side": "buy",
                               "client_order_id": "trend-AAPL261009C00100000-1234"},
                              {"id": "leg", "symbol": "MSFT", "side": "sell", "client_order_id": "uuid-ish"}]
        TradingBot(cfg(), learner=trend_learner(), client=client).run_once()
        self.assertEqual(client.canceled, ["abc"])

    # --- account-size modes ---
    def test_tiny_account_buys_a_slice_of_a_share(self):
        client = FakeClient(make_bars(crossover_series()), equity="10", last_equity="10")
        bot = TradingBot(cfg(), learner=trend_learner(), client=client)
        bot.run_once()
        symbol, qty, tp, sl = client.orders[0]
        self.assertEqual(client.fractional_flags, [True])
        self.assertLess(qty, 1)
        self.assertGreaterEqual(qty * 96, 1.0)          # at least Alpaca's $1 minimum
        self.assertLessEqual(qty * 96, 10 * 0.5 + 1e-9)  # at most half of a tiny account
        self.assertTrue(bot.open_trades["AAPL"]["managed"])  # no bracket, so the bot watches the stop

    def test_big_account_still_uses_whole_shares(self):
        client = FakeClient(make_bars(crossover_series()))
        TradingBot(cfg(), learner=trend_learner(), client=client).run_once()
        self.assertEqual(client.fractional_flags, [False])

    def test_no_shorts_or_options_below_2000(self):
        c = cfg(options_underlyings=["AAPL"])
        client = FakeClient(make_bars(bearish_series()), equity="500", last_equity="500",
                            options=[option("put", "AAPL261009P00100000", -0.5, 2.0, 2.1)])
        bot = TradingBot(c, learner=trend_learner(c, short=True), client=client)
        bot.run_once()
        self.assertNotIn("trend_short", bot.learner.allowed["AAPL"])
        self.assertEqual((client.orders, client.option_orders), ([], []))

    def test_small_account_skips_calls_and_buys_shares(self):
        c = cfg(options_underlyings=["AAPL"])
        client = FakeClient(make_bars(crossover_series()), equity="500", last_equity="500",
                            options=[option("call", "AAPL261009C00100000", 0.5, 2.0, 2.1)])
        TradingBot(c, learner=trend_learner(c), client=client).run_once()
        self.assertEqual(client.option_orders, [])
        self.assertEqual(len(client.orders), 1)

    def test_day_trade_limit_blocks_stock_entries(self):
        client = FakeClient(make_bars(crossover_series()), equity="1000", last_equity="1000", daytrade_count=3)
        TradingBot(cfg(), learner=trend_learner(), client=client).run_once()
        self.assertEqual(client.orders, [])

    def test_day_trades_reserved_for_trades_opened_today(self):
        client = FakeClient(make_bars(crossover_series()), equity="1000", last_equity="1000", daytrade_count=2)
        bot = TradingBot(cfg(), learner=trend_learner(), client=client)
        today = client.now.date().isoformat()
        bot.open_trades["MSFT"] = {"strategy": "trend", "entry": 1, "stop": 0.5, "opened_at": today + "T14:00:00Z"}
        client.positions = [{"symbol": "MSFT", "avg_entry_price": "1", "current_price": "1"}]
        bot.run_once()
        self.assertEqual(client.orders, [])  # 2 used + MSFT's exit later today = 3

    def test_day_trade_limit_ignores_crypto_and_big_accounts(self):
        c = cfg(crypto_symbols=["BTC/USD"])
        client = FakeClient(make_bars(crossover_series()), is_open=False, equity="1000", last_equity="1000",
                            daytrade_count=3)
        TradingBot(c, learner=trend_learner(c), client=client).run_once()
        self.assertEqual(len(client.orders), 1)
        client = FakeClient(make_bars(crossover_series()), daytrade_count=3)
        TradingBot(cfg(), learner=trend_learner(), client=client).run_once()
        self.assertEqual(len(client.orders), 1)

    def test_crypto_below_min_order_size_is_skipped(self):
        c = cfg(crypto_symbols=["BTC/USD"])
        client = FakeClient(make_bars(crossover_series()), is_open=False, equity="10", last_equity="10",
                            asset={"min_order_size": "1"})
        TradingBot(c, learner=trend_learner(c), client=client).run_once()
        self.assertEqual(client.orders, [])

    # --- all major stocks ---
    def test_universe_is_most_traded_over_min_price_built_once_a_day(self):
        c = cfg(universe_size=2)
        client = FakeClient(make_bars(crossover_series()), tradable=["AAA", "BBB", "CCC", "PENNY"],
                            daily_stats={"AAA": (50, 1e8), "BBB": (20, 5e8), "CCC": (30, 1e6), "PENNY": (2, 9e9)})
        bot = TradingBot(c, learner=trend_learner(c), client=client)
        self.assertEqual(bot.stock_universe(client.now), ["AAPL", "BBB", "AAA"])
        bot.stock_universe(client.now)
        self.assertEqual(client.universe_calls, 1)

    def test_best_scoring_signal_gets_the_last_slot(self):
        c = cfg(symbols=["WEAK", "STRONG"], max_open_positions=1)
        series = crossover_series()
        weak, strong = make_bars(series), make_bars([x * 3 for x in series])

        def fake_backtest(fn, bars, c, **kw):
            if fn is not trend or bars[0]["c"] > 150 and bars[-1]["c"] < 150:
                return [-1.0] * 10  # the short twins (flipped charts) lose
            return [2.0] * 10 if bars[0]["c"] > 150 else [0.5] * 10

        client = FakeClient(weak, bars_by_symbol={"WEAK": weak, "STRONG": strong})
        TradingBot(c, learner=Learner(c, backtester=fake_backtest), client=client).run_once()
        self.assertEqual([o[0] for o in client.orders], ["STRONG"])

    def test_only_new_bars_are_downloaded_after_the_first_time(self):
        client = FakeClient(make_bars(crossover_series()))
        bot = TradingBot(cfg(), learner=trend_learner(), client=client)
        bot.run_once()
        first = len(bot.bars["AAPL"])
        extra = make_bars(crossover_series() + [200.0])[-1]
        client.bars = [extra]
        client.now += timedelta(minutes=5)
        bot.run_once()
        self.assertEqual(len(bot.bars["AAPL"]), first + 1)

    def test_leftover_crypto_is_sold_when_crypto_is_off(self):
        client = FakeClient(make_bars(crossover_series()), is_open=False,
                            positions=[{"symbol": "BTCUSD", "asset_class": "crypto", "qty": "0.1",
                                        "avg_entry_price": "100", "current_price": "100"}])
        bot = TradingBot(cfg(), learner=trend_learner(), client=client)  # no memory of buying it
        bot.run_once()
        self.assertEqual(client.closed, ["BTCUSD"])

    # --- exits and overnight holds ---
    def test_live_trailing_stop_and_time_limit(self):
        client = FakeClient(make_bars(crossover_series()),
                            positions=[{"symbol": "AAPL", "avg_entry_price": "100", "current_price": "103"},
                                       {"symbol": "MSFT", "avg_entry_price": "100", "current_price": "100.5"}])
        bot = TradingBot(cfg(), learner=trend_learner(), client=client)
        bot.open_trades["AAPL"] = {"strategy": "trend", "side": "long", "entry": 100.0, "stop": 98.0, "risk": 2.0,
                                   "best": 105.0, "opened_at": client.now.isoformat().replace("+00:00", "Z")}
        old = (client.now - timedelta(minutes=150)).isoformat().replace("+00:00", "Z")
        bot.open_trades["MSFT"] = {"strategy": "trend", "side": "long", "entry": 100.0, "stop": 98.0, "risk": 2.0,
                                   "opened_at": old}
        bot.run_once()
        self.assertEqual(sorted(client.closed), ["AAPL", "MSFT"])  # AAPL fell to its trailed stop 103; MSFT timed out

    def test_end_of_day_keeps_overnight_hold(self):
        client = FakeClient(make_bars(crossover_series()), minutes_to_close=5,
                            positions=[{"symbol": "AAPL", "asset_class": "us_equity", "avg_entry_price": "100",
                                        "current_price": "100"},
                                       {"symbol": "MSFT", "asset_class": "us_equity", "avg_entry_price": "100",
                                        "current_price": "100"}])
        bot = TradingBot(cfg(), learner=trend_learner(), client=client)
        bot.open_trades["MSFT"] = {"strategy": "overnight_hold", "side": "long", "entry": 100.0, "stop": 98.0,
                                   "opened_at": "x"}
        bot.run_once()
        self.assertEqual(client.closed, ["AAPL"])

    def test_overnight_entry_just_before_close_uses_gtc_and_skips_day_trade_limit(self):
        day = [(100, 100.3, 99.7, 100, 1000)] * 76  # up to the bar that closes at 15:50
        bars = session_days(4, day)
        c = cfg()

        def fake_backtest(fn, bars, c, **kw):
            return [1.0] * 10 if fn is overnight_hold and bars[0]["c"] <= 100 else [-1.0] * 10

        client = FakeClient(bars, minutes_to_close=9, equity="1000", last_equity="1000", daytrade_count=3)
        TradingBot(c, learner=Learner(c, backtester=fake_backtest), client=client).run_once()
        self.assertEqual(len(client.orders), 1)
        self.assertEqual(client.overnight_flags, [True])

    def test_overnight_hold_protected_outside_market_hours(self):
        client = FakeClient(make_bars(crossover_series()), is_open=False,
                            positions=[{"symbol": "AAPL", "asset_class": "us_equity", "qty": "10",
                                        "avg_entry_price": "100", "current_price": "97"}])
        bot = TradingBot(cfg(), learner=trend_learner(), client=client)
        bot.open_trades["AAPL"] = {"strategy": "overnight_hold", "side": "long", "entry": 100.0, "stop": 98.0,
                                   "opened_at": "x"}
        bot.run_once()
        bot.run_once()  # only one protective order
        self.assertEqual(len(client.extended_exits), 1)
        symbol, qty, side, limit = client.extended_exits[0]
        self.assertEqual((symbol, side), ("AAPL", "sell"))
        self.assertLess(limit, 97)

    def test_drops_forming_bar(self):
        bars = make_bars([1, 2, 3])
        last = datetime.fromisoformat(bars[-1]["t"].replace("Z", "+00:00"))
        self.assertEqual(len(completed_bars(bars, "5Min", last + timedelta(minutes=2))), 2)
        self.assertEqual(len(completed_bars(bars, "5Min", last + timedelta(minutes=5))), 3)


if __name__ == "__main__":
    unittest.main()
