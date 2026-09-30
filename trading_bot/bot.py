"""Self-learning paper-trading bot for Alpaca.

Trades the ~500 most-traded US stocks and ETFs (long and short) and options (calls and
puts) during US market hours, and optionally crypto 24/7.

Run:  python -m trading_bot.bot            # trade in a loop
      python -m trading_bot.bot --once     # single pass, then exit
      python -m trading_bot.bot --dry-run  # log decisions, place no orders
      python -m trading_bot.bot --report   # show what the bot has learned
"""
import argparse
import bisect
import logging
import time
import math
import uuid
from datetime import datetime, timedelta, timezone

from .alpaca_client import AlpacaClient, AlpacaError, is_crypto, norm
from .config import Config
from .extended import ExtendedHoursMixin
from .learner import Learner, trail_stop
from .risk import account_limits, daily_loss_hit, option_contracts, pick_option, position_size
from .strategy import LONG_STRATEGIES, OPEN_MIN, STRATEGIES, _ny_day_and_minute, annotate_sessions


NEVER_CHOSEN = {"overnight_hold_short"}  # kept defined so trades opened before still resolve


def _overnight(strategy_name):
    return bool(strategy_name) and getattr(STRATEGIES.get(strategy_name), "overnight", False)

log = logging.getLogger("trading_bot")


def _parse_ts(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _timeframe_minutes(timeframe):
    units = {"Min": 1, "Hour": 60}
    for suffix, mult in units.items():
        if timeframe.endswith(suffix):
            return int(timeframe[: -len(suffix)]) * mult
    raise ValueError(f"unsupported timeframe {timeframe}")


def completed_bars(bars, timeframe, now):
    """Drop the still-forming last bar so signals only use closed candles."""
    if bars and _parse_ts(bars[-1]["t"]) + timedelta(minutes=_timeframe_minutes(timeframe)) > now:
        return bars[:-1]
    return bars


class TradingBot(ExtendedHoursMixin):
    def __init__(self, cfg, client=None, learner=None, dry_run=False):
        self.cfg = cfg
        self.client = client or AlpacaClient(cfg)
        self.learner = learner or Learner(cfg)
        self.dry_run = dry_run
        self.last_bar_seen = {}
        self.halted_day = None
        self.scanned = []
        self.scanned_at = None
        self.asset_cache = {}
        self.limits = None
        self.universe = []         # the day's most-traded stocks
        self.universe_day = None
        self.bars = {}             # symbol -> recent closed-and-forming bars, kept between loops
        self.last_stock_bucket = None
        self.replayed_at = {}      # symbol -> when its strategies were last replayed
        self.stock_open = False
        self.news = {}             # symbol -> sorted news times (ISO)
        self.news_until = None     # newest article time fetched so far
        self.bench_day = None
        self._init_extended()

    @property
    def open_trades(self):
        return self.learner.state["open_trades"]

    # --- which symbols to trade ---
    def _build_universe(self, today):
        """Once a day: the universe_size most-traded US stocks/ETFs (by dollar volume) above min_price."""
        if not self.cfg.universe_size or self.universe_day == today:
            return
        try:
            stats = self.client.get_daily_stats(self.client.get_tradable_stocks())
        except AlpacaError as exc:
            log.error("Couldn't build today's stock list, keeping the previous one: %s", exc)
            return
        ranked = sorted((s for s, (price, _) in stats.items() if price >= self.cfg.min_price),
                        key=lambda s: -stats[s][1])
        self.universe = ranked[: self.cfg.universe_size]
        self.universe_day = today
        log.info("Today's stock list: the %d most-traded stocks and ETFs over $%.0f (top 10: %s)",
                 len(self.universe), self.cfg.min_price, ", ".join(self.universe[:10]))

    def stock_universe(self, now):
        self._build_universe(now.date())
        if self.cfg.scan_stocks and (self.scanned_at is None
                                     or now - self.scanned_at >= timedelta(minutes=self.cfg.rescan_minutes)):
            try:
                found = [sym for sym, price, dollar_vol in self.client.get_most_active_stocks(self.cfg.scan_top)
                         if price >= self.cfg.min_price and dollar_vol >= self.cfg.min_dollar_volume]
                # the day's biggest movers: the "stocks in play" day traders focus on
                found += [sym for sym, price, _ in self.client.get_movers(self.cfg.movers_top)
                          if price >= self.cfg.min_price and sym.isalpha() and sym not in found]
                added = set(found[: self.cfg.max_scanned]) - set(self.scanned)
                self.scanned = found[: self.cfg.max_scanned]
                self.scanned_at = now
                log.info("Stock scan: watching %d most-traded stocks and big movers%s", len(self.scanned),
                         f" (new: {', '.join(sorted(added))})" if added else "")
            except AlpacaError as exc:
                log.error("Stock scan failed, keeping previous list: %s", exc)
        held = [s for s in self.open_trades if not is_crypto(s)]
        held += [t["underlying"] for t in self.open_trades.values() if t.get("underlying")]
        return list(dict.fromkeys(self.cfg.symbols + self.cfg.options_underlyings + self.cfg.world_symbols
                                  + self.universe + self.scanned + held))

    def _asset(self, symbol):
        if symbol not in self.asset_cache:
            try:
                self.asset_cache[symbol] = self.client.get_asset(symbol)
            except AlpacaError:
                return {}
        return self.asset_cache[symbol]

    def _shortable(self, symbol):
        asset = self._asset(symbol)
        return bool(asset.get("shortable") and asset.get("easy_to_borrow"))

    def _set_limits(self, account):
        limits = account_limits(float(account["equity"]), self.cfg)
        if self.limits != limits:
            log.info("Account $%.2f -> %s", float(account["equity"]), limits.describe())
        self.limits = limits

    def _allowed(self, symbol):
        """Which strategies a symbol may use: bearish bets need a big enough account and either
        options (puts) or a borrowable stock; crypto is long-only. Overnight holds are long-only too:
        the research behind them is about prices rising overnight."""
        if is_crypto(symbol) or not self.limits.shorts:
            return list(LONG_STRATEGIES)
        if (symbol in self.cfg.options_underlyings and self.limits.options) or self._shortable(symbol):
            return [n for n in STRATEGIES if n not in NEVER_CHOSEN]
        return list(LONG_STRATEGIES)

    def _day_trades_left(self, account, today):
        """Below pdt_equity US rules allow only a few day trades per 5 days. Every stock or option
        opened today will become a day trade when it closes, so count those as already spent."""
        if not self.limits.pdt_limited:
            return math.inf
        opened_today = sum(1 for sym, t in self.open_trades.items()
                           if not is_crypto(sym) and not _overnight(t.get("strategy"))
                           and t.get("opened_at", "")[:10] == today.isoformat())
        return self.cfg.pdt_max_day_trades - int(account.get("daytrade_count") or 0) - opened_today

    @staticmethod
    def _is_entry_order(order):
        return order.get("client_order_id", "").split("-")[0] in STRATEGIES

    def _timeframe(self, symbol):
        return self.cfg.crypto_timeframe if is_crypto(symbol) else self.cfg.timeframe

    def _get_bars(self, symbol, now):
        days = self.cfg.crypto_learn_days if is_crypto(symbol) else self.cfg.learn_days
        timeframe = self._timeframe(symbol)
        bars = completed_bars(self.client.get_bars(symbol, timeframe, days), timeframe, now)
        return bars if is_crypto(symbol) else annotate_sessions(bars)

    def _due(self, symbol, now):
        """Only fetch bars when a new bar should have closed since the last one we saw."""
        last = self.last_bar_seen.get(symbol)
        minutes = _timeframe_minutes(self._timeframe(symbol))
        return last is None or _parse_ts(last) + timedelta(minutes=2 * minutes) <= now

    # --- main loop ---
    def run_once(self):
        clock = self.client.get_clock()
        now = _parse_ts(clock["timestamp"])
        stock_open = self.stock_open = clock["is_open"]
        positions = {norm(p["symbol"]): p for p in self.client.get_positions() or []}
        entry_orders = [o for o in self.client.get_open_orders() or [] if self._is_entry_order(o)]
        pending = {norm(o["symbol"]) for o in entry_orders}
        self._learn_from_closed_trades(positions, pending)
        self._protect_crypto(positions)

        account = self.client.get_account()
        minutes_to_close = (_parse_ts(clock["next_close"]) - now).total_seconds() / 60
        log.info("Equity $%.2f (prev close $%.2f) | stocks %s | %d positions",
                 float(account["equity"]), float(account["last_equity"]),
                 f"open, {minutes_to_close:.0f} min to close" if stock_open else f"closed until {clock['next_open']}",
                 len(positions))
        self._set_limits(account)

        today = now.date()
        self._update_bench(now)
        if self.halted_day == today:
            return
        if daily_loss_hit(account, self.cfg):
            log.warning("Daily loss limit hit, selling everything and pausing for the rest of the day")
            self._flatten(positions, "daily loss limit", stocks_only=False)
            self.halted_day = today
            return

        self._check_managed_exits(positions)
        self._close_disabled_crypto(positions)
        if stock_open:
            self._manage_open_trades(positions, now)
        else:
            self._protect_overnight_holds(positions)

        symbols = list(self.cfg.crypto_symbols)
        if stock_open:
            if minutes_to_close <= self.cfg.flatten_minutes:
                for order in entry_orders:
                    overnight = _overnight(order.get("client_order_id", "").split("-")[0])
                    if not is_crypto(order["symbol"]) and not overnight:
                        self._act(f"CANCEL unfilled {order['symbol']} entry (end of day)",
                                  lambda o=order: self.client.cancel_order(o["id"]))
                self._flatten(positions, "end of day", stocks_only=True)
            symbols = self.stock_universe(now) + symbols

        # 1) fetch new bars
        fresh = {}
        stocks = [x for x in symbols if not is_crypto(x)]
        if stocks:
            fresh.update(self._fresh_stock_bars(stocks, now))
        for symbol in symbols:
            if not is_crypto(symbol) or not self._due(symbol, now):
                continue
            try:
                bars = self._get_bars(symbol, now)
            except AlpacaError as exc:
                log.error("%s: %s", symbol, exc)
                continue
            if bars and self.last_bar_seen.get(symbol) != bars[-1]["t"]:
                self.last_bar_seen[symbol] = bars[-1]["t"]
                fresh[symbol] = bars

        # 2) re-learn: replay strategies for the symbols whose replay is oldest (staggered, so each loop stays quick)
        stale = [x for x in fresh if x not in self.replayed_at
                 or now - self.replayed_at[x] >= timedelta(minutes=self.cfg.replay_minutes)]
        holding = {norm(x) for x in positions} | {t.get("underlying") or x for x, t in self.open_trades.items()}
        # stocks we hold first (their exits use what was learned), then the longest-unlearned
        stale.sort(key=lambda x: (x not in holding, self.replayed_at.get(x, datetime.min.replace(tzinfo=now.tzinfo))))
        for symbol in stale[: self.cfg.max_replays_per_loop]:
            self.learner.update(symbol, fresh[symbol], self._allowed(symbol))
            self.replayed_at[symbol] = now

        # 3) manage open trades, then take the best-scoring new signals first
        stock_entries = stock_open and minutes_to_close > self.cfg.no_new_entries_minutes
        candidates = []
        for symbol, bars in fresh.items():
            learned = symbol in self.replayed_at
            if not learned and symbol not in holding:
                continue  # not learned yet, and nothing to sell
            try:
                candidate = self._handle_symbol(symbol, bars, positions, pending)
                if not learned:
                    candidate = None  # manage the exit, but no new trade until it's learned
            except AlpacaError as exc:
                log.error("%s: %s", symbol, exc)
                continue
            # overnight holds are bought in the last minutes before the close, after other entries stop
            late_ok = stock_open and minutes_to_close > 2 and candidate and _overnight(candidate[2])
            if candidate and (is_crypto(symbol) or stock_entries or late_ok):
                candidates.append(candidate)
        candidates.sort(key=lambda c: -c[0])
        overnight_count = sum(1 for t in self.open_trades.values() if _overnight(t.get("strategy")))
        for i, (score, symbol, choice, signal, bars) in enumerate(candidates):
            if _overnight(choice):
                if overnight_count >= self.cfg.overnight_max_positions:
                    continue
                overnight_count += 1
            if len(positions) + len(pending) >= self.limits.max_positions:
                log.info("Max open positions reached; skipped %d weaker signals", len(candidates) - i)
                break
            try:
                self._enter(symbol, choice, signal, bars, account, positions, pending, today)
            except AlpacaError as exc:
                log.error("%s: %s", symbol, exc)

        # 4) outside regular hours: trade the overnight session (and pre/after-hours with paid data)
        if not stock_open:
            try:
                self._run_extended(now, account, positions, pending, entry_orders)
            except AlpacaError as exc:
                log.error("Extended hours: %s", exc)
        self.learner.save()

    def _fresh_stock_bars(self, symbols, now):
        """Keep a rolling window of bars per stock, downloading only what's new. Runs once per new bar."""
        minutes = _timeframe_minutes(self.cfg.timeframe)
        bucket = int((now - timedelta(minutes=1)).timestamp() // (minutes * 60))
        if bucket == self.last_stock_bucket:
            return {}
        self.last_stock_bucket = bucket
        new = [x for x in symbols if x not in self.bars]
        known = [x for x in symbols if x in self.bars]
        try:
            if new:
                for sym, bars in self.client.get_stock_bars(new, self.cfg.timeframe, self.cfg.learn_days).items():
                    self.bars[sym] = bars
            if known:
                # reach back to the oldest "latest bar" we hold, so a pause (sleep, lost wifi) leaves no gaps
                oldest = min((self.bars[x][-1]["t"] for x in known if self.bars[x]), default=None)
                since = (now - _parse_ts(oldest)) if oldest else timedelta(days=self.cfg.learn_days)
                days = min(max(since, timedelta(minutes=30)), timedelta(days=self.cfg.learn_days))
                recent = self.client.get_stock_bars(known, self.cfg.timeframe, days / timedelta(days=1))
                for sym, bars in recent.items():
                    have = self.bars.setdefault(sym, [])
                    have[:] = [b for b in have if not bars or b["t"] < bars[0]["t"]] + bars
        except AlpacaError as exc:
            log.error("Couldn't download prices: %s", exc)
            return {}
        self._update_news(now)
        cutoff = (now - timedelta(days=self.cfg.learn_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        fresh = {}
        for sym in symbols:
            bars = self.bars.get(sym) or []
            while bars and bars[0]["t"] < cutoff:
                bars.pop(0)
            done = completed_bars(bars, self.cfg.timeframe, now)
            if done and self.last_bar_seen.get(sym) != done[-1]["t"]:
                self.last_bar_seen[sym] = done[-1]["t"]
                fresh[sym] = self._mark_news(sym, annotate_sessions(done))
        return fresh

    def _update_bench(self, now):
        """Once a day: bench strategies whose recent real trades lost money overall. Replays can make a
        strategy look good that doesn't hold up live; real results are the better judge. A benched
        strategy gets another chance once its losing trades are older than bench_days."""
        if not self.cfg.bench or self.bench_day == now.date():
            return
        try:
            results = self.client.get_strategy_results(now - timedelta(days=self.cfg.bench_days), set(STRATEGIES))
        except AlpacaError as exc:
            log.error("Couldn't read past results: %s", exc)
            return
        self.bench_day = now.date()
        benched = {name for name, pnl in results.items()
                   if len(pnl) >= self.cfg.bench_min_trades and sum(pnl) < 0}
        summary = ", ".join(f"{n} {len(p)} trades ${sum(p):+.0f}" for n, p in sorted(results.items()))
        log.info("Real results, last %d days: %s", self.cfg.bench_days, summary or "none yet")
        if benched != self.learner.benched:
            log.info("Benched for losing money in real trading: %s", ", ".join(sorted(benched)) or "none")
        self.learner.benched = benched

    def _update_news(self, now):
        """Download news published since the last check (the first time: the whole learning window)."""
        if not self.cfg.news_filter:
            return
        since = (_parse_ts(self.news_until) if self.news_until else now - timedelta(days=self.cfg.learn_days + 1))
        try:
            articles = self.client.get_news(since)
        except AlpacaError as exc:
            log.error("Couldn't download news: %s", exc)
            return
        for created, symbols in articles:
            for sym in symbols:
                times = self.news.setdefault(sym, [])
                if not times or created > times[-1]:
                    times.append(created)
                elif created not in times:
                    bisect.insort(times, created)
            if not self.news_until or created > self.news_until:
                self.news_until = created
        if articles:
            log.info("News: %d new articles", len(articles))

    def _mark_news(self, symbol, bars):
        """Mark each bar with _news: did this stock have news in the news_window_hours before the bar closed?"""
        if not self.cfg.news_filter or self.news_until is None:
            return bars
        times = self.news.get(symbol, [])
        window = timedelta(hours=self.cfg.news_window_hours)
        step = timedelta(minutes=_timeframe_minutes(self.cfg.timeframe))
        for b in bars:
            end = _parse_ts(b["t"]) + step
            lo = (end - window).strftime("%Y-%m-%dT%H:%M:%SZ")
            hi = end.strftime("%Y-%m-%dT%H:%M:%SZ")
            i = bisect.bisect_left(times, lo)
            b["_news"] = i < len(times) and times[i] <= hi
        return bars

    def _handle_symbol(self, symbol, bars, positions, pending):
        """Manage any open trade on this symbol; return an entry candidate
        (score, symbol, strategy, signal, bars) if its strategy says buy, else None."""
        key = norm(symbol)
        choice, scores = self.learner.choose(symbol)
        previous = self.learner.state["active"].get(symbol)
        if choice != previous:
            log.info("%s: AI switched strategy %s -> %s (%s)", symbol, previous or "none", choice or "sit out",
                     ", ".join(f"{k} {v:+.2f}R" for k, v in scores.items()))
        self.learner.state["active"][symbol] = choice

        # exits follow the strategy that opened the trade (shares, or an option on this underlying)
        options_held = [c for c, t in self.open_trades.items() if t.get("underlying") == symbol]
        for held in ([key] if key in positions else []) + [c for c in options_held if c in positions]:
            trade = self.open_trades.get(symbol if held == key else held, {})
            owner = trade.get("strategy") or choice or "trend"
            signal = STRATEGIES[owner](bars, self.cfg)
            log.info("%-9s [%s] holding %s: %s", symbol, owner, held, signal.reason)
            if signal.action == "sell":
                target = symbol if held == key else held
                self._act(f"EXIT {target} ({owner})", lambda: self.client.close_position(target))
                positions.pop(held, None)
        if key in positions or options_held or choice is None or key in pending:
            return None

        signal = STRATEGIES[choice](bars, self.cfg)
        if signal.action != "buy":
            log.debug("%-9s [%s] %s", symbol, choice, signal.reason)
            return None
        log.info("%-9s [%s] %s signal: %s", symbol, choice, "SHORT" if signal.side == "short" else "BUY",
                 signal.reason)
        return (scores[choice], symbol, choice, signal, bars)

    def _enter(self, symbol, choice, signal, bars, account, positions, pending, today):
        key = norm(symbol)
        crypto = is_crypto(symbol)
        overnight = _overnight(choice)
        if not crypto and not overnight and self._day_trades_left(account, today) < 1:
            log.info("%s: skipping, no day trades left this week (small-account rule)", symbol)
            return

        if symbol in self.cfg.options_underlyings and self.limits.options and not overnight:
            if self._enter_option(symbol, choice, signal, bars, account):
                pending.add(key)
                return
            if signal.side == "short" and not self._shortable(symbol):
                return

        equity = float(account["equity"])
        cash = float(account["non_marginable_buying_power"] if crypto else account["buying_power"])
        max_pct = self.limits.max_position_pct
        if overnight:
            max_pct = min(max_pct, self.cfg.overnight_max_position_pct)
        size = dict(side=signal.side, max_pct=max_pct)
        qty = position_size(equity, cash, signal.price, signal.stop, self.cfg, fractional=crypto, **size)
        fractional = False
        if not qty and not crypto and signal.side == "long" and self._asset(symbol).get("fractionable"):
            # a whole share is too expensive for this account: buy a slice instead
            qty = position_size(equity, cash, signal.price, signal.stop, self.cfg, fractional=True, **size)
            fractional = bool(qty)
        if crypto and qty and qty < float(self._asset(symbol).get("min_order_size") or 0):
            log.info("%s: %s is below the coin's minimum order size", symbol, qty)
            return
        if not qty:
            log.info("%s: position size too small", symbol)
            return
        order_side = "buy" if signal.side == "long" else "sell"
        order_id = f"{choice}-{key}-{uuid.uuid4().hex[:8]}"
        self._act(
            f"{'BUY' if order_side == 'buy' else 'SHORT'} {qty} {symbol} @ ~{signal.price:.4g} "
            f"stop {signal.stop:.4g} target {signal.target:.4g} ({choice})",
            lambda: self.client.submit_entry(symbol, qty, order_side, signal.target, signal.stop, order_id,
                                             fractional=fractional, overnight=overnight),
        )
        pending.add(key)
        if not self.dry_run:
            self.open_trades[symbol] = {"strategy": choice, "side": signal.side, "entry": signal.price,
                                        "stop": signal.stop, "target": signal.target, "opened_at": bars[-1]["t"],
                                        "risk": abs(signal.price - signal.stop), "managed": crypto or fractional}

    def _enter_option(self, symbol, choice, signal, bars, account):
        """Buy a call (bullish) or put (bearish). Returns False if no good contract was found."""
        kind = "call" if signal.side == "long" else "put"
        today = _parse_ts(bars[-1]["t"]).date()
        try:
            candidates = self.client.get_option_candidates(
                symbol, kind, signal.price, today + timedelta(days=self.cfg.option_min_days),
                today + timedelta(days=self.cfg.option_max_days))
        except AlpacaError as exc:
            log.error("%s options: %s", symbol, exc)
            return False
        contract = pick_option(candidates, self.cfg)
        if not contract:
            log.info("%s: no liquid %s found, trading the shares instead", symbol, kind)
            return False
        cash = float(account.get("options_buying_power") or account["buying_power"])
        qty = option_contracts(float(account["equity"]), cash, contract["ask"], self.cfg)
        if qty < 1:
            log.info("%s: option too expensive for the risk limit", symbol)
            return False
        occ, ask = contract["symbol"], contract["ask"]
        order_id = f"{choice}-{occ}-{uuid.uuid4().hex[:8]}"
        self._act(
            f"BUY {qty} {symbol} {kind.upper()} {occ} @ {ask:.2f} (delta {contract['delta']:+.2f}, "
            f"expires {contract['expiration']}) ({choice})",
            lambda: self.client.submit_option_buy(occ, qty, ask, order_id),
        )
        if not self.dry_run:
            self.open_trades[occ] = {
                "strategy": choice, "side": "long", "underlying": symbol, "entry": ask,
                "stop": ask * (1 - self.cfg.option_stop_pct), "target": ask * (1 + self.cfg.option_take_profit_pct),
                "opened_at": bars[-1]["t"], "managed": True}
        return True

    def _protect_crypto(self, positions):
        """Once a crypto buy fills, park a stop-loss order at Alpaca so the position is protected
        even if the bot goes offline. (Options can't have stop orders, so those stay bot-managed.)"""
        for symbol, trade in self.open_trades.items():
            pos = positions.get(norm(symbol))
            if not is_crypto(symbol) or not pos or trade.get("stop_order_id") or self.dry_run:
                continue
            try:
                order = self.client.submit_crypto_stop(symbol, pos["qty"], trade["stop"])
                trade["stop_order_id"] = order["id"]
                log.info("%s: stop-loss parked at Alpaca at %.4g", symbol, trade["stop"])
            except AlpacaError as exc:
                log.error("%s: could not place stop-loss: %s", symbol, exc)

    def _close_disabled_crypto(self, positions):
        """Sell any crypto in the account that isn't on the bot's crypto list (crypto is off by default)."""
        wanted = {norm(x) for x in self.cfg.crypto_symbols}
        for key, pos in list(positions.items()):
            if pos.get("asset_class") == "crypto" and key not in wanted:
                self._act(f"SELL {pos['symbol']} (crypto trading is turned off)",
                          lambda k=key: self.client.close_position(k))
                positions.pop(key)

    def _manage_open_trades(self, positions, now):
        """Every loop during market hours: trail each stock trade's stop once it's ahead, and close
        trades that have run past max_hold_minutes (except orb, intraday_momentum and overnight holds)."""
        for symbol, trade in list(self.open_trades.items()):
            pos = positions.get(norm(symbol))
            if not pos or trade.get("underlying") or is_crypto(symbol):
                continue
            price = float(pos["current_price"])
            entry, risk = trade["entry"], trade.get("risk") or abs(trade["entry"] - trade["stop"])
            short = trade.get("side") == "short"
            if short:  # mirror the long rules
                trade["best"] = min(trade.get("best", entry), price)
                flipped = -trail_stop(-trade["stop"], -entry, risk, -trade["best"], self.cfg)
                trade["stop"], hit = flipped, price >= flipped
            else:
                trade["best"] = max(trade.get("best", entry), price)
                trade["stop"] = trail_stop(trade["stop"], entry, risk, trade["best"], self.cfg)
                hit = price <= trade["stop"]
            fn = STRATEGIES.get(trade.get("strategy"))
            timed = (self.cfg.max_hold_minutes and not getattr(fn, "hold_to_close", False)
                     and not getattr(fn, "overnight", False) and trade.get("opened_at", "").endswith("Z")
                     and now - _parse_ts(trade["opened_at"]) >= timedelta(minutes=self.cfg.max_hold_minutes))
            # an overnight hold is always sold the next morning, even if its exit bar was missed
            morning_after = False
            if getattr(fn, "overnight", False) and trade.get("opened_at", "").endswith("Z"):
                opened_day, _ = _ny_day_and_minute(trade["opened_at"])
                today, minute = _ny_day_and_minute(now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
                morning_after = opened_day < today and minute >= OPEN_MIN + self.cfg.overnight_exit_after_minutes
            if hit or timed or morning_after:
                why = ("trailing stop" if hit else "overnight hold: next morning" if morning_after
                       else f"held {self.cfg.max_hold_minutes} minutes")
                try:
                    self._act(f"EXIT {symbol} @ ~{price:.4g} ({why})", lambda s=symbol: self.client.close_position(s))
                    positions.pop(norm(symbol))
                except AlpacaError as exc:
                    log.error("%s: %s", symbol, exc)

    def _protect_overnight_holds(self, positions):
        """Outside market hours stop orders don't work, so if an overnight hold falls through its stop in
        pre-market, after-hours or the overnight session, sell it with an extended-hours limit order."""
        for symbol, trade in self.open_trades.items():
            pos = positions.get(norm(symbol))
            if not pos or not _overnight(trade.get("strategy")) or trade.get("protective_order_id"):
                continue
            price = float(pos["current_price"])
            short = trade.get("side") == "short"
            if (price >= trade["stop"]) if short else (price <= trade["stop"]):
                limit = price * (1.005 if short else 0.995)  # a little past the last price so it fills
                try:
                    self._act(f"{'COVER' if short else 'SELL'} {symbol} @ limit {limit:.2f} "
                              f"(fell through its stop outside market hours)",
                              lambda s=symbol, q=pos["qty"], l=limit: trade.update(
                                  protective_order_id=self.client.submit_extended_exit(s, q, "buy" if short else "sell", l)["id"]))
                except AlpacaError as exc:
                    log.error("%s: %s", symbol, exc)

    def _check_managed_exits(self, positions):
        """Crypto and options have no bracket orders, so enforce their stop-loss and take-profit here every loop."""
        for symbol, trade in self.open_trades.items():
            pos = positions.get(norm(symbol))
            if not trade.get("managed", is_crypto(symbol)) or not pos:
                continue
            if trade.get("session") and not self.stock_open:
                continue  # extended-hours trades are watched by _run_extended with limit orders
            price = float(pos["current_price"])
            if price <= trade["stop"] or price >= trade.get("target", math.inf):
                why = "stop-loss" if price <= trade["stop"] else "take-profit"
                try:
                    self._act(f"SELL {symbol} @ ~{price:.4g} ({why})", lambda: self.client.close_position(symbol))
                    positions.pop(norm(symbol))
                except AlpacaError as exc:
                    log.error("%s: %s", symbol, exc)

    def _learn_from_closed_trades(self, positions, pending):
        """Record the result of every trade that has closed since the last check."""
        for symbol, trade in list(self.open_trades.items()):
            key = norm(symbol)
            if key in positions:
                trade["entry"] = float(positions[key]["avg_entry_price"])
                continue
            if key in pending:
                continue
            short = trade.get("side") == "short"
            # None if the entry never filled
            exit_price = self.client.get_last_exit_fill(symbol, trade["opened_at"], "buy" if short else "sell")
            if exit_price is not None:
                risk = trade.get("risk") or abs(trade["entry"] - trade["stop"])
                r = ((trade["entry"] - exit_price) if short else (exit_price - trade["entry"])) / risk
                log.info("%s: trade closed at %.4g, %+.2fR -> learning for %s", symbol, exit_price, r, trade["strategy"])
                self.learner.record_trade(trade.get("learn_key") or trade.get("underlying", symbol),
                                          trade["strategy"], r)
            del self.open_trades[symbol]
            self.learner.save()

    def _flatten(self, positions, reason, stocks_only):
        """stocks_only (end of day): sell every stock and option except overnight holds; else sell everything."""
        def kept(p):
            if not stocks_only:
                return False
            trade = self.open_trades.get(p["symbol"]) or {}
            return p.get("asset_class") == "crypto" or _overnight(trade.get("strategy"))
        targets = [p["symbol"] for p in positions.values() if not kept(p)]
        if not targets:
            return
        if not self.stock_open:  # market orders don't work outside regular hours: sell with limits
            for sym in targets:
                pos = positions[norm(sym)]
                if pos.get("asset_class") == "crypto":
                    self._act(f"SELL {sym} ({reason})", lambda s=sym: self.client.close_position(s))
                else:
                    short = float(pos["qty"]) < 0
                    price = float(pos["current_price"]) * (1.005 if short else 0.995)
                    self._act(f"{'COVER' if short else 'SELL'} {sym} @ limit {price:.2f} ({reason})",
                              lambda s=sym, q=pos["qty"], p=price, side="buy" if short else "sell":
                              self.client.submit_extended_exit(s, q, side, p))
                positions.pop(norm(sym))
        elif stocks_only:
            for sym in targets:
                self._act(f"SELL {sym} ({reason})", lambda s=sym: self.client.close_position(s))
                positions.pop(norm(sym))
        else:
            self._act(f"CLOSE ALL {len(targets)} positions ({reason})", self.client.close_all_positions)
            positions.clear()

    def _act(self, description, fn):
        if self.dry_run:
            log.info("[dry-run] would %s", description)
            return
        log.info("ORDER: %s", description)
        fn()

    def report(self):
        """Replay every strategy on current data and print what the bot would pick."""
        now = _parse_ts(self.client.get_clock()["timestamp"])
        self._set_limits(self.client.get_account())
        symbols = self.stock_universe(now) + list(self.cfg.crypto_symbols)
        for symbol in symbols:
            self.learner.update(symbol, self._get_bars(symbol, now), self._allowed(symbol))
        print(f"{'symbol':9} {'strategy':15} {'score':>7} {'replay trades':>13} {'replay R':>9} "
              f"{'real trades':>11} {'real R':>7}")
        using = []
        for symbol in symbols:
            choice, _ = self.learner.choose(symbol)
            for name, score, n, total, live_n, live_total in self.learner.scoreboard(symbol):
                mark = "  <- using" if name == choice else ""
                print(f"{symbol:9} {name:15} {score:+7.2f} {n:13d} {total:+9.2f} {live_n:11d} {live_total:+7.2f}{mark}")
            if choice is None:
                print(f"{symbol:9} (sitting out: no strategy is working right now)")
            else:
                using.append(f"{symbol} ({choice})")
            print()
        print(f"Trading {len(using)} of {len(symbols)}: {', '.join(using) or 'none right now'}")

    def run_forever(self):
        log.info("Trading %d stocks + %d options underlyings + %d world%s + %d crypto, shorts %s, on %s (%s)",
                 len(self.cfg.symbols), len(self.cfg.options_underlyings), len(self.cfg.world_symbols),
                 " + daily most-traded scan" if self.cfg.scan_stocks else "", len(self.cfg.crypto_symbols),
                 "on" if self.cfg.allow_shorts else "off",
                 self.cfg.base_url, "DRY RUN" if self.dry_run else "paper orders")
        while True:
            try:
                self.run_once()
            except (AlpacaError, OSError) as exc:
                log.error("Loop error: %s", exc)
            except Exception:  # never let one bad response stop the bot
                log.exception("Unexpected error, carrying on")
            time.sleep(self.cfg.poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--once", action="store_true", help="run a single pass and exit")
    parser.add_argument("--dry-run", action="store_true", help="log decisions without placing orders")
    parser.add_argument("--report", action="store_true", help="show which strategy each market uses and why")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    bot = TradingBot(Config.from_env(), dry_run=args.dry_run)
    if args.report:
        bot.report()
    elif args.once:
        bot.run_once()
    else:
        try:
            bot.run_forever()
        except KeyboardInterrupt:
            log.info("Stopped. Stock trades keep their stop-loss/take-profit orders; "
                     "crypto and option stops are only watched while the bot runs.")


if __name__ == "__main__":
    main()
