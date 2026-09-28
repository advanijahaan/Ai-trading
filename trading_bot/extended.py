"""Trading outside regular market hours.

Alpaca trades US stocks 24 hours a day, Sunday 8 PM to Friday 8 PM New York time, in four sessions:
overnight (8 PM - 4 AM), pre-market (4 - 9:30 AM), regular (9:30 AM - 4 PM) and after-hours (4 - 8 PM).
Outside regular hours only limit orders work and stop orders don't, so here the bot:
  * trades the most-traded stocks that are allowed in the session, buying only, with limit orders;
  * builds live 5-minute bars from real-time quotes, on top of the session's price history;
  * learns each stock's session separately ("AAPL@overnight"), since it behaves differently;
  * watches stops, targets, trailing stops and the time limit itself, selling with limit orders;
  * sells everything shortly before the session ends.
The overnight session runs on free data (quotes: "overnight" feed, history: "boats"). Pre-market and
after-hours need live extended-hours prices, which on Alpaca means the paid SIP feed (extended_feed).
"""
import logging
import uuid
from datetime import datetime, time, timedelta, timezone

from .alpaca_client import AlpacaError, is_crypto, norm
from .learner import trail_stop
from .risk import position_size
from .strategy import NEW_YORK, STRATEGIES, _ny_day_and_minute

log = logging.getLogger("trading_bot")

EXTENDED_STRATEGIES = ["trend", "mean_reversion", "breakout"]  # the others need the regular session
SESSION_MINUTES = {"overnight": (20 * 60, 4 * 60), "pre": (4 * 60, 9 * 60 + 30), "after": (16 * 60, 20 * 60)}


def extended_session(now):
    """(name, end time) of the extended session running at `now`, or (None, None)."""
    ny = now.astimezone(NEW_YORK) if NEW_YORK else now
    wd, m = ny.weekday(), ny.hour * 60 + ny.minute  # Monday = 0

    def at(day, hour, minute=0):
        return datetime.combine(day, time(hour, minute), tzinfo=ny.tzinfo)

    if m >= 20 * 60 and wd in (6, 0, 1, 2, 3):  # Sunday-Thursday evening
        return "overnight", at(ny.date() + timedelta(days=1), 4)
    if m < 4 * 60 and wd in (0, 1, 2, 3, 4):     # Monday-Friday early morning
        return "overnight", at(ny.date(), 4)
    if wd < 5 and 4 * 60 <= m < 9 * 60 + 30:
        return "pre", at(ny.date(), 9, 30)
    if wd < 5 and 16 * 60 <= m < 20 * 60:
        return "after", at(ny.date(), 20)
    return None, None


def in_session(minute, name):
    start, end = SESSION_MINUTES[name]
    return start <= minute < end if start < end else minute >= start or minute < end


class ExtendedHoursMixin:
    """Mixed into TradingBot; uses its client, learner, cfg, limits and open_trades."""

    def _init_extended(self):
        self.ext_bars = {}         # (session, symbol) -> bars
        self.ext_live = {}         # (session, symbol) -> the 5-minute bar being built from quotes
        self.ext_seen = {}         # (session, symbol) -> time of the last bar acted on
        self.ext_replayed = {}     # learner key -> when last replayed
        self.overnight_ok = set()
        self.overnight_ok_day = None

    def _ext_feeds(self, name):
        """(quote feed, history feed) for a session, or (None, None) if we can't trade it."""
        if not self.cfg.extended_trading:
            return None, None
        if name == "overnight":
            return "overnight", "boats"
        return (self.cfg.extended_feed or None,) * 2

    def _ext_universe(self, name, now):
        self._build_universe(now.date())
        symbols = [x for x in self.universe if not is_crypto(x)]
        if name == "overnight":
            if self.overnight_ok_day != now.date():
                try:
                    self.overnight_ok = self.client.get_overnight_tradable()
                    self.overnight_ok_day = now.date()
                except AlpacaError as exc:
                    log.error("Couldn't load overnight-tradable stocks: %s", exc)
            symbols = [x for x in symbols if x in self.overnight_ok]
        return symbols[: self.cfg.ext_universe_size]

    def _ext_history(self, name, symbols, feed):
        missing = [x for x in symbols if (name, x) not in self.ext_bars]
        if not missing:
            return
        try:
            history = self.client.get_stock_bars(missing, self.cfg.timeframe, self.cfg.learn_days, feed=feed)
        except AlpacaError as exc:
            log.error("Couldn't download %s price history: %s", name, exc)
            return
        for sym in missing:
            bars = []
            for b in history.get(sym, []):
                b["_day"], b["_min"] = _ny_day_and_minute(b["t"])
                if in_session(b["_min"], name):
                    bars.append(b)
            self.ext_bars[(name, sym)] = bars

    def _ext_add_quote(self, name, sym, mid, now):
        """Fold a live quote into the 5-minute bar being built; return True when a bar just closed."""
        minutes = 5
        now = now.astimezone(timezone.utc)
        start = now.replace(second=0, microsecond=0, minute=now.minute - now.minute % minutes)
        t = start.strftime("%Y-%m-%dT%H:%M:%SZ")
        live = self.ext_live.get((name, sym))
        closed = False
        if live and live["t"] != t:
            bars = self.ext_bars.setdefault((name, sym), [])
            if not bars or live["t"] > bars[-1]["t"]:
                bars.append(live)
                closed = True
            live = None
        if live is None:
            live = {"t": t, "o": mid, "h": mid, "l": mid, "c": mid, "v": 0}
            live["_day"], live["_min"] = _ny_day_and_minute(t)
            self.ext_live[(name, sym)] = live
        live["h"], live["l"], live["c"] = max(live["h"], mid), min(live["l"], mid), mid
        return closed

    def _run_extended(self, now, account, positions, pending, entry_orders):
        name, end = extended_session(now)
        quote_feed, history_feed = self._ext_feeds(name) if name else (None, None)
        if not quote_feed:
            return
        minutes_left = (end - now).total_seconds() / 60
        trades = {s: t for s, t in self.open_trades.items() if t.get("session") == name}
        held = [s for s in trades if norm(s) in positions]

        if minutes_left <= self.cfg.ext_flatten_minutes:
            for order in entry_orders:
                if order["symbol"] in trades and norm(order["symbol"]) not in positions:
                    self._act(f"CANCEL unfilled {order['symbol']} {name} entry (session ending)",
                              lambda o=order: self.client.cancel_order(o["id"]))
            quotes = self.client.get_latest_quotes(held, quote_feed) if held else {}
            for sym in held:
                self._ext_exit(sym, trades[sym], positions, quotes.get(sym), f"{name} session ending")
            return

        symbols = self._ext_universe(name, now)
        try:
            quotes = self.client.get_latest_quotes(list(dict.fromkeys(symbols + held)), quote_feed)
        except AlpacaError as exc:
            log.error("Couldn't get %s quotes: %s", name, exc)
            return
        self._ext_history(name, symbols, history_feed)

        # manage open trades on live prices
        for sym in held:
            quote = quotes.get(sym)
            trade = trades[sym]
            if not quote or trade.get("exit_order_id"):
                continue
            mid = sum(quote) / 2
            trade["best"] = max(trade.get("best", trade["entry"]), mid)
            trade["stop"] = trail_stop(trade["stop"], trade["entry"], trade["risk"], trade["best"], self.cfg)
            too_long = (self.cfg.max_hold_minutes and trade.get("opened_at", "").endswith("Z") and
                        now - datetime.fromisoformat(trade["opened_at"].replace("Z", "+00:00"))
                        >= timedelta(minutes=self.cfg.max_hold_minutes))
            if mid <= trade["stop"] or mid >= trade["target"] or too_long:
                why = ("stop" if mid <= trade["stop"] else "target" if mid >= trade["target"]
                       else f"held {self.cfg.max_hold_minutes} minutes")
                self._ext_exit(sym, trade, positions, quote, why)

        # build live bars and look for new signals
        fresh = {}
        for sym in symbols:
            quote = quotes.get(sym)
            if quote and self._ext_add_quote(name, sym, sum(quote) / 2, now):
                bars = self.ext_bars[(name, sym)]
                if self.ext_seen.get((name, sym)) != bars[-1]["t"]:
                    self.ext_seen[(name, sym)] = bars[-1]["t"]
                    fresh[sym] = bars
        stale = [x for x in fresh if f"{x}@{name}" not in self.ext_replayed or
                 now - self.ext_replayed[f"{x}@{name}"] >= timedelta(minutes=self.cfg.replay_minutes)]
        for sym in stale[: self.cfg.max_replays_per_loop]:
            self.learner.update(f"{sym}@{name}", fresh[sym], EXTENDED_STRATEGIES)
            self.ext_replayed[f"{sym}@{name}"] = now
        if minutes_left <= self.cfg.ext_no_entry_minutes:
            return

        candidates = []
        for sym, bars in fresh.items():
            key = f"{sym}@{name}"
            if key not in self.ext_replayed or norm(sym) in positions or norm(sym) in pending or sym not in quotes:
                continue
            choice, scores = self.learner.choose(key)
            if choice is None:
                continue
            signal = STRATEGIES[choice](bars, self.cfg)
            if signal.action == "buy" and signal.side == "long":
                log.info("%-9s [%s, %s] BUY signal: %s", sym, choice, name, signal.reason)
                candidates.append((scores[choice], sym, choice, signal, bars))
        for score, sym, choice, signal, bars in sorted(candidates, key=lambda c: -c[0]):
            if len(positions) + len(pending) >= self.limits.max_positions:
                break
            self._ext_enter(name, sym, choice, signal, bars, quotes[sym], account, pending, now)

    def _ext_enter(self, name, sym, choice, signal, bars, quote, account, pending, now):
        if self._day_trades_left(account, now.date()) < 1:
            log.info("%s: skipping, no day trades left this week (small-account rule)", sym)
            return
        ask = quote[1]
        stop, target = signal.stop - (signal.price - ask), signal.target - (signal.price - ask)
        if stop >= ask:
            return
        equity = float(account["equity"])
        cash = float(account.get("non_marginable_buying_power") or account["buying_power"])
        size = dict(max_pct=self.limits.max_position_pct)
        qty = position_size(equity, cash, ask, stop, self.cfg, **size) or position_size(
            equity, cash, ask, stop, self.cfg, fractional=True, **size)
        if not qty:
            return
        limit = ask * (1 + self.cfg.ext_limit_pct)
        order_id = f"{choice}-{norm(sym)}-{uuid.uuid4().hex[:8]}"
        self._act(f"BUY {qty} {sym} @ limit {limit:.2f} stop {stop:.2f} target {target:.2f} ({choice}, {name})",
                  lambda: self.client.submit_extended_entry(sym, qty, limit, order_id))
        pending.add(norm(sym))
        if not self.dry_run:
            self.open_trades[sym] = {
                "strategy": choice, "side": "long", "entry": ask, "stop": stop, "target": target,
                "risk": ask - stop, "opened_at": now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "managed": True,
                "session": name, "learn_key": f"{sym}@{name}"}

    def _ext_exit(self, sym, trade, positions, quote, why):
        pos = positions.get(norm(sym))
        if not pos or trade.get("exit_order_id"):
            return
        bid = quote[0] if quote else float(pos["current_price"])
        limit = bid * (1 - self.cfg.ext_limit_pct)
        try:
            self._act(f"SELL {sym} @ limit {limit:.2f} ({why})", lambda: trade.update(
                exit_order_id=self.client.submit_extended_exit(sym, pos["qty"], "sell", limit)["id"]))
        except AlpacaError as exc:
            log.error("%s: %s", sym, exc)
