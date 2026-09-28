"""The bot's learning brain: decides which strategy each symbol should trade.

How it learns:
  1. Replay. On every new bar it replays every strategy over the last
     `learn_days` of that symbol's bars. It counts each simulated trade's
     result in R (profit divided by the amount risked; the stop-loss = -1R).
  2. Experience. Every real trade the bot closes is saved to the state file with
     its R. Real trades count `live_weight` times as much as replayed ones.
  3. Pick. A strategy's score is its average R per trade. It is pulled toward
     that strategy's average on the other symbols (stocks and crypto learn
     separately) until the symbol has enough trades of its own
     (`prior_strength`), so a few lucky trades don't win. Each symbol uses its
     highest-scoring strategy, or sits out when nothing scores above `min_score`.
"""
import json
import os

from .strategy import STRATEGIES, mirror

WARMUP_BARS = 30   # bars needed before a strategy can signal
WINDOW = 120       # bars handed to a strategy per step (bounds replay cost)


def _day(bar):
    return bar["t"][:10]


def _timeframe_minutes(timeframe):
    return int(timeframe[:-3]) if timeframe.endswith("Min") else int(timeframe[:-4]) * 60


def trail_stop(stop, entry, risk, best, cfg):
    """Once a trade is up by trail_start_r times its risk, drag the stop up behind the best price
    (trail_r times the risk below it), so a winner can't turn into a loser."""
    if cfg.trail_start_r and best - entry >= cfg.trail_start_r * risk:
        return max(stop, best - cfg.trail_r * risk)
    return stop


def backtest(strategy, bars, cfg, eod_exit=True, cost_pct=None):
    """Replay `strategy` over `bars` and return the R-multiple of each closed trade.

    Entries fill at the signal bar's close. Stops and targets are checked from the next bar;
    if both are touched in one bar the stop is assumed (pessimistic). The stop trails the best
    price once the trade is ahead (see trail_stop), and with eod_exit (stocks) trades are closed
    after max_hold_minutes and at the last bar of each day, like the live bot. Strategies marked
    `overnight` are held into the next session and those marked `hold_to_close` skip the time limit.
    """
    cost_pct = cfg.cost_pct if cost_pct is None else cost_pct
    overnight = getattr(strategy, "overnight", False)
    max_bars = 0
    if eod_exit and cfg.max_hold_minutes and not overnight and not getattr(strategy, "hold_to_close", False):
        max_bars = cfg.max_hold_minutes // _timeframe_minutes(cfg.timeframe)
    results, pos = [], None
    for i in range(WARMUP_BARS, len(bars)):
        bar = bars[i]
        new_day_next = i + 1 < len(bars) and _day(bars[i + 1]) != _day(bar)
        last_of_day = eod_exit and not overnight and new_day_next

        if pos:
            pos["held"] += 1
            exit_price = None
            if bar["l"] <= pos["stop"]:
                exit_price = min(bar["o"], pos["stop"])
            elif bar["h"] >= pos["target"]:
                exit_price = max(bar["o"], pos["target"])
            elif (last_of_day or (max_bars and pos["held"] >= max_bars)
                  or strategy(bars[max(0, i - WINDOW):i + 1], cfg).action == "sell"):
                exit_price = bar["c"]
            if exit_price is not None:
                cost = pos["entry"] * cost_pct
                results.append((exit_price - pos["entry"] - cost) / pos["risk"])
                pos = None
            else:
                pos["best"] = max(pos["best"], bar["h"])
                pos["stop"] = trail_stop(pos["stop"], pos["entry"], pos["risk"], pos["best"], cfg)
            continue

        if last_of_day or i + 1 == len(bars):
            continue
        sig = strategy(bars[max(0, i - WINDOW):i + 1], cfg)
        if sig.action == "buy" and sig.stop < sig.price:
            pos = {"entry": sig.price, "stop": sig.stop, "target": sig.target,
                   "risk": sig.price - sig.stop, "best": sig.price, "held": 0}
    return results


class Learner:
    def __init__(self, cfg, strategies=STRATEGIES, backtester=backtest):
        self.cfg = cfg
        self.strategies = strategies
        self.backtester = backtester
        self.state = self._load()
        self.replay = {}   # symbol -> {strategy: [R, ...]} from the latest replay
        self.allowed = {}  # symbol -> strategy names it may use (e.g. no shorts for crypto)
        self._group_cache = {}  # (strategy, group) -> totals over every symbol; cleared when data changes

    # --- persistence ---
    def _load(self):
        state = {"active": {}, "open_trades": {}, "live_results": {}}
        if os.path.exists(self.cfg.state_file):
            with open(self.cfg.state_file) as fh:
                state.update(json.load(fh))
        return state

    def save(self):
        tmp = self.cfg.state_file + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.state, fh, indent=2)
        os.replace(tmp, self.cfg.state_file)

    # --- learning ---
    def record_trade(self, symbol, strategy, r_multiple):
        by_symbol = self.state["live_results"].setdefault(strategy, {})
        by_symbol.setdefault(symbol, []).append(round(r_multiple, 4))
        self._group_cache.clear()
        self.save()

    def _totals(self, strategy, symbols):
        """Weighted (sum of R, number of trades) from replay + live results."""
        total, n = 0.0, 0.0
        for sym in symbols:
            replayed = self.replay.get(sym, {}).get(strategy, [])
            live = self.state["live_results"].get(strategy, {}).get(sym, [])
            total += sum(replayed) + self.cfg.live_weight * sum(live)
            n += len(replayed) + self.cfg.live_weight * len(live)
        return total, n

    def score(self, strategy, symbol):
        k = self.cfg.prior_strength
        group = self._group(symbol)
        if (strategy, group) not in self._group_cache:
            members = {s for s in set(self.replay) | set(self.state["live_results"].get(strategy, {}))
                       if self._group(s) == group}
            self._group_cache[(strategy, group)] = self._totals(strategy, members)
        g_total, g_n = self._group_cache[(strategy, group)]
        s_total, s_n = self._totals(strategy, [symbol])
        o_total, o_n = g_total - s_total, g_n - s_n  # everyone else in the group
        others_mean = o_total / (o_n + k)  # shrunk toward 0 when there's little data anywhere
        return (s_total + k * others_mean) / (s_n + k)

    def _group(self, symbol):
        """Symbols only learn from others in their group. "AAPL@overnight" is Apple in the overnight
        session, which behaves differently from the daytime market."""
        if "@" in symbol:
            return symbol.split("@", 1)[1]
        return "crypto" if "/" in symbol else "stock"

    def update(self, symbol, bars, allowed=None):
        """Replay the allowed strategies on the latest bars for `symbol`.

        Short strategies are replayed as their long twin on the flipped chart.
        """
        group = self._group(symbol)
        crypto = group == "crypto"
        cost = {"crypto": self.cfg.crypto_cost_pct, "stock": self.cfg.cost_pct}.get(group, self.cfg.ext_cost_pct)
        kwargs = {"eod_exit": not crypto, "cost_pct": cost}
        names = [n for n in self.strategies if allowed is None or n in allowed]
        self.allowed[symbol] = names
        self._group_cache.clear()
        flipped = mirror(bars) if bars and any(hasattr(self.strategies[n], "base") for n in names) else bars
        self.replay[symbol] = {}
        for name in names:
            fn = self.strategies[name]
            base = getattr(fn, "base", None)
            self.replay[symbol][name] = (self.backtester(base, flipped, self.cfg, **kwargs) if base
                                         else self.backtester(fn, bars, self.cfg, **kwargs))

    def choose(self, symbol):
        """Return (best strategy name or None to sit out, {name: score})."""
        scores = {name: self.score(name, symbol) for name in self.allowed.get(symbol, self.strategies)}
        best = max(scores, key=scores.get)
        choice = best if scores[best] > self.cfg.min_score else None
        return choice, scores

    def scoreboard(self, symbol):
        rows = []
        for name in self.allowed.get(symbol, self.strategies):
            replayed = self.replay.get(symbol, {}).get(name, [])
            live = self.state["live_results"].get(name, {}).get(symbol, [])
            rows.append((name, self.score(name, symbol), len(replayed),
                         sum(replayed), len(live), sum(live)))
        return rows
