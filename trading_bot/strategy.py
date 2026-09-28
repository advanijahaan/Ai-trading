"""Pure indicator and strategy logic (no network), so it is easy to test.

Each strategy looks at closed bars (oldest first, Alpaca keys o/h/l/c/v) and
returns a Signal for the latest bar:
  * "buy"  - open a position, with an ATR-based stop and target
  * "sell" - exit an open position
  * "hold" - do nothing
Every strategy also has a "_short" twin that bets on prices falling. It runs the
same rules on an upside-down price chart (see mirror). For a short, stop is
above the price and target is below it.
The learner (learner.py) decides which strategy each symbol should use.

The session strategies (orb, vwap_trend, intraday_momentum) come from published research
on US stocks and need each bar marked with its trading-session data (see annotate_sessions).
"""
from dataclasses import dataclass, replace
from datetime import datetime

try:
    from zoneinfo import ZoneInfo
    NEW_YORK = ZoneInfo("America/New_York")
except Exception:  # Windows without the tzdata package
    NEW_YORK = None

OPEN_MIN, CLOSE_MIN = 9 * 60 + 30, 16 * 60  # regular session, minutes after midnight New York time


def _ny_day_and_minute(t):
    dt = datetime.fromisoformat(t.replace("Z", "+00:00"))
    if NEW_YORK is not None:
        dt = dt.astimezone(NEW_YORK)
    else:  # rough fallback: daylight-saving offset
        from datetime import timedelta, timezone
        dt = dt.astimezone(timezone(timedelta(hours=-4)))
    return dt.date().isoformat(), dt.hour * 60 + dt.minute


def annotate_sessions(bars, rel_volume_days=14):
    """Mark each stock bar (in place) with its session data, in one pass:
    _day/_min (New York date and minute of the bar's start), _orh/_orl/_oro/_orc (today's first
    5-minute bar), _rv (that bar's volume relative to the average first bar of recent days),
    _vwap (today's volume-weighted average price so far), _pc (previous session's close) and
    _c10 (today's close at 10:00). Bars outside 9:30-16:00 get _min only. Returns bars."""
    day, first, pv, vol, prev_close, last_close, c10 = None, None, 0.0, 0.0, None, None, None
    first_volumes = []
    for b in bars:
        if "_min" not in b:
            b["_day"], b["_min"] = _ny_day_and_minute(b["t"])
        if not OPEN_MIN <= b["_min"] < CLOSE_MIN:
            b["_orh"] = None
            continue
        if b["_day"] != day:  # a new session starts
            if first is not None:
                first_volumes.append(first["v"])
                prev_close = last_close
            day, first, pv, vol, c10 = b["_day"], b, 0.0, 0.0, None
            recent = first_volumes[-rel_volume_days:]
            b["_rv_today"] = b["v"] / (sum(recent) / len(recent)) if len(recent) >= 3 and sum(recent) else None
        typical = (b["h"] + b["l"] + b["c"]) / 3
        pv, vol = pv + typical * b["v"], vol + b["v"]
        if b["_min"] == 595:  # the 9:55 bar closes at 10:00
            c10 = b["c"]
        b["_orh"], b["_orl"], b["_oro"], b["_orc"] = first["h"], first["l"], first["o"], first["c"]
        b["_rv"], b["_vwap"], b["_pc"], b["_c10"] = first.get("_rv_today"), pv / vol if vol else b["c"], prev_close, c10
        last_close = b["c"]
    return bars


def ema(values, period):
    if len(values) < period:
        return []
    k = 2 / (period + 1)
    out = [sum(values[:period]) / period]
    for v in values[period:]:
        out.append(v * k + out[-1] * (1 - k))
    # pad so out[i] lines up with values[i]
    return [None] * (period - 1) + out


def rsi(values, period=14):
    if len(values) <= period:
        return None
    gains, losses = [], []
    for prev, cur in zip(values, values[1:]):
        change = cur - prev
        gains.append(max(change, 0))
        losses.append(max(-change, 0))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for g, l in zip(gains[period:], losses[period:]):
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
    if avg_loss == 0:
        return 100.0
    return 100 - 100 / (1 + avg_gain / avg_loss)


def atr(highs, lows, closes, period=14):
    if len(closes) <= period:
        return None
    trs = [
        max(h - l, abs(h - pc), abs(l - pc))
        for h, l, pc in zip(highs[1:], lows[1:], closes[:-1])
    ]
    value = sum(trs[:period]) / period
    for tr in trs[period:]:
        value = (value * (period - 1) + tr) / period
    return value


@dataclass
class Signal:
    action: str  # "buy", "sell" or "hold"
    price: float = 0.0
    stop: float = 0.0
    target: float = 0.0
    reason: str = ""
    side: str = "long"


def trend(bars, cfg):
    """Trend following: buy when the fast EMA crosses above the slow EMA."""
    needed = max(cfg.slow_ema, cfg.rsi_period, cfg.atr_period) + 2
    if len(bars) < needed:
        return Signal("hold", reason=f"not enough bars ({len(bars)}/{needed})")

    closes = [b["c"] for b in bars]
    fast = ema(closes, cfg.fast_ema)
    slow = ema(closes, cfg.slow_ema)
    price = closes[-1]
    crossed_up = fast[-2] <= slow[-2] and fast[-1] > slow[-1]
    crossed_down = fast[-2] >= slow[-2] and fast[-1] < slow[-1]

    if crossed_down:
        return Signal("sell", price=price, reason="fast EMA crossed below slow EMA")

    if crossed_up:
        r = rsi(closes, cfg.rsi_period)
        if r is not None and r >= cfg.rsi_max_entry:
            return Signal("hold", price=price, reason=f"crossover but RSI {r:.1f} overbought")
        return _entry(bars, cfg, cfg.stop_atr_mult, cfg.take_profit_atr_mult,
                      f"bullish EMA crossover, RSI {r:.1f}")

    return Signal("hold", price=price, reason="no crossover")


def mean_reversion(bars, cfg):
    """Buy the bounce: RSI climbs back out of oversold. Exit once price recovers to its average."""
    needed = max(cfg.mr_sma, cfg.rsi_period, cfg.atr_period) + 2
    if len(bars) < needed:
        return Signal("hold", reason=f"not enough bars ({len(bars)}/{needed})")

    closes = [b["c"] for b in bars]
    price = closes[-1]
    sma = sum(closes[-cfg.mr_sma:]) / cfg.mr_sma
    r_now, r_prev = rsi(closes, cfg.rsi_period), rsi(closes[:-1], cfg.rsi_period)

    if price >= sma or r_now >= cfg.mr_exit_rsi:
        return Signal("sell", price=price, reason=f"back to average (RSI {r_now:.1f})")
    if r_prev < cfg.mr_oversold <= r_now:
        return _entry(bars, cfg, cfg.stop_atr_mult, cfg.mr_target_atr_mult,
                      f"RSI bounced out of oversold ({r_prev:.1f} -> {r_now:.1f})")
    return Signal("hold", price=price, reason=f"RSI {r_now:.1f}, no bounce")


def breakout(bars, cfg):
    """Buy a fresh break above the recent high on above-average volume."""
    n = cfg.breakout_lookback
    needed = max(n, cfg.atr_period) + 2
    if len(bars) < needed:
        return Signal("hold", reason=f"not enough bars ({len(bars)}/{needed})")

    price, prev_close = bars[-1]["c"], bars[-2]["c"]
    prior = bars[-n - 1:-1]
    high = max(b["h"] for b in prior)
    exit_low = min(b["l"] for b in bars[-cfg.breakout_exit_lookback - 1:-1])
    avg_vol = sum(b["v"] for b in prior) / len(prior)

    if price < exit_low:
        return Signal("sell", price=price, reason=f"broke below {cfg.breakout_exit_lookback}-bar low")
    if price > high >= prev_close and bars[-1]["v"] >= cfg.breakout_volume_mult * avg_vol:
        return _entry(bars, cfg, cfg.breakout_stop_atr_mult, cfg.breakout_target_atr_mult,
                      f"broke {n}-bar high {high:.2f} on {bars[-1]['v'] / max(avg_vol, 1):.1f}x volume")
    return Signal("hold", price=price, reason="no breakout")


def _entry(bars, cfg, stop_mult, target_mult, reason):
    price = bars[-1]["c"]
    vol = atr([b["h"] for b in bars], [b["l"] for b in bars], [b["c"] for b in bars], cfg.atr_period)
    if not vol:
        return Signal("hold", price=price, reason="ATR unavailable")
    return Signal("buy", price=price, stop=price - stop_mult * vol, target=price + target_mult * vol,
                  reason=f"{reason}, ATR {vol:.2f}")


def mirror(bars, k=None):
    """Flip the chart upside down (price -> k/price) so a falling market looks like a rising one.

    With k = last close**2 the last close is unchanged, and prices map back the same way.
    """
    k = k or bars[-1]["c"] ** 2
    flipped = [{"t": b["t"], "o": k / b["o"], "h": k / b["l"], "l": k / b["h"], "c": k / b["c"], "v": b["v"]}
               for b in bars]
    for f, b in zip(flipped, bars):  # news doesn't depend on price direction
        if "_news" in b:
            f["_news"] = b["_news"]
    if bars and "_min" in bars[-1]:  # keep session data (cheap: no timestamp parsing again)
        for f, b in zip(flipped, bars):
            f["_day"], f["_min"] = b["_day"], b["_min"]
        annotate_sessions(flipped)
    return flipped


def short_version(fn):
    def short(bars, cfg):
        k = bars[-1]["c"] ** 2
        sig = fn(mirror(bars, k), cfg)
        if sig.action == "buy":
            sig = replace(sig, stop=k / sig.stop, target=k / sig.target)
        return replace(sig, side="short", price=bars[-1]["c"] if bars else 0.0,
                       reason=f"{sig.reason} (on flipped chart)")
    short.__name__ = fn.__name__ + "_short"
    short.base = fn
    for flag in ("hold_to_close", "overnight"):
        if hasattr(fn, flag):
            setattr(short, flag, getattr(fn, flag))
    return short


LONG_STRATEGIES = {"trend": trend, "mean_reversion": mean_reversion, "breakout": breakout}
def _session(bars):
    """Today's bars so far, if the latest bar is annotated and inside the regular session."""
    b = bars[-1] if bars else {}
    if b.get("_orh") is None:
        return None
    return b


def orb(bars, cfg):
    """Opening range breakout on stocks in play (Zarattini, Barbon & Aziz): the first 5 minutes set
    a range; on a stock trading far above its usual opening volume, buy when price breaks above the
    range after an up opening bar. Stop at the range low; hold until the close."""
    b = _session(bars)
    if b is None:
        return Signal("hold", reason="needs stock session data")
    if b["_min"] == OPEN_MIN or b["_min"] > OPEN_MIN + cfg.orb_entry_window_minutes:
        return Signal("hold", price=b["c"], reason="outside the opening-range window")
    if b["_rv"] is None or b["_rv"] < cfg.orb_min_rel_volume:
        return Signal("hold", price=b["c"], reason="not in play (normal opening volume)")
    prev = bars[-2] if len(bars) > 1 else b
    same_day = prev.get("_day") == b["_day"]
    if b["_orc"] > b["_oro"] and b["c"] > b["_orh"] and (not same_day or prev["c"] <= b["_orh"]):
        risk = b["c"] - b["_orl"]
        if risk <= 0:
            return Signal("hold", price=b["c"], reason="no room for a stop")
        return Signal("buy", price=b["c"], stop=b["_orl"], target=b["c"] + cfg.orb_target_r * risk,
                      reason=f"broke the opening range high {b['_orh']:.2f}, opening volume {b['_rv']:.1f}x normal")
    return Signal("hold", price=b["c"], reason="no opening-range breakout")


def vwap_trend(bars, cfg):
    """VWAP trend (Zarattini & Aziz): buy when price crosses above today's volume-weighted average
    price, get out when it falls back below."""
    b = _session(bars)
    if b is None:
        return Signal("hold", reason="needs stock session data")
    price, vwap = b["c"], b["_vwap"]
    if price < vwap:
        return Signal("sell", price=price, reason=f"below VWAP {vwap:.2f}")
    prev = bars[-2] if len(bars) > 1 else b
    if (b["_min"] >= OPEN_MIN + cfg.vwap_skip_minutes and prev.get("_day") == b["_day"]
            and prev["c"] <= prev.get("_vwap", prev["c"])):
        return _entry(bars, cfg, cfg.stop_atr_mult, cfg.vwap_target_atr_mult, f"crossed above VWAP {vwap:.2f}")
    return Signal("hold", price=price, reason=f"above VWAP {vwap:.2f}, no fresh cross")


def intraday_momentum(bars, cfg):
    """Market intraday momentum (Gao, Han, Li & Zhou, 2018): if the first half hour (previous close
    to 10:00) was up, buy at 15:30 for the last half hour."""
    b = _session(bars)
    if b is None:
        return Signal("hold", reason="needs stock session data")
    if b["_min"] != cfg.im_entry_minute or not b["_pc"] or not b["_c10"]:
        return Signal("hold", price=b["c"], reason="not the 15:30 entry bar")
    first_half_hour = b["_c10"] / b["_pc"] - 1
    if first_half_hour > cfg.im_min_move:
        return _entry(bars, cfg, cfg.stop_atr_mult, cfg.take_profit_atr_mult,
                      f"first half hour up {first_half_hour:+.2%}")
    return Signal("hold", price=b["c"], reason=f"first half hour {first_half_hour:+.2%}")


def overnight_hold(bars, cfg):
    """Overnight drift (Cliff, Cooper & Gulen, "Like Night and Day"): historically most of the US market's
    gains came between the close and the next open. Buy just before the close, sell after the next open."""
    b = _session(bars)
    if b is None:
        return Signal("hold", reason="needs stock session data")
    if b["_min"] < OPEN_MIN + cfg.overnight_exit_after_minutes:
        return Signal("sell", price=b["c"], reason="morning after an overnight hold")
    if b["_min"] == cfg.overnight_entry_minute:
        price = b["c"]
        stop = price * (1 - cfg.overnight_stop_pct)
        return Signal("buy", price=price, stop=stop, target=price * (1 + 10 * cfg.overnight_stop_pct),
                      reason="holding overnight into the next open")
    return Signal("hold", price=b["c"], reason="waits for the close")


def news_gate(fn, needs_news):
    """Research on news and returns: moves driven by news tend to continue, moves without news tend to
    reverse. So "ride the move" strategies only buy stocks with news in the last day (needs_news=True)
    and the bounce-back strategy only buys stocks without it (needs_news=False). Bars without news data
    (crypto, overnight) aren't filtered."""
    def gated(bars, cfg):
        sig = fn(bars, cfg)
        if sig.action == "buy" and cfg.news_filter and bars and "_news" in bars[-1]:
            if bars[-1]["_news"] != needs_news:
                return replace(sig, action="hold",
                               reason=f"{sig.reason}; skipped: {'no news today' if needs_news else 'stock is in the news'}")
        return sig
    gated.__name__ = fn.__name__
    gated.__doc__ = fn.__doc__
    return gated


trend = news_gate(trend, True)
breakout = news_gate(breakout, True)
mean_reversion = news_gate(mean_reversion, False)
orb = news_gate(orb, True)
vwap_trend = news_gate(vwap_trend, True)
intraday_momentum = news_gate(intraday_momentum, True)

orb.hold_to_close = True
intraday_momentum.hold_to_close = True
overnight_hold.overnight = True

LONG_STRATEGIES.update({"trend": trend, "mean_reversion": mean_reversion, "breakout": breakout, "orb": orb,
                        "vwap_trend": vwap_trend, "intraday_momentum": intraday_momentum,
                        "overnight_hold": overnight_hold})
STRATEGIES = dict(LONG_STRATEGIES)
STRATEGIES.update({f"{name}_short": short_version(fn) for name, fn in LONG_STRATEGIES.items()})
