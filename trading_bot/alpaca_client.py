"""Thin REST wrapper around the Alpaca trading and market-data APIs."""
from datetime import datetime, timedelta, timezone

import time
import uuid
from collections import deque
from urllib.parse import quote

import requests


class AlpacaError(RuntimeError):
    pass


def is_crypto(symbol):
    return "/" in symbol


def norm(symbol):
    """Alpaca reports crypto positions as BTCUSD but orders as BTC/USD; compare on this."""
    return symbol.replace("/", "")


STOCK_EXCHANGES = {"NYSE", "NASDAQ", "ARCA", "AMEX", "BATS"}
MAX_REQUESTS_PER_MINUTE = 180  # Alpaca's free plan allows 200


class AlpacaClient:
    def __init__(self, config, session=None):
        self.cfg = config
        self._recent = deque()
        self.session = session or requests.Session()
        self.session.headers.update({
            "APCA-API-KEY-ID": config.api_key,
            "APCA-API-SECRET-KEY": config.api_secret,
        })

    def _throttle(self):
        now = time.monotonic()
        while self._recent and now - self._recent[0] > 60:
            self._recent.popleft()
        if len(self._recent) >= MAX_REQUESTS_PER_MINUTE:
            time.sleep(60 - (now - self._recent[0]) + 0.1)
        self._recent.append(time.monotonic())

    def _request(self, method, url, **kwargs):
        self._throttle()
        resp = self.session.request(method, url, timeout=30, **kwargs)
        if resp.status_code >= 400:
            raise AlpacaError(f"{method} {url} -> {resp.status_code}: {resp.text}")
        if resp.status_code == 204 or not resp.content:
            return None
        return resp.json()

    def _trade(self, method, path, **kwargs):
        return self._request(method, f"{self.cfg.base_url}{path}", **kwargs)

    # --- account / market state ---
    def get_account(self):
        return self._trade("GET", "/account")

    def get_clock(self):
        return self._trade("GET", "/clock")

    def get_positions(self):
        return self._trade("GET", "/positions")

    def get_open_orders(self):
        return self._trade("GET", "/orders", params={"status": "open", "limit": 500})

    # --- orders ---
    def get_asset(self, symbol):
        return self._trade("GET", f"/assets/{quote(symbol, safe='')}")

    def submit_entry(self, symbol, qty, side, take_profit, stop_loss, client_order_id=None, fractional=False,
                     overnight=False):
        """Open a position. side is "buy" (long) or "sell" (short).

        Whole-share stock trades get a bracket order. Crypto and fractional shares can't
        use brackets, so the bot watches their stop/target itself.
        """
        if is_crypto(symbol) or fractional:
            body = {"symbol": symbol, "qty": str(qty), "side": "buy", "type": "market",
                    "time_in_force": "gtc" if is_crypto(symbol) else "day"}
            if client_order_id:
                body["client_order_id"] = client_order_id
            return self._trade("POST", "/orders", json=body)
        body = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side,
            "type": "market",
            "time_in_force": "gtc" if overnight else "day",  # gtc keeps an overnight hold's stop alive tomorrow
            "order_class": "bracket",
            "take_profit": {"limit_price": f"{take_profit:.2f}"},
            "stop_loss": {"stop_price": f"{stop_loss:.2f}"},
        }
        if client_order_id:
            body["client_order_id"] = client_order_id
        return self._trade("POST", "/orders", json=body)

    def submit_option_buy(self, contract, qty, limit_price, client_order_id=None):
        body = {"symbol": contract, "qty": str(qty), "side": "buy", "type": "limit",
                "limit_price": f"{limit_price:.2f}", "time_in_force": "day"}
        if client_order_id:
            body["client_order_id"] = client_order_id
        return self._trade("POST", "/orders", json=body)

    def get_option_candidates(self, underlying, kind, price, exp_from, exp_to, strike_pct=0.05):
        """Tradable calls/puts near the money with live quotes and delta."""
        contracts = self._trade("GET", "/options/contracts", params={
            "underlying_symbols": underlying, "type": kind, "status": "active",
            "expiration_date_gte": exp_from.isoformat(), "expiration_date_lte": exp_to.isoformat(),
            "strike_price_gte": f"{price * (1 - strike_pct):.2f}",
            "strike_price_lte": f"{price * (1 + strike_pct):.2f}", "limit": 100,
        }).get("option_contracts") or []
        contracts = [c for c in contracts if c.get("tradable")]
        if not contracts:
            return []
        data_root = self.cfg.data_url.rsplit("/", 1)[0]
        snaps = self._request("GET", f"{data_root}/v1beta1/options/snapshots", params={
            "symbols": ",".join(c["symbol"] for c in contracts), "feed": "indicative",
        }).get("snapshots") or {}
        out = []
        for c in contracts:
            snap = snaps.get(c["symbol"]) or {}
            quote, greeks = snap.get("latestQuote") or {}, snap.get("greeks") or {}
            if quote.get("bp") and quote.get("ap") and greeks.get("delta") is not None:
                out.append({"symbol": c["symbol"], "expiration": c["expiration_date"],
                            "strike": float(c["strike_price"]), "bid": quote["bp"], "ask": quote["ap"],
                            "delta": greeks["delta"]})
        return out

    def get_last_exit_fill(self, symbol, after, exit_side="sell"):
        """Average fill price of the most recent filled exit order for `symbol` since `after` (ISO time)."""
        orders = self._trade("GET", "/orders", params={
            "status": "closed", "symbols": symbol, "after": after,
            "direction": "desc", "nested": "true", "limit": 50,
        }) or []
        fills = []
        for order in orders:
            for o in [order] + (order.get("legs") or []):
                if o["side"] == exit_side and o["status"] == "filled" and o.get("filled_avg_price"):
                    fills.append((o["filled_at"], float(o["filled_avg_price"])))
        return max(fills)[1] if fills else None

    def close_position(self, symbol):
        # Cancel the bracket legs first, otherwise the shares are held by them.
        for order in self.get_open_orders() or []:
            if norm(order["symbol"]) == norm(symbol):
                self._trade("DELETE", f"/orders/{order['id']}")
        return self._trade("DELETE", f"/positions/{norm(symbol)}")

    def submit_crypto_stop(self, symbol, qty, stop_price):
        """Good-til-cancelled stop-limit sell that sits at Alpaca (limit 1% below the stop)."""
        return self._trade("POST", "/orders", json={
            "symbol": symbol, "qty": str(qty), "side": "sell", "type": "stop_limit",
            "stop_price": f"{stop_price:.6g}", "limit_price": f"{stop_price * 0.99:.6g}",
            "time_in_force": "gtc", "client_order_id": f"stop-{norm(symbol)}-{uuid.uuid4().hex[:8]}",
        })

    def submit_extended_entry(self, symbol, qty, limit_price, client_order_id):
        """Limit buy that can fill in pre-market, after-hours and the overnight session."""
        return self._trade("POST", "/orders", json={
            "symbol": symbol, "qty": str(qty), "side": "buy", "type": "limit",
            "limit_price": f"{limit_price:.2f}", "time_in_force": "day", "extended_hours": True,
            "client_order_id": client_order_id,
        })

    def submit_extended_exit(self, symbol, qty, side, limit_price):
        """Limit order that can fill in pre-market, after-hours and the overnight session.
        Cancels the position's other orders first (a bracket's stop holds the shares)."""
        for order in self.get_open_orders() or []:
            if norm(order["symbol"]) == norm(symbol):
                self._trade("DELETE", f"/orders/{order['id']}")
        return self._trade("POST", "/orders", json={
            "symbol": symbol, "qty": str(abs(float(qty))), "side": side, "type": "limit",
            "limit_price": f"{limit_price:.2f}", "time_in_force": "day", "extended_hours": True,
            "client_order_id": f"protect-{norm(symbol)}-{uuid.uuid4().hex[:8]}",
        })

    def cancel_order(self, order_id):
        return self._trade("DELETE", f"/orders/{order_id}")

    def close_all_positions(self):
        return self._trade("DELETE", "/positions", params={"cancel_orders": "true"})

    # --- market data ---
    def get_most_active_stocks(self, top):
        """Most-traded stocks today with price and IEX dollar volume: [(symbol, price, dollar_volume)]."""
        data_root = self.cfg.data_url.rsplit("/", 1)[0]
        actives = self._request("GET", f"{data_root}/v1beta1/screener/stocks/most-actives",
                                params={"by": "trades", "top": top})["most_actives"]
        symbols = [a["symbol"] for a in actives]
        snaps = self._request("GET", f"{self.cfg.data_url}/stocks/snapshots",
                              params={"symbols": ",".join(symbols), "feed": self.cfg.data_feed}) or {}
        out = []
        for sym in symbols:
            snap = snaps.get(sym) or {}
            bar = snap.get("dailyBar") or snap.get("prevDailyBar") or {}
            if bar:
                out.append((sym, bar["c"], bar["c"] * bar["v"]))
        return out

    def get_tradable_stocks(self):
        """Every active, tradable US-listed stock and ETF on a major exchange."""
        assets = self._trade("GET", "/assets", params={"status": "active", "asset_class": "us_equity"}) or []
        return [a["symbol"] for a in assets
                if a.get("tradable") and a.get("exchange") in STOCK_EXCHANGES and a["symbol"].isalpha()]

    def get_overnight_tradable(self):
        """Symbols Alpaca lets you trade in the overnight session (8 PM - 4 AM New York)."""
        assets = self._trade("GET", "/assets", params={"status": "active", "asset_class": "us_equity"}) or []
        return {a["symbol"] for a in assets
                if a.get("tradable") and "overnight_tradable" in (a.get("attributes") or [])
                and not a.get("overnight_halted")}

    def get_latest_quotes(self, symbols, feed, batch=200):
        """{symbol: (bid, ask)} right now. feed "overnight" is free and live during the overnight session."""
        out = {}
        for i in range(0, len(symbols), batch):
            data = self._request("GET", f"{self.cfg.data_url}/stocks/quotes/latest", params={
                "symbols": ",".join(symbols[i:i + batch]), "feed": feed}) or {}
            for sym, q in (data.get("quotes") or {}).items():
                if q.get("bp") and q.get("ap") and q["ap"] >= q["bp"]:
                    out[sym] = (q["bp"], q["ap"])
        return out

    def get_news(self, start, max_pages=600):
        """[(created_at ISO, [symbols])] for every news article since `start` (Alpaca's free Benzinga feed)."""
        data_root = self.cfg.data_url.rsplit("/", 1)[0]
        params = {"start": start.strftime("%Y-%m-%dT%H:%M:%SZ"), "limit": 50, "sort": "asc"}
        out = []
        for _ in range(max_pages):
            data = self._request("GET", f"{data_root}/v1beta1/news", params=params) or {}
            out += [(a["created_at"], a.get("symbols") or []) for a in data.get("news") or []]
            if not data.get("next_page_token"):
                break
            params["page_token"] = data["next_page_token"]
        return out

    def get_daily_stats(self, symbols, batch=200):
        """{symbol: (price, dollar_volume)} from the latest daily bar, fetched in batches."""
        out = {}
        for i in range(0, len(symbols), batch):
            snaps = self._request("GET", f"{self.cfg.data_url}/stocks/snapshots", params={
                "symbols": ",".join(symbols[i:i + batch]), "feed": self.cfg.data_feed}) or {}
            for sym, snap in snaps.items():
                bar = (snap or {}).get("dailyBar") or (snap or {}).get("prevDailyBar") or {}
                if bar.get("c"):
                    out[sym] = (bar["c"], bar["c"] * bar.get("v", 0))
        return out

    def get_stock_bars(self, symbols, timeframe, lookback_days=5, batch=50, feed=None):
        """{symbol: bars} for many stocks, a few requests per batch instead of one per stock.
        feed "boats" gives overnight-session history (free, about 15 minutes behind)."""
        start = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()
        out = {s: [] for s in symbols}
        for i in range(0, len(symbols), batch):
            params = {"symbols": ",".join(symbols[i:i + batch]), "timeframe": timeframe, "start": start,
                      "limit": 10000, "feed": feed or self.cfg.data_feed, "adjustment": "raw"}
            while True:
                data = self._request("GET", f"{self.cfg.data_url}/stocks/bars", params=params)
                for sym, bars in (data.get("bars") or {}).items():
                    out.setdefault(sym, []).extend(
                        {"t": b["t"], "o": b["o"], "h": b["h"], "l": b["l"], "c": b["c"], "v": b["v"]} for b in bars)
                if not data.get("next_page_token"):
                    break
                params["page_token"] = data["next_page_token"]
        return out

    def get_movers(self, top=20):
        """Today's biggest gainers and losers: [(symbol, price, percent_change)]."""
        data_root = self.cfg.data_url.rsplit("/", 1)[0]
        data = self._request("GET", f"{data_root}/v1beta1/screener/stocks/movers", params={"top": top}) or {}
        return [(m["symbol"], m.get("price", 0), m.get("percent_change", 0))
                for m in (data.get("gainers") or []) + (data.get("losers") or [])]

    def get_bars(self, symbol, timeframe, lookback_days=5):
        start = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()
        params = {"symbols": symbol, "timeframe": timeframe, "start": start, "limit": 10000}
        if is_crypto(symbol):
            url = f"{self.cfg.data_url.rsplit('/', 1)[0]}/v1beta3/crypto/us/bars"
        else:
            url = f"{self.cfg.data_url}/stocks/bars"
            params.update(feed=self.cfg.data_feed, adjustment="raw")
        bars = []
        while True:
            data = self._request("GET", url, params=params)
            bars.extend((data.get("bars") or {}).get(symbol, []))
            token = data.get("next_page_token")
            if not token:
                return bars
            params["page_token"] = token
