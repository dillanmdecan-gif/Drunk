"""
MULTI-SYMBOL ADAPTIVE TRADING BOT - DERIV PLATFORM
Parallel per-symbol bots | RL + Micro Model + Probabilistic Fusion

SETUP:
  pip install websockets pandas numpy scipy colorama python-dotenv

USAGE:
  python multi_symbol_bot.py

ENVIRONMENT (.env file or export):
  DERIV_API_TOKEN=your_api_token_here
  DERIV_APP_ID=1089          # Use 1089 for demo, your app ID for live
  TRADE_MODE=demo            # demo | live
  BASE_STAKE=0.5             # Minimum stake per trade in USD
  MAX_DAILY_LOSS=10.0        # Per-symbol kill switch: max $ loss per symbol/day
  MAX_GLOBAL_DAILY_LOSS=30.0 # Global kill switch: max $ loss across ALL symbols/day
  MAX_CONCURRENT_TRADES=3    # Max number of open positions at once across all symbols

SYMBOL SELECTION:
  Edit ACTIVE_SYMBOLS below. Each entry must exist in SYMBOL_CONFIGS.
  Add/remove freely — each symbol runs as an independent parallel bot.

ARCHITECTURE:
  - One DerivBot instance per symbol, all sharing a single WebSocket connection
  - Each bot has its own: CandleStore, RLAgent, Calibrator, Martingale, Thresholds
  - GlobalRiskManager enforces MAX_CONCURRENT_TRADES and MAX_GLOBAL_DAILY_LOSS
  - All bots authorized under the same API token / account
  - Per-symbol granularity and expiry come from SYMBOL_CONFIGS metadata
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import asyncio
import json
import os
import time
import math
import logging
from collections import deque
from datetime import datetime, timezone
from typing import Optional, Dict
import websockets
from websockets.exceptions import ConnectionClosed
import pandas as pd
import numpy as np
from scipy.special import expit
from colorama import init, Fore, Style

init(autoreset=True)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# -------------------------------------------------------
# GLOBAL CONFIGURATION
# -------------------------------------------------------
DERIV_API_TOKEN       = os.getenv("DERIV_API_TOKEN", "YOUR_TOKEN_HERE")
DERIV_APP_ID          = os.getenv("DERIV_APP_ID", "1089")
TRADE_MODE            = os.getenv("TRADE_MODE", "demo")
BASE_STAKE            = float(os.getenv("BASE_STAKE", "0.5"))
MAX_DAILY_LOSS        = float(os.getenv("MAX_DAILY_LOSS", "10.0"))
MAX_GLOBAL_DAILY_LOSS = float(os.getenv("MAX_GLOBAL_DAILY_LOSS", "100.0"))
MAX_CONCURRENT_TRADES = int(os.getenv("MAX_CONCURRENT_TRADES", "22"))

WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"

RECONNECT_DELAY_MIN = 5
RECONNECT_DELAY_MAX = 120
WS_PING_INTERVAL    = 60
WS_PING_TIMEOUT     = 60

# Per-trade safety constants
MARTINGALE_MULTIPLIER  = 1.25
MARTINGALE_MAX_STEPS   = 4
MARTINGALE_TRIGGER     = 2
THRESHOLD_NO_TRADE     = 0.55
THRESHOLD_SMALL        = 0.57
THRESHOLD_NORMAL       = 0.67
MAX_CONSECUTIVE_LOSSES = 3
KILL_SWITCH_PAUSE_MIN  = 45
CANDLE_HISTORY         = 50
MIN_CANDLES            = 20
CONTRACT_REQUERY_TIMEOUT   = 15
SETTLEMENT_GRACE_SECONDS   = 30
SETTLEMENT_POLL_INTERVAL   = 10
SETTLEMENT_POLL_MAX_TRIES  = 6
SETTLEMENT_TERMINAL        = {"won", "lost", "cancelled", "resale_cancelled", "sold"}

# -------------------------------------------------------
# SYMBOL CATALOGUE
# Each entry: symbol_id -> config dict
#   granularity_main : seconds for the primary (directional bias) candle
#   granularity_entry: seconds for the entry-timing candle (always 1/5 of main, min 60)
#   expiry_seconds   : contract duration in seconds
#   display_name     : human-readable label used in logs
#   candle_history_entry: how many entry-TF candles to keep in memory
# -------------------------------------------------------
SYMBOL_CONFIGS: Dict[str, dict] = {
    # ── Commodities (5-min / 1-min) ────────────────────────────────────────
    "frxXAUUSD": {
        "display_name":          "Gold/USD",
        "granularity_main":      300,
        "granularity_entry":     60,
        "expiry_seconds":        300,
        "candle_history_entry":  60,
    },
    "frxXAGUSD": {
        "display_name":          "Silver/USD",
        "granularity_main":      300,
        "granularity_entry":     60,
        "expiry_seconds":        300,
        "candle_history_entry":  60,
    },

    # ── Major Forex (15-min / 3-min) ───────────────────────────────────────
    "frxEURUSD": {
        "display_name":          "EUR/USD",
        "granularity_main":      900,
        "granularity_entry":     180,
        "expiry_seconds":        900,
        "candle_history_entry":  60,
    },
    "frxGBPUSD": {
        "display_name":          "GBP/USD",
        "granularity_main":      900,
        "granularity_entry":     180,
        "expiry_seconds":        900,
        "candle_history_entry":  60,
    },
    "frxUSDJPY": {
        "display_name":          "USD/JPY",
        "granularity_main":      900,
        "granularity_entry":     180,
        "expiry_seconds":        900,
        "candle_history_entry":  60,
    },
    "frxAUDUSD": {
        "display_name":          "AUD/USD",
        "granularity_main":      900,
        "granularity_entry":     180,
        "expiry_seconds":        900,
        "candle_history_entry":  60,
    },
    "frxUSDCAD": {
        "display_name":          "USD/CAD",
        "granularity_main":      900,
        "granularity_entry":     180,
        "expiry_seconds":        900,
        "candle_history_entry":  60,
    },
    "frxUSDCHF": {
        "display_name":          "USD/CHF",
        "granularity_main":      900,
        "granularity_entry":     180,
        "expiry_seconds":        900,
        "candle_history_entry":  60,
    },
    "frxNZDUSD": {
        "display_name":          "NZD/USD",
        "granularity_main":      900,
        "granularity_entry":     180,
        "expiry_seconds":        900,
        "candle_history_entry":  60,
    },

    # ── Minor / Cross Forex (15-min / 3-min) ───────────────────────────────
    "frxEURGBP": {"display_name": "EUR/GBP", "granularity_main": 900, "granularity_entry": 180, "expiry_seconds": 900, "candle_history_entry": 60},
    "frxEURJPY": {"display_name": "EUR/JPY", "granularity_main": 900, "granularity_entry": 180, "expiry_seconds": 900, "candle_history_entry": 60},
    "frxGBPJPY": {"display_name": "GBP/JPY", "granularity_main": 900, "granularity_entry": 180, "expiry_seconds": 900, "candle_history_entry": 60},
    "frxAUDJPY": {"display_name": "AUD/JPY", "granularity_main": 900, "granularity_entry": 180, "expiry_seconds": 900, "candle_history_entry": 60},
    "frxEURCAD": {"display_name": "EUR/CAD", "granularity_main": 900, "granularity_entry": 180, "expiry_seconds": 900, "candle_history_entry": 60},
    "frxEURCHF": {"display_name": "EUR/CHF", "granularity_main": 900, "granularity_entry": 180, "expiry_seconds": 900, "candle_history_entry": 60},
    "frxEURAUD": {"display_name": "EUR/AUD", "granularity_main": 900, "granularity_entry": 180, "expiry_seconds": 900, "candle_history_entry": 60},
    "frxGBPCAD": {"display_name": "GBP/CAD", "granularity_main": 900, "granularity_entry": 180, "expiry_seconds": 900, "candle_history_entry": 60},
    "frxGBPCHF": {"display_name": "GBP/CHF", "granularity_main": 900, "granularity_entry": 180, "expiry_seconds": 900, "candle_history_entry": 60},
    "frxGBPAUD": {"display_name": "GBP/AUD", "granularity_main": 900, "granularity_entry": 180, "expiry_seconds": 900, "candle_history_entry": 60},
    "frxGBPNZD": {"display_name": "GBP/NZD", "granularity_main": 900, "granularity_entry": 180, "expiry_seconds": 900, "candle_history_entry": 60},
    "frxAUDCAD": {"display_name": "AUD/CAD", "granularity_main": 900, "granularity_entry": 180, "expiry_seconds": 900, "candle_history_entry": 60},
    "frxAUDCHF": {"display_name": "AUD/CHF", "granularity_main": 900, "granularity_entry": 180, "expiry_seconds": 900, "candle_history_entry": 60},
    "frxAUDNZD": {"display_name": "AUD/NZD", "granularity_main": 900, "granularity_entry": 180, "expiry_seconds": 900, "candle_history_entry": 60},
    "frxNZDJPY": {"display_name": "NZD/JPY", "granularity_main": 900, "granularity_entry": 180, "expiry_seconds": 900, "candle_history_entry": 60},

    # ── Synthetic Indices (1-min / 1-min) ──────────────────────────────────
    "R_100":    {"display_name": "Volatility 100 Index",     "granularity_main": 60, "granularity_entry": 60, "expiry_seconds": 60, "candle_history_entry": 60},
    "R_75":     {"display_name": "Volatility 75 Index",      "granularity_main": 60, "granularity_entry": 60, "expiry_seconds": 60, "candle_history_entry": 60},
    "R_50":     {"display_name": "Volatility 50 Index",      "granularity_main": 60, "granularity_entry": 60, "expiry_seconds": 60, "candle_history_entry": 60},
    "R_25":     {"display_name": "Volatility 25 Index",      "granularity_main": 60, "granularity_entry": 60, "expiry_seconds": 60, "candle_history_entry": 60},
    "R_10":     {"display_name": "Volatility 10 Index",      "granularity_main": 60, "granularity_entry": 60, "expiry_seconds": 60, "candle_history_entry": 60},
    "1HZ100V":  {"display_name": "Volatility 100 (1s) Index","granularity_main": 60, "granularity_entry": 60, "expiry_seconds": 60, "candle_history_entry": 60},
    "1HZ75V":   {"display_name": "Volatility 75 (1s) Index", "granularity_main": 60, "granularity_entry": 60, "expiry_seconds": 60, "candle_history_entry": 60},
    "1HZ50V":   {"display_name": "Volatility 50 (1s) Index", "granularity_main": 60, "granularity_entry": 60, "expiry_seconds": 60, "candle_history_entry": 60},
    "JD100":    {"display_name": "Jump 100 Index",           "granularity_main": 60, "granularity_entry": 60, "expiry_seconds": 60, "candle_history_entry": 60},
    "JD75":     {"display_name": "Jump 75 Index",            "granularity_main": 60, "granularity_entry": 60, "expiry_seconds": 60, "candle_history_entry": 60},
    "JD50":     {"display_name": "Jump 50 Index",            "granularity_main": 60, "granularity_entry": 60, "expiry_seconds": 60, "candle_history_entry": 60},
    "stpRNG":   {"display_name": "Step Index 100",           "granularity_main": 60, "granularity_entry": 60, "expiry_seconds": 60, "candle_history_entry": 60},

    # ── Forex Baskets (15-min / 3-min) ────────────────────────────────────
    "WLDXAU":   {"display_name": "Gold Basket",              "granularity_main": 900, "granularity_entry": 180, "expiry_seconds": 900, "candle_history_entry": 60},
    "WLDUSD":   {"display_name": "USD Basket",               "granularity_main": 900, "granularity_entry": 180, "expiry_seconds": 900, "candle_history_entry": 60},
    "WLDEUR":   {"display_name": "EUR Basket",               "granularity_main": 900, "granularity_entry": 180, "expiry_seconds": 900, "candle_history_entry": 60},
    "WLDGBP":   {"display_name": "GBP Basket",               "granularity_main": 900, "granularity_entry": 180, "expiry_seconds": 900, "candle_history_entry": 60},
}

# -------------------------------------------------------
# *** EDIT THIS LIST TO CHOOSE WHICH SYMBOLS TO TRADE ***
# -------------------------------------------------------
ACTIVE_SYMBOLS = [
    # ── Major Forex (15-min expiry) ────────────────────────────────────────
    "frxEURUSD",   # EUR/USD
    "frxGBPUSD",   # GBP/USD
    "frxUSDJPY",   # USD/JPY
    "frxAUDUSD",   # AUD/USD
    "frxUSDCAD",   # USD/CAD
    "frxUSDCHF",   # USD/CHF
    "frxNZDUSD",   # NZD/USD

    # ── Minor / Cross Forex (15-min expiry) ────────────────────────────────
    "frxEURGBP",   # EUR/GBP
    "frxEURJPY",   # EUR/JPY
    "frxGBPJPY",   # GBP/JPY
    "frxAUDJPY",   # AUD/JPY
    "frxEURCAD",   # EUR/CAD
    "frxEURCHF",   # EUR/CHF
    "frxEURAUD",   # EUR/AUD
    "frxGBPCAD",   # GBP/CAD
    "frxGBPCHF",   # GBP/CHF
    "frxGBPAUD",   # GBP/AUD
    "frxGBPNZD",   # GBP/NZD
    "frxAUDCAD",   # AUD/CAD
    "frxAUDCHF",   # AUD/CHF
    "frxAUDNZD",   # AUD/NZD
   
   ]

# -------------------------------------------------------
# LOGGING SETUP
# -------------------------------------------------------
log_filename = f"multi_bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG:    Fore.CYAN,
        logging.INFO:     Fore.WHITE,
        logging.WARNING:  Fore.YELLOW,
        logging.ERROR:    Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
        color = self.COLORS.get(record.levelno, "")
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return (
            f"{Fore.CYAN}[{ts}]{Style.RESET_ALL} "
            f"{color}{record.getMessage()}{Style.RESET_ALL}"
        )


logger = logging.getLogger("MULTI_BOT")
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(ColorFormatter())
logger.addHandler(ch)

fh = logging.FileHandler(log_filename, encoding="utf-8")
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
logger.addHandler(fh)


def sym_log(symbol: str, level: int, msg: str):
    """Prefix every log line with the symbol tag for easy filtering."""
    cfg = SYMBOL_CONFIGS.get(symbol, {})
    name = cfg.get("display_name", symbol)
    tag = f"[{name:<18}] "
    logger.log(level, tag + msg)


# -------------------------------------------------------
# GLOBAL RISK MANAGER
# Shared across all SymbolBot instances. Enforces:
#   - MAX_CONCURRENT_TRADES: caps simultaneous open positions
#   - MAX_GLOBAL_DAILY_LOSS: hard stop when total daily PnL hits the limit
# -------------------------------------------------------
class GlobalRiskManager:
    def __init__(self):
        self._lock           = asyncio.Lock()
        self._open_trades    = 0
        self._daily_pnl      = 0.0
        self._daily_pnl_date = datetime.now(timezone.utc).date()
        self._killed         = False

    def _maybe_reset_daily(self):
        today = datetime.now(timezone.utc).date()
        if today != self._daily_pnl_date:
            logger.info(
                f"[GlobalRisk] New UTC day — resetting global PnL "
                f"(was ${self._daily_pnl:+.2f})"
            )
            self._daily_pnl      = 0.0
            self._daily_pnl_date = today
            self._killed         = False

    async def request_trade(self) -> bool:
        """
        Called before opening a new contract.
        Returns True if the trade is allowed, False if it should be blocked.
        """
        async with self._lock:
            self._maybe_reset_daily()
            if self._killed:
                logger.warning("[GlobalRisk] BLOCKED — global kill switch is active.")
                return False
            if self._open_trades >= MAX_CONCURRENT_TRADES:
                logger.warning(
                    f"[GlobalRisk] BLOCKED — concurrent trade limit reached "
                    f"({self._open_trades}/{MAX_CONCURRENT_TRADES})."
                )
                return False
            self._open_trades += 1
            return True

    async def release_trade(self, profit: float):
        """Called when a trade settles (win/loss/cancel)."""
        async with self._lock:
            self._open_trades = max(0, self._open_trades - 1)
            self._daily_pnl  += profit
            self._maybe_reset_daily()
            if self._daily_pnl <= -MAX_GLOBAL_DAILY_LOSS and not self._killed:
                self._killed = True
                logger.critical(
                    f"[GlobalRisk] GLOBAL KILL SWITCH — daily PnL ${self._daily_pnl:.2f} "
                    f"hit limit -${MAX_GLOBAL_DAILY_LOSS:.2f}. All symbols halted."
                )

    @property
    def is_killed(self) -> bool:
        self._maybe_reset_daily()
        return self._killed

    def status(self) -> str:
        return (
            f"open={self._open_trades}/{MAX_CONCURRENT_TRADES} "
            f"global_pnl=${self._daily_pnl:+.2f} "
            f"killed={self._killed}"
        )


# Module-level singleton shared by all bots
global_risk = GlobalRiskManager()


# -------------------------------------------------------
# CANDLE STORE
# -------------------------------------------------------
class CandleStore:
    def __init__(self, maxlen: int = CANDLE_HISTORY, min_candles: int = MIN_CANDLES):
        self.candles    = deque(maxlen=maxlen)
        self.min_candles = min_candles

    def add(self, epoch, open_, high, low, close, volume=0):
        self.candles.append({
            "epoch": epoch, "open": float(open_), "high": float(high),
            "low": float(low), "close": float(close), "volume": float(volume)
        })

    def df(self) -> pd.DataFrame:
        return pd.DataFrame(list(self.candles))

    def ready(self) -> bool:
        return len(self.candles) >= self.min_candles


# -------------------------------------------------------
# ADAPTIVE THRESHOLDS
# -------------------------------------------------------
class AdaptiveThresholds:
    WINDOW    = 30
    MAX_SHIFT = 0.08

    def __init__(self):
        self._outcomes: deque = deque(maxlen=self.WINDOW)
        self._pnl:      deque = deque(maxlen=self.WINDOW)

    def record(self, won: bool, profit: float):
        self._outcomes.append(1 if won else 0)
        self._pnl.append(profit)

    def _shift(self) -> float:
        n = len(self._outcomes)
        if n < 5:
            return -0.03
        win_rate = sum(self._outcomes) / n
        if win_rate < 0.40:
            t = (0.40 - win_rate) / 0.40
            return -self.MAX_SHIFT * min(t, 1.0)
        elif win_rate > 0.60:
            t = (win_rate - 0.60) / 0.40
            return +self.MAX_SHIFT * min(t, 1.0)
        return 0.0

    @property
    def no_trade(self) -> float:
        return round(float(np.clip(
            THRESHOLD_NO_TRADE + self._shift(),
            THRESHOLD_NO_TRADE - self.MAX_SHIFT,
            THRESHOLD_NO_TRADE + self.MAX_SHIFT
        )), 4)

    @property
    def small(self) -> float:
        return round(float(np.clip(
            THRESHOLD_SMALL + self._shift(),
            THRESHOLD_SMALL - self.MAX_SHIFT,
            THRESHOLD_SMALL + self.MAX_SHIFT
        )), 4)

    @property
    def normal(self) -> float:
        return round(float(np.clip(
            THRESHOLD_NORMAL + self._shift(),
            THRESHOLD_NORMAL - self.MAX_SHIFT,
            THRESHOLD_NORMAL + self.MAX_SHIFT
        )), 4)

    @property
    def status(self) -> str:
        n = len(self._outcomes)
        wr = f"{sum(self._outcomes)/n*100:.1f}%" if n > 0 else "N/A"
        shift = self._shift()
        direction = "loosened" if shift < 0 else ("tightened" if shift > 0 else "neutral")
        return (
            f"WR={wr}({n}) shift={shift:+.3f}({direction}) | "
            f"NoTrade<{self.no_trade} Small<{self.small} Normal<{self.normal}"
        )


# -------------------------------------------------------
# REGIME DETECTOR
# -------------------------------------------------------
def detect_regime(df: pd.DataFrame) -> str:
    atr = compute_atr(df, 14)
    if len(atr) < 5:
        return "QUIET"
    current_atr = atr.iloc[-1]
    avg_atr     = atr.iloc[-14:].mean()
    atr_ratio   = current_atr / avg_atr if avg_atr > 0 else 1.0
    closes      = df["close"].values
    diffs       = np.diff(closes[-9:])
    positive    = np.sum(diffs > 0)
    negative    = np.sum(diffs < 0)
    consistency = max(positive, negative) / len(diffs)
    if atr_ratio > 1.5 and consistency > 0.7:
        return "TRENDING"
    elif atr_ratio > 1.3:
        return "EXPANDING"
    else:
        recent_momentum = abs(closes[-1] - closes[-5])
        avg_momentum    = np.mean(np.abs(np.diff(closes[-10:]))) * 4
        if recent_momentum < avg_momentum * 0.5 and atr_ratio > 1.0:
            return "EXHAUSTION"
        return "QUIET"


# -------------------------------------------------------
# DIRECTION PREDICTOR
# -------------------------------------------------------
def predict_direction(df: pd.DataFrame) -> dict:
    closes = df["close"].values
    opens  = df["open"].values

    if len(closes) < 15:
        return {"direction": "HOLD", "confidence": 0.5, "votes": 0, "signals": {}}

    signals = {}

    mom3 = closes[-1] - closes[-4]
    signals["mom3"] = 1 if mom3 > 0 else -1

    mom8 = closes[-1] - closes[-9]
    signals["mom8"] = 1 if mom8 > 0 else -1

    def ema(arr, n):
        k = 2 / (n + 1)
        e = arr[0]
        for x in arr[1:]:
            e = x * k + e * (1 - k)
        return e

    ema5  = ema(closes[-20:], 5)
    ema13 = ema(closes[-20:], 13)
    signals["ema_cross"] = 1 if ema5 > ema13 else -1

    sma20  = np.mean(closes[-20:])
    std20  = np.std(closes[-20:])
    z_score = (closes[-1] - sma20) / (std20 + 1e-9)
    if abs(z_score) > 1.5:
        signals["mean_rev"] = -1 if z_score > 0 else 1
    else:
        signals["mean_rev"] = signals["mom3"]

    last_body = closes[-1] - opens[-1]
    signals["candle"] = 1 if last_body > 0 else -1

    vote_sum   = sum(signals.values())
    confidence = (vote_sum + 5) / 10.0

    if vote_sum > 0:
        direction = "BUY"
        conf_out  = confidence
    elif vote_sum < 0:
        direction = "SELL"
        conf_out  = 1.0 - confidence
    else:
        direction = "BUY" if signals["ema_cross"] > 0 else "SELL"
        conf_out  = 0.52

    return {
        "direction":  direction,
        "confidence": round(conf_out, 4),
        "votes":      vote_sum,
        "signals":    signals,
        "z_score":    round(float(z_score), 3),
        "ema_gap":    round(float(ema5 - ema13), 4),
    }


# -------------------------------------------------------
# TECHNICAL HELPERS
# -------------------------------------------------------
def compute_atr(df: pd.DataFrame, period=14) -> pd.Series:
    high       = df["high"]
    low        = df["low"]
    close      = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_momentum(df: pd.DataFrame, period=10) -> dict:
    closes = df["close"].values
    if len(closes) < period + 1:
        return {"strength": 0.0, "consistency": 0.0, "acceleration": 0.0}
    diffs       = np.diff(closes[-period:])
    strength    = (closes[-1] - closes[-period]) / closes[-period] * 100
    pos         = np.sum(diffs > 0)
    neg         = np.sum(diffs < 0)
    consistency = (pos - neg) / len(diffs)
    half        = period // 2
    mom1        = closes[-half] - closes[-period]
    mom2        = closes[-1]    - closes[-half]
    return {
        "strength":     float(strength),
        "consistency":  float(consistency),
        "acceleration": float(mom2 - mom1),
    }


def compute_structure(df: pd.DataFrame, lookback=20) -> dict:
    highs = df["high"].values[-lookback:]
    lows  = df["low"].values[-lookback:]
    close = df["close"].values[-1]
    swing_high = float(np.max(highs))
    swing_low  = float(np.min(lows))
    rng        = swing_high - swing_low
    return {
        "swing_high":     swing_high,
        "swing_low":      swing_low,
        "dist_from_high": (swing_high - close) / rng if rng > 0 else 0.5,
        "dist_from_low":  (close - swing_low)  / rng if rng > 0 else 0.5,
        "range":          float(rng),
    }


# -------------------------------------------------------
# RL AGENT  (one instance per symbol)
# -------------------------------------------------------
_RL_FEATURE_NAMES    = ["momentum", "acceleration", "structure", "volatility", "exhaustion_flip"]
_RL_DEFAULT_WEIGHTS  = np.array([0.40, 0.20, 0.20, 0.10, 0.10], dtype=np.float64)


class RLAgent:
    GAIN      = 5.0
    LR_WARMUP = 0.05
    LR_ONLINE = 0.01
    EPOCHS    = 30
    L2        = 1e-4

    def __init__(self):
        self.weights    = _RL_DEFAULT_WEIGHTS.copy()
        self._warmed_up = False

    @staticmethod
    def _extract_features(df: pd.DataFrame, regime: str) -> np.ndarray:
        momentum    = compute_momentum(df, 10)
        structure   = compute_structure(df, 20)
        atr_vals    = compute_atr(df, 14)
        current_atr = float(atr_vals.iloc[-1]) if not atr_vals.empty else 0
        avg_atr     = float(atr_vals.iloc[-14:].mean()) if len(atr_vals) >= 14 else current_atr
        atr_ratio   = current_atr / avg_atr if avg_atr > 0 else 1.0

        f_momentum     = momentum["consistency"]
        f_acceleration = float(np.tanh(momentum["acceleration"] / (avg_atr * 0.5 + 1e-9)))

        mom_signal = np.sign(momentum["consistency"])
        if mom_signal > 0:
            s = structure["dist_from_low"] * 2 - 1
            f_structure = float(np.clip(-s + 0.3, -1, 1))
        else:
            s = structure["dist_from_high"] * 2 - 1
            f_structure = float(np.clip(-s + 0.3, -1, 1))

        f_volatility = float(np.clip((atr_ratio - 1.0) * 0.5, -0.5, 0.5))

        if regime == "EXHAUSTION":
            if structure["dist_from_high"] < 0.15:
                f_exhaustion = -0.3
            elif structure["dist_from_low"] < 0.15:
                f_exhaustion = 0.3
            else:
                f_exhaustion = 0.0
        else:
            f_exhaustion = 0.0

        return np.array(
            [f_momentum, f_acceleration, f_structure, f_volatility, f_exhaustion],
            dtype=np.float64
        )

    def warmup(self, df: pd.DataFrame):
        closes = df["close"].values
        n      = len(df)
        if n < MIN_CANDLES + 1:
            return
        samples = []
        for i in range(MIN_CANDLES, n - 1):
            window = df.iloc[:i + 1]
            regime = detect_regime(window)
            feat   = self._extract_features(window, regime)
            label  = 1.0 if closes[i + 1] > closes[i] else 0.0
            samples.append((feat, label))
        if not samples:
            return
        feats  = np.array([s[0] for s in samples], dtype=np.float64)
        labels = np.array([s[1] for s in samples], dtype=np.float64)
        m      = len(samples)
        for _ in range(self.EPOCHS):
            idx     = np.random.permutation(m)
            scores  = feats[idx] @ self.weights * self.GAIN
            preds   = expit(scores)
            errors  = preds - labels[idx]
            grad    = (feats[idx].T @ errors) / m * self.GAIN + self.L2 * self.weights
            self.weights -= self.LR_WARMUP * grad
        self._warmed_up = True

    def update(self, features: np.ndarray, outcome: int):
        score = float(np.dot(self.weights, features)) * self.GAIN
        pred  = float(expit(score))
        error = pred - float(outcome)
        grad  = features * error * self.GAIN + self.L2 * self.weights
        self.weights -= self.LR_ONLINE * grad

    def predict(self, df: pd.DataFrame, regime: str) -> dict:
        momentum    = compute_momentum(df, 10)
        atr_vals    = compute_atr(df, 14)
        current_atr = float(atr_vals.iloc[-1]) if not atr_vals.empty else 0
        avg_atr     = float(atr_vals.iloc[-14:].mean()) if len(atr_vals) >= 14 else current_atr
        atr_ratio   = current_atr / avg_atr if avg_atr > 0 else 1.0

        features   = self._extract_features(df, regime)
        raw_score  = float(np.dot(self.weights, features))
        confidence = float(expit(raw_score * self.GAIN))

        if confidence > 0.52:
            direction = "BUY"
        elif confidence < 0.48:
            direction = "SELL"
            confidence = 1.0 - confidence
        else:
            dp        = predict_direction(df)
            direction = dp["direction"]
            confidence = dp["confidence"]

        return {
            "direction":  direction,
            "confidence": round(confidence, 4),
            "features":   features,
            "reason": (
                f"mom={momentum['consistency']:.2f} "
                f"accel={momentum['acceleration']:.4f} "
                f"atr_ratio={atr_ratio:.2f} "
                f"warmed_up={self._warmed_up}"
            ),
        }


# -------------------------------------------------------
# MICRO MODEL
# -------------------------------------------------------
def micro_model(df: pd.DataFrame, direction: str) -> dict:
    last_n = df.tail(7)
    if len(last_n) < 5:
        return {"entry_prob": 0.0, "action": "SKIP", "reason": "Insufficient candles"}

    candles = last_n.to_dict("records")
    latest  = candles[-1]
    prev    = candles[-2]

    body       = abs(latest["close"] - latest["open"])
    candle_rng = latest["high"] - latest["low"]
    body_ratio = body / candle_rng if candle_rng > 0 else 0

    avg_body      = np.mean([abs(c["close"] - c["open"]) for c in candles[-5:]])
    body_strength = body / avg_body if avg_body > 0 else 1.0

    if latest["close"] >= latest["open"]:
        upper_wick = latest["high"]  - latest["close"]
        lower_wick = latest["open"]  - latest["low"]
    else:
        upper_wick = latest["high"]  - latest["open"]
        lower_wick = latest["close"] - latest["low"]

    upper_wick_ratio = upper_wick / candle_rng if candle_rng > 0 else 0
    lower_wick_ratio = lower_wick / candle_rng if candle_rng > 0 else 0

    prior_high = prev["high"]
    prior_low  = prev["low"]
    breakout   = 0.0
    if direction == "BUY" and latest["close"] > prior_high:
        breakout = float(np.clip(
            (latest["close"] - prior_high) / (prior_high - prior_low + 1e-9), 0, 1))
    elif direction == "SELL" and latest["close"] < prior_low:
        breakout = float(np.clip(
            (prior_low - latest["close"]) / (prior_high - prior_low + 1e-9), 0, 1))

    micro_mom = 0.0
    if direction == "BUY":
        micro_mom = sum(1 for c in candles[-3:] if c["close"] > c["open"]) / 3.0
    else:
        micro_mom = sum(1 for c in candles[-3:] if c["close"] < c["open"]) / 3.0

    reject_reason = None
    if direction == "BUY"  and upper_wick_ratio > 0.40:
        reject_reason = f"Strong upper wick ({upper_wick_ratio:.2f}) on BUY"
    if direction == "SELL" and lower_wick_ratio > 0.40:
        reject_reason = f"Strong lower wick ({lower_wick_ratio:.2f}) on SELL"
    if body_strength > 2.5:
        reject_reason = f"Candle overextended (body_strength={body_strength:.2f})"

    atr_vals = compute_atr(df, 14)
    if not atr_vals.empty:
        avg_atr = float(atr_vals.iloc[-14:].mean())
        if candle_rng > avg_atr * 2.0:
            reject_reason = f"Late entry - range spike {candle_rng:.4f} > 2x ATR {avg_atr:.4f}"

    if reject_reason:
        return {"entry_prob": 0.20, "action": "SKIP", "reason": reject_reason}

    entry_score = (
        body_ratio    * 0.25 +
        body_strength * 0.15 / 2.0 +
        micro_mom     * 0.35 +
        breakout      * 0.20 +
        (1 - upper_wick_ratio if direction == "BUY" else 1 - lower_wick_ratio) * 0.05
    )
    entry_prob = float(np.clip(entry_score, 0.0, 1.0))
    action = "ENTER" if entry_prob >= 0.50 else "WAIT"
    return {
        "entry_prob":    round(entry_prob, 4),
        "action":        action,
        "body_ratio":    round(body_ratio, 3),
        "body_strength": round(body_strength, 3),
        "micro_mom":     round(micro_mom, 3),
        "breakout":      round(breakout, 3),
        "reason":        f"body={body_ratio:.2f} micro_mom={micro_mom:.2f} breakout={breakout:.2f}"
    }


# -------------------------------------------------------
# PROBABILISTIC FUSION
# -------------------------------------------------------
def fuse_probabilities(
    rl_conf: float, micro_prob: float, regime: str, dp_conf: float = 0.5,
    rl_weight: float = 0.55, micro_weight: float = 0.25, dp_weight: float = 0.20,
) -> float:
    fused = rl_weight * rl_conf + micro_weight * micro_prob + dp_weight * dp_conf
    modifiers = {"TRENDING": 1.08, "EXPANDING": 1.04, "EXHAUSTION": 0.96, "QUIET": 0.92}
    fused *= modifiers.get(regime, 1.0)
    return float(np.clip(fused, 0.0, 1.0))


# -------------------------------------------------------
# ONLINE CALIBRATOR
# -------------------------------------------------------
class OnlineCalibrator:
    def __init__(self, min_samples=50, window=200):
        self.buffer      = deque(maxlen=window)
        self.min_samples = min_samples

    def record(self, predicted_prob: float, outcome: int):
        self.buffer.append((predicted_prob, outcome))

    def calibrate(self, raw_prob: float) -> float:
        if len(self.buffer) < self.min_samples:
            return raw_prob
        bins    = np.linspace(0, 1, 6)
        bin_idx = int(np.clip(np.digitize(raw_prob, bins) - 1, 0, 4))
        n       = len(self.buffer)
        weights, actuals = [], []
        for i, (p, o) in enumerate(self.buffer):
            b = int(np.clip(np.digitize(p, bins) - 1, 0, 4))
            if b == bin_idx:
                decay = math.exp(-0.01 * (n - i))
                weights.append(decay)
                actuals.append(o)
        if len(actuals) < 5:
            return raw_prob
        actual_rate = np.average(actuals, weights=weights)
        trust = min(len(actuals) / 50, 0.7)
        return float(raw_prob * (1 - trust) + actual_rate * trust)

    @property
    def stats(self) -> dict:
        if not self.buffer:
            return {"samples": 0, "win_rate": None}
        outcomes = [o for _, o in self.buffer]
        return {"samples": len(self.buffer), "win_rate": round(sum(outcomes) / len(outcomes), 3)}


# -------------------------------------------------------
# MARTINGALE MANAGER
# -------------------------------------------------------
class MartingaleManager:
    def __init__(self):
        self.base_stake  = BASE_STAKE
        self.multiplier  = MARTINGALE_MULTIPLIER
        self.max_steps   = MARTINGALE_MAX_STEPS
        self.trigger     = MARTINGALE_TRIGGER
        self.loss_streak = 0
        self.mart_step   = 0

    def get_stake(self) -> float:
        if self.mart_step == 0:
            return self.base_stake
        return round(self.base_stake * (self.multiplier ** self.mart_step), 2)

    def record_loss(self):
        self.loss_streak += 1
        if self.loss_streak >= self.trigger:
            self.mart_step = min(self.mart_step + 1, self.max_steps)

    def record_win(self):
        self.loss_streak = 0
        self.mart_step   = 0

    @property
    def status(self) -> str:
        return (
            f"streak={self.loss_streak} "
            f"mart_step={self.mart_step}/{self.max_steps} "
            f"stake=${self.get_stake():.2f}"
        )


# -------------------------------------------------------
# POSITION SIZING
# -------------------------------------------------------
def compute_position_size(
    final_prob: float,
    martingale_stake: float,
    thresholds: AdaptiveThresholds,
) -> tuple:
    edge     = (final_prob * 2) - 1
    no_trade = thresholds.no_trade
    small    = thresholds.small
    normal   = thresholds.normal

    if final_prob < no_trade:
        return 0.0, "NO TRADE"
    elif final_prob < small:
        size_label, size_mult = "SMALL",      0.75
    elif final_prob < normal:
        size_label, size_mult = "NORMAL",     1.0
    else:
        size_label, size_mult = "AGGRESSIVE", 1.25

    size = max(BASE_STAKE, martingale_stake * size_mult * edge * 2)
    return round(max(BASE_STAKE, size), 2), size_label


# -------------------------------------------------------
# DUAL-TIMEFRAME AGREEMENT CHECK
# -------------------------------------------------------
def _entry_tf_agrees(df_entry: pd.DataFrame, direction: str, min_entry_candles: int) -> tuple:
    if df_entry is None or len(df_entry) < min_entry_candles:
        return True, "entry-TF data insufficient - bias only"

    micro_e = micro_model(df_entry, direction)
    mom_e   = compute_momentum(df_entry, 5)
    recent  = df_entry.tail(4).to_dict("records")

    if direction == "BUY":
        aligned = sum(1 for c in recent if c["close"] > c["open"])
        mom_ok  = mom_e["consistency"] > 0.3
    else:
        aligned = sum(1 for c in recent if c["close"] < c["open"])
        mom_ok  = mom_e["consistency"] < -0.3

    candle_agree = aligned >= 3
    micro_agree  = micro_e["action"] == "ENTER"
    agrees       = candle_agree or micro_agree or mom_ok
    reason = (
        f"entry-TF: aligned={aligned}/4 micro={micro_e['action']} "
        f"mom={mom_e['consistency']:.2f} -> {'AGREE' if agrees else 'DISAGREE'}"
    )
    return agrees, reason


# -------------------------------------------------------
# DECISION ENGINE
# -------------------------------------------------------
def make_decision(
    df: pd.DataFrame,
    calibrator: OnlineCalibrator,
    martingale: MartingaleManager,
    thresholds: AdaptiveThresholds,
    df_entry: pd.DataFrame = None,
    min_entry_candles: int = 10,
) -> dict:
    regime = detect_regime(df)
    dp     = predict_direction(df)
    rl     = RLAgent.__new__(RLAgent)   # caller passes the real agent via closure — see below

    # NOTE: actual rl is injected by SymbolBot._on_new_candle; this function
    # is called with a pre-computed rl dict via make_decision_with_rl()
    raise NotImplementedError("Use make_decision_with_rl() instead.")


def make_decision_with_rl(
    df: pd.DataFrame,
    calibrator: OnlineCalibrator,
    martingale: MartingaleManager,
    thresholds: AdaptiveThresholds,
    rl_agent_instance: RLAgent,
    df_entry: pd.DataFrame = None,
    min_entry_candles: int = 10,
) -> dict:
    """Decision engine — identical logic to original but takes per-symbol RLAgent."""
    regime = detect_regime(df)
    dp     = predict_direction(df)
    rl     = rl_agent_instance.predict(df, regime)
    direction = rl["direction"]

    dp_agrees = (dp["direction"] == direction)
    if not dp_agrees:
        if dp["confidence"] > rl["confidence"] + 0.05:
            direction = dp["direction"]

    micro = micro_model(df, direction)
    micro_prob = micro["entry_prob"]

    tf_agrees, tf_reason = _entry_tf_agrees(df_entry, direction, min_entry_candles)
    if not tf_agrees and regime not in ("QUIET",):
        return {
            "regime": regime, "direction": direction,
            "rl_confidence": rl["confidence"],
            "micro_prob": micro_prob, "final_prob": 0.0,
            "stake": 0.0, "decision": "SKIP",
            "dp_votes": dp["votes"], "dp_conf": dp["confidence"],
            "tf_reason": tf_reason,
            "reasoning": f"Dual-TF disagreement: {tf_reason}"
        }

    raw_prob  = fuse_probabilities(rl["confidence"], micro_prob, regime, dp_conf=dp["confidence"])
    cal_prob  = calibrator.calibrate(raw_prob)
    mart_stake = martingale.get_stake()
    stake, size_label = compute_position_size(cal_prob, mart_stake, thresholds)
    decision  = "EXECUTE" if stake > 0.0 else "SKIP"

    reasoning = (
        f"Regime={regime} | RL: {direction} conf={rl['confidence']:.3f} | "
        f"DP: {dp['direction']} conf={dp['confidence']:.3f} votes={dp['votes']}/5 "
        f"z={dp.get('z_score',0):.2f} | "
        f"Micro: {micro['action']} prob={micro_prob:.3f} | "
        f"Fused={raw_prob:.3f} Cal={cal_prob:.3f} | "
        f"{tf_reason} | Size={size_label} Stake=${stake:.2f}"
    )
    return {
        "regime":        regime,
        "direction":     direction,
        "rl_confidence": round(rl["confidence"], 4),
        "micro_prob":    round(micro_prob, 4),
        "raw_prob":      round(raw_prob, 4),
        "final_prob":    round(cal_prob, 4),
        "dp_votes":      dp["votes"],
        "dp_conf":       round(dp["confidence"], 4),
        "dp_signals":    dp["signals"],
        "stake":         stake,
        "size_label":    size_label,
        "decision":      decision,
        "mart_status":   martingale.status,
        "tf_reason":     tf_reason,
        "reasoning":     reasoning,
    }


# -------------------------------------------------------
# BOT STATE  (per symbol)
# -------------------------------------------------------
class BotState:
    def __init__(self, symbol: str):
        self.symbol              = symbol
        self.balance             = 0.0
        self.start_balance       = 0.0
        self.daily_pnl           = 0.0
        self._daily_pnl_date     = datetime.now(timezone.utc).date()
        self.total_trades        = 0
        self.wins                = 0
        self.losses              = 0
        self.consecutive_losses  = 0
        self.kill_switch_until:  Optional[float] = None
        self.open_contract_id:   Optional[str]   = None
        self.open_direction:     Optional[str]   = None
        self.open_stake:         float           = 0.0
        self.open_prob:          float           = 0.0

    def maybe_reset_daily_pnl(self):
        today = datetime.now(timezone.utc).date()
        if today != self._daily_pnl_date:
            sym_log(self.symbol, logging.INFO,
                    f"New UTC day — resetting daily PnL (was ${self.daily_pnl:+.2f})")
            self.daily_pnl       = 0.0
            self._daily_pnl_date = today
            if self.kill_switch_active:
                sym_log(self.symbol, logging.INFO, "Kill switch cleared on new trading day.")
                self.kill_switch_until = None

    @property
    def win_rate(self) -> float:
        return self.wins / self.total_trades if self.total_trades > 0 else 0.0

    @property
    def kill_switch_active(self) -> bool:
        if self.kill_switch_until is None:
            return False
        return time.time() < self.kill_switch_until

    def activate_kill_switch(self, reason: str):
        pause = KILL_SWITCH_PAUSE_MIN * 60
        self.kill_switch_until = time.time() + pause
        resume = datetime.fromtimestamp(self.kill_switch_until).strftime("%H:%M:%S")
        sym_log(self.symbol, logging.CRITICAL,
                f"KILL SWITCH — {reason} | Resume at {resume}")

    def summary(self) -> str:
        wr = f"{self.win_rate*100:.1f}%" if self.total_trades > 0 else "N/A"
        return (
            f"Balance=${self.balance:.2f} | Daily PnL=${self.daily_pnl:+.2f} | "
            f"Trades={self.total_trades} W={self.wins} L={self.losses} WR={wr}"
        )


# -------------------------------------------------------
# SYMBOL BOT
# One instance per symbol. All share the same ws / req_id counter
# through SharedConnection, below.
# -------------------------------------------------------
class SymbolBot:
    def __init__(self, symbol: str, shared: "SharedConnection"):
        cfg = SYMBOL_CONFIGS[symbol]
        self.symbol              = symbol
        self.display_name        = cfg["display_name"]
        self.gran_main           = cfg["granularity_main"]
        self.gran_entry          = cfg["granularity_entry"]
        self.expiry_seconds      = cfg["expiry_seconds"]
        self.candle_history_entry = cfg["candle_history_entry"]
        self.min_entry_candles   = max(10, self.candle_history_entry // 6)

        # Determine if main and entry timeframes are the same
        self.dual_tf = (self.gran_main != self.gran_entry)

        self.shared      = shared
        self.candles     = CandleStore(maxlen=CANDLE_HISTORY, min_candles=MIN_CANDLES)
        self.candles_entry = CandleStore(
            maxlen=self.candle_history_entry,
            min_candles=self.min_entry_candles,
        )
        self.calibrator  = OnlineCalibrator()
        self.martingale  = MartingaleManager()
        self.thresholds  = AdaptiveThresholds()
        self.rl_agent    = RLAgent()
        self.state       = BotState(symbol)

        self._last_main_epoch  = 0
        self._last_entry_epoch = 0
        self._in_trade         = False
        self._trade_lock       = asyncio.Lock()
        self._push_tasks: set  = set()
        self._watchdog_task: Optional[asyncio.Task] = None

    def _log(self, level: int, msg: str):
        sym_log(self.symbol, level, msg)

    # ---- Trade lock helpers -----------------------------------------------

    def _release_trade_lock(self):
        self._in_trade = False

    # ---- Historical candles -----------------------------------------------

    async def fetch_history(self):
        self._log(logging.INFO,
                  f"Fetching {CANDLE_HISTORY}x {self.gran_main}s candles...")
        resp = await self.shared.send({
            "ticks_history": self.symbol,
            "style":         "candles",
            "granularity":   self.gran_main,
            "count":         CANDLE_HISTORY,
            "end":           "latest",
        })
        if "error" in resp:
            raise RuntimeError(
                f"{self.symbol} history failed: {resp['error']['message']}")
        for c in resp.get("candles", []):
            self.candles.add(c["epoch"], c["open"], c["high"], c["low"], c["close"])
        if resp.get("candles"):
            self._last_main_epoch = int(resp["candles"][-1]["epoch"])
        self._log(logging.INFO,
                  f"Loaded {len(resp.get('candles', []))} main-TF candles.")

        if self.dual_tf:
            self._log(logging.INFO,
                      f"Fetching {self.candle_history_entry}x {self.gran_entry}s entry candles...")
            resp_e = await self.shared.send({
                "ticks_history": self.symbol,
                "style":         "candles",
                "granularity":   self.gran_entry,
                "count":         self.candle_history_entry,
                "end":           "latest",
            })
            if "error" not in resp_e:
                for c in resp_e.get("candles", []):
                    self.candles_entry.add(
                        c["epoch"], c["open"], c["high"], c["low"], c["close"])
                if resp_e.get("candles"):
                    self._last_entry_epoch = int(resp_e["candles"][-1]["epoch"])
                self._log(logging.INFO,
                          f"Loaded {len(resp_e.get('candles', []))} entry-TF candles.")
            else:
                self._log(logging.WARNING,
                          f"Entry-TF history failed: {resp_e['error']['message']}")

        # RL warmup on historical data
        df = self.candles.df()
        if len(df) >= MIN_CANDLES + 1:
            self.rl_agent.warmup(df)
            self._log(logging.INFO, "RL warmup complete.")

    # ---- Subscribe to live candles ----------------------------------------

    async def subscribe_candles(self):
        self._log(logging.INFO, f"Subscribing to {self.gran_main}s candles...")
        resp = await self.shared.send({
            "ticks_history": self.symbol,
            "style":         "candles",
            "granularity":   self.gran_main,
            "subscribe":     1,
            "end":           "latest",
            "count":         1,
        })
        if "error" in resp:
            raise RuntimeError(
                f"{self.symbol} subscription failed: {resp['error']['message']}")
        self._log(logging.INFO, f"Subscribed to main-TF ({self.gran_main}s).")

        if self.dual_tf:
            resp_e = await self.shared.send({
                "ticks_history": self.symbol,
                "style":         "candles",
                "granularity":   self.gran_entry,
                "subscribe":     1,
                "end":           "latest",
                "count":         1,
            })
            if "error" in resp_e:
                self._log(logging.WARNING,
                          f"Entry-TF subscription failed: {resp_e['error']['message']}")
            else:
                self._log(logging.INFO, f"Subscribed to entry-TF ({self.gran_entry}s).")

    # ---- Handle incoming OHLC push ----------------------------------------

    def on_ohlc(self, ohlc: dict):
        """Called by SharedConnection when an OHLC push arrives for this symbol."""
        granularity = int(ohlc.get("granularity", self.gran_main))
        open_epoch  = int(ohlc.get("open_time", int(ohlc.get("epoch", 0))))

        if granularity == self.gran_entry and self.dual_tf:
            # ---- entry-timeframe update ----
            if open_epoch > self._last_entry_epoch:
                self._last_entry_epoch = open_epoch
                self.candles_entry.add(
                    open_epoch,
                    ohlc.get("open"), ohlc.get("high"),
                    ohlc.get("low"),  ohlc.get("close"),
                    ohlc.get("volume", 0),
                )
            else:
                if self.candles_entry.candles:
                    last = self.candles_entry.candles[-1]
                    if last["epoch"] == open_epoch:
                        last["high"]  = max(last["high"],  float(ohlc.get("high",  last["high"])))
                        last["low"]   = min(last["low"],   float(ohlc.get("low",   last["low"])))
                        last["close"] = float(ohlc.get("close", last["close"]))

        if granularity == self.gran_main or (not self.dual_tf):
            # ---- main-timeframe update ----
            if open_epoch > self._last_main_epoch:
                self._last_main_epoch = open_epoch
                self.candles.add(
                    open_epoch,
                    ohlc.get("open"), ohlc.get("high"),
                    ohlc.get("low"),  ohlc.get("close"),
                    ohlc.get("volume", 0),
                )
                # Fire analysis on confirmed candle close
                task = asyncio.create_task(self._on_new_candle())
                self._push_tasks.add(task)
                task.add_done_callback(self._push_tasks.discard)
            else:
                if self.candles.candles:
                    last = self.candles.candles[-1]
                    if last["epoch"] == open_epoch:
                        last["high"]   = max(last["high"],  float(ohlc.get("high",  last["high"])))
                        last["low"]    = min(last["low"],   float(ohlc.get("low",   last["low"])))
                        last["close"]  = float(ohlc.get("close", last["close"]))
                        last["volume"] = float(ohlc.get("volume", last["volume"]))

    # ---- New candle analysis ----------------------------------------------

    async def _on_new_candle(self):
        self.state.maybe_reset_daily_pnl()

        if not self.candles.ready():
            self._log(logging.DEBUG,
                      f"Building history... ({len(self.candles.candles)}/{MIN_CANDLES})")
            return

        if self._in_trade:
            self._log(logging.DEBUG, "Contract open — waiting for outcome.")
            return

        if self.state.kill_switch_active:
            resume = datetime.fromtimestamp(
                self.state.kill_switch_until).strftime("%H:%M:%S")
            self._log(logging.WARNING, f"Kill switch active — paused until {resume}")
            return

        if global_risk.is_killed:
            self._log(logging.WARNING, "Global kill switch active — no trade.")
            return

        df      = self.candles.df()
        df_entry = (self.candles_entry.df()
                    if len(self.candles_entry.candles) >= self.min_entry_candles
                    else None)

        self._log(logging.INFO, "-" * 60)
        self._log(logging.INFO,
                  f"Candle close | {datetime.now().strftime('%H:%M:%S')} | "
                  f"Close={df['close'].iloc[-1]:.5f} | "
                  f"Global: {global_risk.status()}")

        decision = make_decision_with_rl(
            df, self.calibrator, self.martingale, self.thresholds,
            self.rl_agent, df_entry, self.min_entry_candles,
        )
        self._log_decision(decision)

        if decision["decision"] == "EXECUTE":
            await self.place_trade(
                decision["direction"],
                decision["stake"],
                decision["final_prob"],
            )
        else:
            self._log(logging.INFO,
                      f"SKIP — {decision['reasoning'][:120]}")

    # ---- Place trade ------------------------------------------------------

    async def place_trade(self, direction: str, stake: float, final_prob: float):
        async with self._trade_lock:
            if self._in_trade:
                self._log(logging.DEBUG, "place_trade skipped — already in trade")
                return
            # Ask GlobalRiskManager before locking in
            allowed = await global_risk.request_trade()
            if not allowed:
                self._log(logging.WARNING, "Trade blocked by global risk manager.")
                return
            self._in_trade = True

        try:
            await self._execute_trade(direction, stake, final_prob)
        except Exception as e:
            self._log(logging.ERROR, f"place_trade unhandled exception: {e}")
            await global_risk.release_trade(0.0)
            self._release_trade_lock()

    async def _execute_trade(self, direction: str, stake: float, final_prob: float):
        contract_type = "CALL" if direction == "BUY" else "PUT"
        self._log(logging.INFO,
                  f"Placing {direction} ({contract_type}) | "
                  f"Stake=${stake:.2f} | Prob={final_prob:.3f} | "
                  f"{self.martingale.status}")

        expiry_m = self.expiry_seconds // 60

        try:
            proposal_resp = await self.shared.send({
                "proposal":      1,
                "amount":        float(stake),
                "basis":         "stake",
                "contract_type": contract_type,
                "currency":      "USD",
                "duration":      expiry_m,
                "duration_unit": "m",
                "symbol":        self.symbol,
            })
        except asyncio.TimeoutError:
            self._log(logging.ERROR, "Proposal timed out.")
            await global_risk.release_trade(0.0)
            self._release_trade_lock()
            return

        if "error" in proposal_resp:
            self._log(logging.ERROR,
                      f"Proposal error: {proposal_resp['error']['message']}")
            await global_risk.release_trade(0.0)
            self._release_trade_lock()
            return

        proposal_id = proposal_resp["proposal"]["id"]
        ask_price   = float(proposal_resp["proposal"]["ask_price"])

        try:
            buy_resp = await self.shared.send({
                "buy":   proposal_id,
                "price": float(ask_price),
            })
        except asyncio.TimeoutError:
            self._log(logging.ERROR, "Buy request timed out.")
            await global_risk.release_trade(0.0)
            self._release_trade_lock()
            return

        if "error" in buy_resp:
            self._log(logging.ERROR,
                      f"Buy error: {buy_resp['error']['message']}")
            await global_risk.release_trade(0.0)
            self._release_trade_lock()
            return

        contract = buy_resp["buy"]
        self.state.open_contract_id = str(contract["contract_id"])
        self.state.open_direction   = direction
        self.state.open_stake       = stake
        self.state.open_prob        = final_prob
        self.state.total_trades    += 1

        self._log(logging.INFO,
                  f"Contract opened | ID={self.state.open_contract_id} | "
                  f"Payout=${float(contract.get('payout', 0)):.2f} | "
                  f"Balance=${float(contract.get('balance_after', 0)):.2f}")

        # Subscribe to settlement push
        cid_int = int(self.state.open_contract_id)
        try:
            await self.shared.ws.send(json.dumps({
                "proposal_open_contract": 1,
                "contract_id": cid_int,
                "subscribe":   1,
                "req_id":      self.shared.next_req_id(),
            }))
        except Exception as sub_err:
            self._log(logging.WARNING,
                      f"Settlement subscription failed: {sub_err} — watchdog will poll.")

        # Start settlement watchdog
        if self._watchdog_task and not self._watchdog_task.done():
            self._watchdog_task.cancel()
        self._watchdog_task = asyncio.create_task(
            self._settlement_watchdog(self.state.open_contract_id)
        )

    # ---- Contract update handler ------------------------------------------

    async def on_contract_update(self, poc: dict):
        """Called by SharedConnection when a settlement push arrives."""
        status = poc.get("status", "")
        if status not in SETTLEMENT_TERMINAL:
            return

        cid    = str(poc.get("contract_id", ""))
        profit = float(poc.get("profit", 0))

        if cid != self.state.open_contract_id:
            return

        if self._watchdog_task and not self._watchdog_task.done():
            self._watchdog_task.cancel()
            self._watchdog_task = None

        if status not in ("won", "lost"):
            self._log(logging.WARNING,
                      f"Contract {cid} closed as '{status}' (no PnL). Releasing.")
            self.state.open_contract_id = None
            await global_risk.release_trade(0.0)
            self._release_trade_lock()
            return

        won = (status == "won")
        self.state.daily_pnl        += profit
        self.state.open_contract_id  = None

        try:
            self.calibrator.record(self.state.open_prob, 1 if won else 0)
            self.thresholds.record(won, profit)

            df_now = self.candles.df()
            if len(df_now) >= MIN_CANDLES:
                regime   = detect_regime(df_now)
                features = RLAgent._extract_features(df_now, regime)
                self.rl_agent.update(features, 1 if won else 0)

            if won:
                self.state.wins               += 1
                self.state.consecutive_losses  = 0
                self.martingale.record_win()
                self._log(logging.INFO,
                          Fore.GREEN + Style.BRIGHT +
                          f"WIN  | Profit=${profit:+.2f} | {self.state.summary()} | "
                          f"Cal: {self.calibrator.stats}")
            else:
                self.state.losses             += 1
                self.state.consecutive_losses += 1
                self.martingale.record_loss()
                self._log(logging.WARNING,
                          Fore.RED +
                          f"LOSS | Profit=${profit:+.2f} | {self.state.summary()} | "
                          f"Martingale: {self.martingale.status}")
                if self.state.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                    self.state.activate_kill_switch(
                        f"{MAX_CONSECUTIVE_LOSSES} consecutive losses")

            if self.state.daily_pnl <= -MAX_DAILY_LOSS:
                self.state.activate_kill_switch(
                    f"Daily loss limit ${MAX_DAILY_LOSS:.2f} hit "
                    f"(PnL=${self.state.daily_pnl:.2f})")

        finally:
            await global_risk.release_trade(profit)
            self._release_trade_lock()

    # ---- Settlement watchdog ----------------------------------------------

    async def _settlement_watchdog(self, contract_id: str):
        wait_secs = self.expiry_seconds + SETTLEMENT_GRACE_SECONDS
        self._log(logging.DEBUG,
                  f"[Watchdog] Polling in {wait_secs}s if no push for {contract_id}.")
        try:
            await asyncio.sleep(wait_secs)
        except asyncio.CancelledError:
            return

        if not self._in_trade or self.state.open_contract_id != contract_id:
            return

        self._log(logging.WARNING,
                  f"[Watchdog] No settlement push after {wait_secs}s — polling API.")

        for attempt in range(1, SETTLEMENT_POLL_MAX_TRIES + 1):
            try:
                if self.shared.ws is None:
                    await asyncio.sleep(SETTLEMENT_POLL_INTERVAL)
                    continue
                resp = await asyncio.wait_for(
                    self.shared.send({
                        "proposal_open_contract": 1,
                        "contract_id": int(contract_id),
                    }),
                    timeout=15,
                )
            except Exception as e:
                self._log(logging.WARNING,
                          f"[Watchdog] Attempt {attempt} poll failed: {e}")
                if attempt < SETTLEMENT_POLL_MAX_TRIES:
                    await asyncio.sleep(SETTLEMENT_POLL_INTERVAL)
                continue

            poc    = resp.get("proposal_open_contract", {})
            status = poc.get("status", "unknown")

            if status in ("won", "lost"):
                self._log(logging.INFO,
                          f"[Watchdog] Contract {contract_id} resolved: {status.upper()}")
                await self.on_contract_update(poc)
                return
            elif status in SETTLEMENT_TERMINAL:
                self._log(logging.WARNING,
                          f"[Watchdog] Terminal status '{status}' — releasing lock.")
                self.state.open_contract_id = None
                await global_risk.release_trade(0.0)
                self._release_trade_lock()
                return
            elif status == "open":
                self._log(logging.WARNING,
                          f"[Watchdog] Attempt {attempt}: still open — retrying...")
                if attempt < SETTLEMENT_POLL_MAX_TRIES:
                    await asyncio.sleep(SETTLEMENT_POLL_INTERVAL)
            else:
                self._log(logging.ERROR,
                          f"[Watchdog] Unknown status '{status}'.")
                if attempt < SETTLEMENT_POLL_MAX_TRIES:
                    await asyncio.sleep(SETTLEMENT_POLL_INTERVAL)

        # Exhausted retries — count as loss
        self._log(logging.ERROR,
                  f"[Watchdog] Settlement unknown after {SETTLEMENT_POLL_MAX_TRIES} polls. Counting loss.")
        if self.state.open_contract_id == contract_id:
            self.state.losses             += 1
            self.state.consecutive_losses += 1
            self.martingale.record_loss()
            self.calibrator.record(self.state.open_prob, 0)
            self.thresholds.record(False, -self.state.open_stake)
            self.state.open_contract_id = None
            if self.state.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                self.state.activate_kill_switch("Settlement watchdog: forced loss")
        await global_risk.release_trade(-self.state.open_stake)
        self._release_trade_lock()

    # ---- Reconnect requery ------------------------------------------------

    async def requery_open_contract(self):
        cid = self.state.open_contract_id
        if not cid:
            return
        self._log(logging.WARNING,
                  f"Open contract {cid} found after reconnect — querying...")
        try:
            resp = await asyncio.wait_for(
                self.shared.send({
                    "proposal_open_contract": 1,
                    "contract_id": int(cid),
                }),
                timeout=CONTRACT_REQUERY_TIMEOUT,
            )
        except Exception as e:
            self._log(logging.ERROR,
                      f"Requery failed: {e} — counting as loss.")
            self.state.losses             += 1
            self.state.consecutive_losses += 1
            self.martingale.record_loss()
            self.calibrator.record(self.state.open_prob, 0)
            self.state.open_contract_id = None
            await global_risk.release_trade(-self.state.open_stake)
            self._release_trade_lock()
            return

        poc    = resp.get("proposal_open_contract", {})
        status = poc.get("status", "unknown")
        if status in ("won", "lost"):
            self._log(logging.INFO, f"Requery resolved contract {cid} as {status.upper()}")
            await self.on_contract_update(poc)
        elif status == "open":
            self._log(logging.INFO,
                      f"Contract {cid} still open — re-subscribing...")
            await self.shared.ws.send(json.dumps({
                "proposal_open_contract": 1,
                "contract_id": int(cid),
                "subscribe":   1,
                "req_id":      self.shared.next_req_id(),
            }))
        else:
            self._log(logging.WARNING,
                      f"Requery unknown status '{status}' — releasing lock.")
            self.state.open_contract_id = None
            await global_risk.release_trade(0.0)
            self._release_trade_lock()

    def cancel_push_tasks(self):
        for task in list(self._push_tasks):
            task.cancel()
        self._push_tasks.clear()
        if self._watchdog_task and not self._watchdog_task.done():
            self._watchdog_task.cancel()
            self._watchdog_task = None

    # ---- Decision logger --------------------------------------------------

    def _log_decision(self, d: dict):
        regime    = d["regime"]
        direction = d["direction"]
        decision  = d["decision"]
        regime_colors = {
            "EXPANDING":  Fore.CYAN,
            "TRENDING":   Fore.GREEN,
            "EXHAUSTION": Fore.YELLOW,
            "QUIET":      Fore.WHITE,
        }
        dir_color = Fore.GREEN if direction == "BUY" else Fore.RED
        dec_color = Fore.GREEN if decision == "EXECUTE" else Fore.YELLOW
        rc        = regime_colors.get(regime, Fore.WHITE)
        self._log(logging.INFO, f"  Regime    : {rc}{regime}{Style.RESET_ALL}")
        self._log(logging.INFO, f"  Direction : {dir_color}{direction}{Style.RESET_ALL}")
        self._log(logging.INFO, f"  RL Conf   : {d.get('rl_confidence', 0):.4f}")
        self._log(logging.INFO, f"  Micro     : {d.get('micro_prob', 0):.4f}")
        self._log(logging.INFO, f"  Final Prob: {d.get('final_prob', 0):.4f}")
        self._log(logging.INFO, f"  TF        : {d.get('tf_reason', 'N/A')}")
        self._log(logging.INFO, f"  Thresholds: {self.thresholds.status}")
        self._log(logging.INFO, f"  Stake     : ${d.get('stake', 0):.2f}  [{d.get('size_label', 'N/A')}]")
        self._log(logging.INFO, f"  Martingale: {d.get('mart_status', 'N/A')}")
        self._log(logging.INFO, f"  Decision  : {dec_color}{decision}{Style.RESET_ALL}")


# -------------------------------------------------------
# SHARED CONNECTION
# One WebSocket shared by all SymbolBots.
# Routes OHLC and contract pushes to the right bot.
# -------------------------------------------------------
class SharedConnection:
    def __init__(self, bots: Dict[str, SymbolBot]):
        self.bots    = bots          # symbol -> SymbolBot
        self.ws      = None
        self._req_id = 1
        self._req_id_lock = asyncio.Lock()
        self._pending: dict = {}
        self._authorized    = False
        self.balance        = 0.0

        # Map contract_id -> symbol for routing settlement pushes
        self._contract_to_symbol: Dict[str, str] = {}

    def next_req_id(self) -> int:
        rid = self._req_id
        self._req_id += 1
        return rid

    async def send(self, payload: dict) -> dict:
        loop   = asyncio.get_running_loop()
        rid    = self.next_req_id()
        payload["req_id"] = rid
        future = loop.create_future()
        self._pending[rid] = future
        await self.ws.send(json.dumps(payload))
        try:
            return await asyncio.wait_for(future, timeout=30)
        except asyncio.TimeoutError:
            self._pending.pop(rid, None)
            raise

    async def _recv_loop(self):
        try:
            async for raw in self.ws:
                msg      = json.loads(raw)
                rid      = msg.get("req_id")
                msg_type = msg.get("msg_type")

                # Route RPC responses
                if rid and rid in self._pending:
                    fut = self._pending.pop(rid)
                    if not fut.done():
                        fut.set_result(msg)
                    continue

                # Route push messages
                if msg_type == "ohlc":
                    ohlc   = msg.get("ohlc", {})
                    symbol = ohlc.get("symbol", "")
                    bot    = self.bots.get(symbol)
                    if bot:
                        bot.on_ohlc(ohlc)

                elif msg_type == "proposal_open_contract":
                    poc    = msg.get("proposal_open_contract", {})
                    cid    = str(poc.get("contract_id", ""))
                    symbol = self._contract_to_symbol.get(cid)
                    if not symbol:
                        # Try to match by checking all bots
                        for sym, bot in self.bots.items():
                            if bot.state.open_contract_id == cid:
                                symbol = sym
                                self._contract_to_symbol[cid] = sym
                                break
                    if symbol:
                        bot = self.bots.get(symbol)
                        if bot:
                            asyncio.create_task(bot.on_contract_update(poc))

                elif msg_type == "error":
                    logger.error(f"Server error push: {msg.get('error', {}).get('message', msg)}")

        except ConnectionClosed as exc:
            logger.warning(
                f"WebSocket closed (code={exc.rcvd.code if exc.rcvd else 'N/A'} "
                f"reason={exc.rcvd.reason if exc.rcvd else str(exc)}) — reconnecting..."
            )
            raise ConnectionError(str(exc)) from exc

    async def authorize(self):
        logger.info("Authorizing with Deriv API...")
        resp = await self.send({"authorize": DERIV_API_TOKEN})
        if "error" in resp:
            raise RuntimeError(f"Authorization failed: {resp['error']['message']}")
        account      = resp["authorize"]
        self.balance = float(account.get("balance", 0))
        self._authorized = True
        logger.info(
            f"Authorized | Account: {account.get('loginid')} | "
            f"Balance: ${self.balance:.2f} | Currency: {account.get('currency')}"
        )
        # Share balance with all bots
        for bot in self.bots.values():
            bot.state.balance       = self.balance
            bot.state.start_balance = self.balance

    def reset_connection(self):
        for fut in self._pending.values():
            if not fut.done():
                fut.cancel()
        self._pending.clear()
        self._authorized = False
        self.ws = None
        for bot in self.bots.values():
            bot.cancel_push_tasks()


# -------------------------------------------------------
# BANNER
# -------------------------------------------------------
def print_banner(symbols):
    print(Fore.YELLOW + Style.BRIGHT + """
+--------------------------------------------------------------+
|      MULTI-SYMBOL ADAPTIVE TRADING BOT  -  DERIV            |
|  Parallel per-symbol | RL + Micro + Fusion | Adaptive TF    |
+--------------------------------------------------------------+
""" + Style.RESET_ALL)
    logger.info(f"Mode              : {TRADE_MODE.upper()}")
    logger.info(f"Active symbols    : {len(symbols)}")
    for sym in symbols:
        cfg = SYMBOL_CONFIGS[sym]
        logger.info(
            f"  {sym:<14} {cfg['display_name']:<22} "
            f"main={cfg['granularity_main']}s  "
            f"entry={cfg['granularity_entry']}s  "
            f"expiry={cfg['expiry_seconds']}s"
        )
    logger.info(f"Base stake        : ${BASE_STAKE}")
    logger.info(f"Max daily loss    : ${MAX_DAILY_LOSS} per symbol")
    logger.info(f"Global daily loss : ${MAX_GLOBAL_DAILY_LOSS}")
    logger.info(f"Max concurrent    : {MAX_CONCURRENT_TRADES}")
    logger.info(f"Log file          : {log_filename}")
    print()


# -------------------------------------------------------
# MAIN RUN LOOP
# -------------------------------------------------------
async def run():
    # Validate ACTIVE_SYMBOLS
    for sym in ACTIVE_SYMBOLS:
        if sym not in SYMBOL_CONFIGS:
            raise ValueError(
                f"Symbol '{sym}' not in SYMBOL_CONFIGS. "
                f"Add it or remove it from ACTIVE_SYMBOLS."
            )

    print_banner(ACTIVE_SYMBOLS)

    reconnect_delay = RECONNECT_DELAY_MIN

    while True:
        # Build fresh bot instances each reconnect cycle (preserves state across reconnects)
        bots   = {sym: SymbolBot(sym, None) for sym in ACTIVE_SYMBOLS}  # type: ignore
        shared = SharedConnection(bots)
        for bot in bots.values():
            bot.shared = shared

        logger.info("Connecting to Deriv WebSocket...")
        try:
            async with websockets.connect(
                WS_URL,
                ping_interval=WS_PING_INTERVAL,
                ping_timeout=WS_PING_TIMEOUT,
            ) as ws:
                shared.ws    = ws
                recv_task    = asyncio.create_task(shared._recv_loop())

                await shared.authorize()

                # Fetch history and subscribe for every symbol
                # Run sequentially to avoid hammering the API with parallel requests
                for sym, bot in bots.items():
                    try:
                        await bot.fetch_history()
                        await bot.subscribe_candles()
                        # Requery open contracts after reconnect
                        if bot._in_trade and bot.state.open_contract_id:
                            await bot.requery_open_contract()
                    except Exception as e:
                        logger.error(
                            f"Init failed for {sym}: {e} — skipping this symbol.")

                logger.info(
                    f"All symbols live. "
                    f"Global limits: "
                    f"concurrent={MAX_CONCURRENT_TRADES} "
                    f"global_loss=${MAX_GLOBAL_DAILY_LOSS}"
                )

                reconnect_delay = RECONNECT_DELAY_MIN  # reset backoff on success

                try:
                    await recv_task
                except asyncio.CancelledError:
                    pass
                except ConnectionError:
                    recv_task.cancel()
                except Exception as e:
                    logger.critical(f"Fatal recv loop error: {e}", exc_info=True)
                    recv_task.cancel()

        except KeyboardInterrupt:
            logger.info("Bot stopped by user.")
            return
        except Exception as e:
            logger.error(
                f"Connection error: {e} — "
                f"reconnecting in {reconnect_delay}s...")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, RECONNECT_DELAY_MAX)


# -------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------
async def main():
    if DERIV_API_TOKEN == "YOUR_TOKEN_HERE":
        logger.critical("No API token set! Set DERIV_API_TOKEN env variable.")
        logger.critical("  Example: export DERIV_API_TOKEN=your_real_token")
        logger.critical("  Or create a .env file with DERIV_API_TOKEN=your_token")
        sys.exit(1)

    if not ACTIVE_SYMBOLS:
        logger.critical("ACTIVE_SYMBOLS is empty. Add symbols to trade.")
        sys.exit(1)

    await run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
