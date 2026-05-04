"""
XAU/USD ADAPTIVE TRADING BOT - DERIV PLATFORM
5-Minute Expiry | RL + Micro Model + Probabilistic Fusion

SETUP:
  pip install websockets pandas numpy scipy colorama python-dotenv

USAGE:
  python xauusd_bot.py

ENVIRONMENT (.env file or export):
  DERIV_API_TOKEN=your_api_token_here
  DERIV_APP_ID=1089          # Use 1089 for demo, your app ID for live
  TRADE_MODE=demo            # demo | live
  BASE_STAKE=0.5             # Minimum stake in USD
  MAX_DAILY_LOSS=10.0        # Kill switch: max $ loss per day

BUGS FIXED (11 original + 6 resilience improvements):
  1.  Trade lock set BEFORE any await in place_trade() to prevent race.
  2.  Buy/proposal failure calls release_trade_lock() not on_close().
  3.  API amount/price sent as float (number), not str.
  4.  subscribe_candles() uses self.send() (rpc path) so errors surface.
  5.  _settle finally always ensures _in_trade=False via release path.
  6.  _on_new_candle only fires on new epoch; _handle_push runs as task.
  7.  asyncio.get_event_loop() replaced with asyncio.get_running_loop().
  8.  Unicode/emoji removed; stdout reconfigured for UTF-8 safety.
  9.  _execute race: asyncio.Lock() guards entry, set before first await.
  10. _settle finally always ensures lock released cleanly.
  11. Reconnect releases trade lock before resetting state.
  12. Reconnect resets connection state (pending futures, push tasks, ws ref)
      so stale futures from the dead socket never block new requests.
  13. Open contract on reconnect is requeried from Deriv API; outcome is
      settled properly (win/loss/martingale/calibrator) before trading resumes.
  14. Exponential backoff on reconnect (5s -> 10s -> 20s ... cap 120s).
  15. ping_timeout added to websockets.connect() so a silent unresponsive
      server forces a proper disconnect instead of hanging forever.
  16. Background push-handler tasks are tracked in _push_tasks and cancelled
      cleanly on reconnect so no zombie coroutines remain.
  17. fetch_history failure is now fatal (raises) so the bot never runs
      with an empty candle store silently doing nothing useful.
  18. Daily PnL resets at UTC midnight so the kill switch does not carry
      over yesterday's losses into the new trading day.
  19. RL warmup training on the 50 historical candles before live trading
      starts. The RL agent now carries a lightweight weight vector that is
      gradient-updated (mini-batch gradient descent) on the historical candle
      outcomes as supervised labels, so the agent enters live trading with
      calibrated weights rather than fixed hand-tuned constants.
  20. Keepalive ping timeout (websockets error 1011) is now caught explicitly
      in the recv loop and treated as a clean reconnect trigger. Previously
      the exception propagated as a "Fatal error" log; now it is demoted to
      a WARNING and the normal exponential-backoff reconnect loop takes over
      silently.
"""

# FIX #8 - stdout reconfiguration before any imports that write output
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
from typing import Optional
import websockets
from websockets.exceptions import ConnectionClosed
import pandas as pd
import numpy as np
from scipy.special import expit  # sigmoid
from colorama import init, Fore, Style

init(autoreset=True)

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DERIV_API_TOKEN  = os.getenv("DERIV_API_TOKEN", "YOUR_TOKEN_HERE")
DERIV_APP_ID     = os.getenv("DERIV_APP_ID", "1089")
TRADE_MODE       = os.getenv("TRADE_MODE", "demo")
BASE_STAKE       = float(os.getenv("BASE_STAKE", "0.5"))
MAX_DAILY_LOSS   = float(os.getenv("MAX_DAILY_LOSS", "10.0"))

SYMBOL           = "frxXAUUSD"
GRANULARITY      = 300
EXPIRY_SECONDS   = 300
CANDLE_HISTORY   = 50
MIN_CANDLES      = 20

MARTINGALE_MULTIPLIER  = 1.78
MARTINGALE_MAX_STEPS   = 5
MARTINGALE_TRIGGER     = 2

THRESHOLD_NO_TRADE     = 0.54
THRESHOLD_SMALL        = 0.65
THRESHOLD_NORMAL       = 0.75

MAX_CONSECUTIVE_LOSSES = 5
KILL_SWITCH_PAUSE_MIN  = 45

WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"

# Reconnection backoff: starts at RECONNECT_DELAY_MIN, doubles each failure
# up to RECONNECT_DELAY_MAX, resets on successful connection.
RECONNECT_DELAY_MIN = 5
RECONNECT_DELAY_MAX = 120

# WebSocket ping config - timeout forces disconnect if server goes silent
WS_PING_INTERVAL = 20
WS_PING_TIMEOUT  = 30

# How long to wait for open contract outcome after reconnect before giving up
CONTRACT_REQUERY_TIMEOUT = 15

# -------------------------------------------------------
# LOGGING SETUP  (no emoji/unicode in format strings)
# -------------------------------------------------------
log_filename = f"xauusd_bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


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


logger = logging.getLogger("XAUUSD_BOT")
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
ch.setFormatter(ColorFormatter())
logger.addHandler(ch)

fh = logging.FileHandler(log_filename, encoding="utf-8")
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
logger.addHandler(fh)


# -------------------------------------------------------
# BANNER  (ASCII only, no box-drawing or emoji)
# -------------------------------------------------------
def print_banner():
    print(Fore.YELLOW + Style.BRIGHT + """
+----------------------------------------------------------------------+
|        XAU/USD ADAPTIVE TRADING BOT  -  DERIV PLATFORM              |
|        5-Min Expiry  |  RL + Micro + Fusion  |  Auto-Calibrating    |
+----------------------------------------------------------------------+
""" + Style.RESET_ALL)
    logger.info(f"Mode        : {TRADE_MODE.upper()}")
    logger.info(f"Symbol      : {SYMBOL}")
    logger.info(f"Base Stake  : ${BASE_STAKE}")
    logger.info(
        f"Martingale  : {MARTINGALE_MULTIPLIER}x after {MARTINGALE_TRIGGER} "
        f"losses, max {MARTINGALE_MAX_STEPS} steps"
    )
    logger.info(f"Log file    : {log_filename}")
    print()


# -------------------------------------------------------
# CANDLE STORE
# -------------------------------------------------------
class CandleStore:
    def __init__(self, maxlen=CANDLE_HISTORY):
        self.candles = deque(maxlen=maxlen)

    def add(self, epoch, open_, high, low, close, volume=0):
        self.candles.append({
            "epoch": epoch, "open": float(open_), "high": float(high),
            "low": float(low), "close": float(close), "volume": float(volume)
        })

    def df(self) -> pd.DataFrame:
        return pd.DataFrame(list(self.candles))

    def ready(self) -> bool:
        return len(self.candles) >= MIN_CANDLES


# -------------------------------------------------------
# REGIME DETECTOR
# -------------------------------------------------------
def detect_regime(df: pd.DataFrame) -> str:
    atr = compute_atr(df, 14)
    if len(atr) < 5:
        return "CALM"

    current_atr = atr.iloc[-1]
    avg_atr     = atr.iloc[-14:].mean()
    atr_ratio   = current_atr / avg_atr if avg_atr > 0 else 1.0

    closes = df["close"].values
    diffs  = np.diff(closes[-9:])
    positive = np.sum(diffs > 0)
    negative = np.sum(diffs < 0)
    consistency = max(positive, negative) / len(diffs)

    if atr_ratio < 0.75:
        return "CALM"
    elif atr_ratio > 1.5 and consistency > 0.7:
        return "TRENDING"
    elif atr_ratio > 1.3:
        return "EXPANDING"
    else:
        recent_momentum = abs(closes[-1] - closes[-5])
        avg_momentum    = np.mean(np.abs(np.diff(closes[-10:]))) * 4
        if recent_momentum < avg_momentum * 0.5 and atr_ratio > 1.0:
            return "EXHAUSTION"
        return "CALM"


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

    half  = period // 2
    mom1  = closes[-half] - closes[-period]
    mom2  = closes[-1]    - closes[-half]
    acceleration = mom2 - mom1

    return {
        "strength":     float(strength),
        "consistency":  float(consistency),
        "acceleration": float(acceleration)
    }


def compute_structure(df: pd.DataFrame, lookback=20) -> dict:
    highs = df["high"].values[-lookback:]
    lows  = df["low"].values[-lookback:]
    close = df["close"].values[-1]

    swing_high = float(np.max(highs))
    swing_low  = float(np.min(lows))
    rng        = swing_high - swing_low

    dist_from_high = (swing_high - close) / rng if rng > 0 else 0.5
    dist_from_low  = (close - swing_low)  / rng if rng > 0 else 0.5

    return {
        "swing_high":     swing_high,
        "swing_low":      swing_low,
        "dist_from_high": float(dist_from_high),
        "dist_from_low":  float(dist_from_low),
        "range":          float(rng)
    }


# -------------------------------------------------------
# RL AGENT  (trainable weight vector with warmup)
# -------------------------------------------------------

# Feature names — order must match _extract_rl_features()
_RL_FEATURE_NAMES = [
    "momentum", "acceleration", "structure", "volatility", "exhaustion_flip"
]

# Default hand-tuned weights (used before/during warmup as the starting point)
_RL_DEFAULT_WEIGHTS = np.array([0.40, 0.20, 0.20, 0.10, 0.10], dtype=np.float64)


class RLAgent:
    """
    Lightweight linear RL agent with a trainable weight vector.

    Architecture:
      score  = dot(weights, features)        # linear combination
      conf   = sigmoid(score * GAIN)         # map to [0, 1]
      direction = BUY / SELL / HOLD based on conf

    Warmup training (FIX #19):
      Before live trading starts we run mini-batch gradient descent on the
      50 historical candles, using next-candle return sign as the binary label.
      This gives the agent calibrated weights rather than fixed constants.

    Online updates:
      After each settled trade the weight vector is nudged toward the actual
      outcome (win=1 / loss=0) via a single stochastic gradient step.
    """

    GAIN       = 5.0    # sigmoid sharpness
    LR_WARMUP  = 0.05   # learning rate during warmup
    LR_ONLINE  = 0.01   # learning rate during live trading
    EPOCHS     = 30     # warmup passes over the historical window
    L2         = 1e-4   # L2 regularisation (weight decay)

    def __init__(self):
        self.weights = _RL_DEFAULT_WEIGHTS.copy()
        self._warmed_up = False

    # ------------------------------------------------------------------
    # Feature extraction (same signals as the old stateless rl_agent)
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_features(df: pd.DataFrame, regime: str) -> np.ndarray:
        momentum    = compute_momentum(df, 10)
        structure   = compute_structure(df, 20)
        atr_vals    = compute_atr(df, 14)
        current_atr = float(atr_vals.iloc[-1]) if not atr_vals.empty else 0
        avg_atr     = float(atr_vals.iloc[-14:].mean()) if len(atr_vals) >= 14 else current_atr
        atr_ratio   = current_atr / avg_atr if avg_atr > 0 else 1.0

        mom_signal = np.sign(momentum["consistency"])
        f_momentum     = momentum["consistency"]
        f_acceleration = float(np.tanh(momentum["acceleration"] / (avg_atr * 0.5 + 1e-9)))

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

    # ------------------------------------------------------------------
    # Warmup: supervised training on historical candles (FIX #19)
    # ------------------------------------------------------------------
    def warmup(self, df: pd.DataFrame):
        """
        Train the weight vector on historical candles.

        Label for candle i  = 1 if close[i+1] > close[i]  (price went up)
                            = 0 otherwise
        We slide a growing window (min 20 candles) over the history,
        extract features for each window, compute the sigmoid loss, and
        backprop through the linear layer.
        """
        closes = df["close"].values
        n      = len(df)
        if n < MIN_CANDLES + 1:
            logger.warning(
                f"RL warmup skipped: only {n} candles (need {MIN_CANDLES + 1})"
            )
            return

        # Build (features, label) pairs for each trainable window
        samples = []
        for i in range(MIN_CANDLES, n - 1):
            window = df.iloc[:i + 1]
            regime = detect_regime(window)
            if regime == "CALM":
                continue                    # skip — bot won't trade in CALM anyway
            feat  = self._extract_features(window, regime)
            label = 1.0 if closes[i + 1] > closes[i] else 0.0
            samples.append((feat, label))

        if not samples:
            logger.warning("RL warmup: no non-CALM windows found in history.")
            return

        feats  = np.array([s[0] for s in samples], dtype=np.float64)
        labels = np.array([s[1] for s in samples], dtype=np.float64)
        m      = len(samples)

        logger.info(
            f"RL warmup: training on {m} historical windows "
            f"({self.EPOCHS} epochs, lr={self.LR_WARMUP})..."
        )

        for epoch in range(self.EPOCHS):
            # Shuffle each epoch
            idx  = np.random.permutation(m)
            fsh  = feats[idx]
            lsh  = labels[idx]

            # Vectorised forward + backward pass over all samples
            scores  = fsh @ self.weights * self.GAIN          # (m,)
            preds   = expit(scores)                            # sigmoid -> (m,)
            errors  = preds - lsh                              # (m,)  dL/d_score
            # gradient of BCE loss w.r.t. weights + L2 term
            grad    = (fsh.T @ errors) / m * self.GAIN + self.L2 * self.weights
            self.weights -= self.LR_WARMUP * grad

        self._warmed_up = True
        logger.info(
            f"RL warmup complete. Trained weights: "
            + " ".join(
                f"{n}={w:.4f}"
                for n, w in zip(_RL_FEATURE_NAMES, self.weights)
            )
        )

    # ------------------------------------------------------------------
    # Online update: called after each settled trade
    # ------------------------------------------------------------------
    def update(self, features: np.ndarray, outcome: int):
        """Single stochastic gradient step after a trade result."""
        score = float(np.dot(self.weights, features)) * self.GAIN
        pred  = float(expit(score))
        error = pred - float(outcome)
        grad  = features * error * self.GAIN + self.L2 * self.weights
        self.weights -= self.LR_ONLINE * grad

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self, df: pd.DataFrame, regime: str) -> dict:
        if regime == "CALM":
            return {
                "direction": "HOLD", "confidence": 0.0,
                "reason": "CALM regime - skipping",
                "features": None,
            }

        momentum    = compute_momentum(df, 10)
        atr_vals    = compute_atr(df, 14)
        current_atr = float(atr_vals.iloc[-1]) if not atr_vals.empty else 0
        avg_atr     = float(atr_vals.iloc[-14:].mean()) if len(atr_vals) >= 14 else current_atr
        atr_ratio   = current_atr / avg_atr if avg_atr > 0 else 1.0

        features   = self._extract_features(df, regime)
        raw_score  = float(np.dot(self.weights, features))
        confidence = float(expit(raw_score * self.GAIN))

        if confidence > 0.55:
            direction = "BUY"
        elif confidence < 0.45:
            direction = "SELL"
            confidence = 1.0 - confidence
        else:
            direction = "HOLD"

        if confidence < 0.55:
            direction = "HOLD"

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


# Module-level singleton — shared across make_decision() calls
_rl_agent_instance = RLAgent()


def rl_agent(df: pd.DataFrame, regime: str) -> dict:
    """Thin wrapper kept for API compatibility with make_decision()."""
    return _rl_agent_instance.predict(df, regime)


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
        breakout = (latest["close"] - prior_high) / (prior_high - prior_low + 1e-9)
        breakout = float(np.clip(breakout, 0, 1))
    elif direction == "SELL" and latest["close"] < prior_low:
        breakout = (prior_low - latest["close"]) / (prior_high - prior_low + 1e-9)
        breakout = float(np.clip(breakout, 0, 1))

    micro_mom = 0.0
    if direction == "BUY":
        micro_mom = sum(1 for c in candles[-3:] if c["close"] > c["open"]) / 3.0
    else:
        micro_mom = sum(1 for c in candles[-3:] if c["close"] < c["open"]) / 3.0

    reject_reason = None
    if direction == "BUY" and upper_wick_ratio > 0.40:
        reject_reason = f"Strong upper wick ({upper_wick_ratio:.2f}) on BUY"
    if direction == "SELL" and lower_wick_ratio > 0.40:
        reject_reason = f"Strong lower wick ({lower_wick_ratio:.2f}) on SELL"
    if body_strength > 2.5:
        reject_reason = f"Candle overextended (body_strength={body_strength:.2f})"

    atr_vals = compute_atr(df, 14)
    if not atr_vals.empty:
        avg_atr = float(atr_vals.iloc[-14:].mean())
        if candle_rng > avg_atr * 2.0:
            reject_reason = (
                f"Late entry - range spike {candle_rng:.4f} > 2x ATR {avg_atr:.4f}"
            )

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
def fuse_probabilities(rl_conf: float, micro_prob: float, regime: str) -> float:
    fused = 0.75 * rl_conf + 0.25 * micro_prob
    modifiers = {
        "TRENDING":   1.05,
        "EXPANDING":  1.02,
        "EXHAUSTION": 0.95,
        "CALM":       0.70,
    }
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
        bin_idx = np.digitize(raw_prob, bins) - 1
        bin_idx = int(np.clip(bin_idx, 0, 4))

        n = len(self.buffer)
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
        return {
            "samples":  len(self.buffer),
            "win_rate": round(sum(outcomes) / len(outcomes), 3) if outcomes else None
        }


# -------------------------------------------------------
# MARTINGALE MANAGER
# -------------------------------------------------------
class MartingaleManager:
    def __init__(self, base_stake=BASE_STAKE, multiplier=MARTINGALE_MULTIPLIER,
                 max_steps=MARTINGALE_MAX_STEPS, trigger=MARTINGALE_TRIGGER):
        self.base_stake  = base_stake
        self.multiplier  = multiplier
        self.max_steps   = max_steps
        self.trigger     = trigger
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
def compute_position_size(final_prob: float, martingale_stake: float) -> tuple:
    edge = (final_prob * 2) - 1

    if final_prob < THRESHOLD_NO_TRADE:
        return 0.0, "NO TRADE"
    elif final_prob < THRESHOLD_SMALL:
        size_label = "SMALL"
        size_mult  = 0.75
    elif final_prob < THRESHOLD_NORMAL:
        size_label = "NORMAL"
        size_mult  = 1.0
    else:
        size_label = "AGGRESSIVE"
        size_mult  = 1.25

    size = max(BASE_STAKE, martingale_stake * size_mult * edge * 2)
    size = round(max(BASE_STAKE, size), 2)
    return size, size_label


# -------------------------------------------------------
# DECISION ENGINE
# -------------------------------------------------------
def make_decision(df: pd.DataFrame, calibrator: OnlineCalibrator,
                  martingale: MartingaleManager) -> dict:
    regime    = detect_regime(df)
    rl        = rl_agent(df, regime)
    direction = rl["direction"]

    if direction == "HOLD":
        return {
            "regime": regime, "direction": "HOLD",
            "rl_confidence": rl["confidence"],
            "micro_prob": 0.0, "final_prob": 0.0,
            "stake": 0.0, "decision": "SKIP",
            "reasoning": rl["reason"]
        }

    micro = micro_model(df, direction)

    if micro["action"] == "SKIP":
        return {
            "regime": regime, "direction": direction,
            "rl_confidence": rl["confidence"],
            "micro_prob": micro["entry_prob"], "final_prob": 0.0,
            "stake": 0.0, "decision": "SKIP",
            "reasoning": micro["reason"]
        }

    raw_prob  = fuse_probabilities(rl["confidence"], micro["entry_prob"], regime)
    cal_prob  = calibrator.calibrate(raw_prob)
    mart_stake = martingale.get_stake()
    stake, size_label = compute_position_size(cal_prob, mart_stake)

    decision = "EXECUTE" if stake > 0.0 else "SKIP"

    reasoning = (
        f"Regime={regime} | RL: {direction} conf={rl['confidence']:.3f} | "
        f"Micro: {micro['action']} prob={micro['entry_prob']:.3f} | "
        f"Fused={raw_prob:.3f} Calibrated={cal_prob:.3f} | "
        f"Size={size_label} Stake=${stake:.2f} | "
        f"RL[{rl['reason']}] Micro[{micro['reason']}]"
    )

    return {
        "regime":        regime,
        "direction":     direction,
        "rl_confidence": round(rl["confidence"], 4),
        "micro_prob":    round(micro["entry_prob"], 4),
        "raw_prob":      round(raw_prob, 4),
        "final_prob":    round(cal_prob, 4),
        "stake":         stake,
        "size_label":    size_label,
        "decision":      decision,
        "mart_status":   martingale.status,
        "reasoning":     reasoning,
    }


# -------------------------------------------------------
# BOT STATE
# -------------------------------------------------------
class BotState:
    def __init__(self):
        self.balance            = 0.0
        self.start_balance      = 0.0
        self.daily_pnl          = 0.0
        self._daily_pnl_date    = datetime.now(timezone.utc).date()
        self.total_trades       = 0
        self.wins               = 0
        self.losses             = 0
        self.consecutive_losses = 0
        self.kill_switch_until: Optional[float] = None
        self.open_contract_id:  Optional[str]   = None
        self.open_direction:    Optional[str]   = None
        self.open_stake:        float           = 0.0
        self.open_prob:         float           = 0.0

    def maybe_reset_daily_pnl(self):
        """Reset daily PnL at UTC midnight so kill switch does not fire across days."""
        today = datetime.now(timezone.utc).date()
        if today != self._daily_pnl_date:
            logger.info(
                f"New UTC day - resetting daily PnL "
                f"(was ${self.daily_pnl:+.2f})"
            )
            self.daily_pnl       = 0.0
            self._daily_pnl_date = today
            # Clear a kill switch that fired due to yesterday's loss limit
            if self.kill_switch_active:
                logger.info("Kill switch cleared on new trading day.")
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
        logger.critical(f"KILL SWITCH ACTIVATED - {reason} | Resume at {resume}")

    def summary(self) -> str:
        wr = f"{self.win_rate*100:.1f}%" if self.total_trades > 0 else "N/A"
        return (
            f"Balance=${self.balance:.2f} | Daily PnL=${self.daily_pnl:+.2f} | "
            f"Trades={self.total_trades} W={self.wins} L={self.losses} WR={wr}"
        )


# -------------------------------------------------------
# DERIV WEBSOCKET CLIENT
# -------------------------------------------------------
class DerivBot:
    def __init__(self):
        self.ws                  = None
        self.req_id              = 1
        self.candles             = CandleStore()
        self.calibrator          = OnlineCalibrator()
        self.martingale          = MartingaleManager()
        self.state               = BotState()
        self._pending: dict      = {}       # req_id -> asyncio.Future
        self._authorized         = False
        self._last_candle_epoch  = 0

        # FIX #1/#9 - single asyncio.Lock prevents concurrent place_trade calls
        self._trade_lock         = asyncio.Lock()
        # FIX #5/#10/#11 - explicit boolean so _settle / reconnect can force-clear
        self._in_trade           = False
        # Track background push-handler tasks so they can be cancelled on shutdown
        self._push_tasks: set    = set()

    # ---- trade lock helpers -----------------------------------------------

    def _acquire_trade_lock_sync(self) -> bool:
        """
        Non-async guard: returns True and marks in-trade if not already locked.
        Must be called *before* the first await inside place_trade so no other
        coroutine can slip in during an await gap.
        """
        if self._in_trade:
            return False
        self._in_trade = True
        return True

    def _release_trade_lock(self):
        """Safely release the trade lock without triggering martingale logic."""
        self._in_trade = False

    # ---- WebSocket I/O ----------------------------------------------------

    async def send(self, payload: dict) -> dict:
        """Send a request and wait for the matching response (RPC pattern)."""
        # FIX #7 - use get_running_loop() instead of deprecated get_event_loop()
        loop = asyncio.get_running_loop()
        rid  = self.req_id
        self.req_id += 1
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
        """
        Receive loop: routes request-response pairs via pending futures,
        and dispatches push messages as background tasks so the loop itself
        never blocks (FIX #6 - no direct await of _handle_push in recv loop).

        FIX #20 - ConnectionClosed (including keepalive ping timeout code 1011)
        is caught here and re-raised as a plain ConnectionError so the outer
        run() loop treats it as a normal reconnect rather than a fatal crash.
        """
        try:
            async for raw in self.ws:
                msg = json.loads(raw)
                rid = msg.get("req_id")
                if rid and rid in self._pending:
                    fut = self._pending.pop(rid)
                    if not fut.done():
                        fut.set_result(msg)
                else:
                    # FIX #6 - create_task so recv_loop is never blocked
                    task = asyncio.create_task(self._handle_push(msg))
                    self._push_tasks.add(task)
                    task.add_done_callback(self._push_tasks.discard)
        except ConnectionClosed as exc:
            # FIX #20 - keepalive ping timeout (code 1011) and other
            # clean/unclean closes all arrive here. Demote to WARNING so the
            # terminal does not show a red "Fatal error" on every network blip,
            # then let the run() reconnect loop take over.
            logger.warning(
                f"WebSocket closed (code={exc.rcvd.code if exc.rcvd else 'N/A'} "
                f"reason={exc.rcvd.reason if exc.rcvd else str(exc)}) - reconnecting..."
            )
            raise ConnectionError(str(exc)) from exc

    async def _handle_push(self, msg: dict):
        """Handle subscription pushes (candle ticks, contract updates)."""
        msg_type = msg.get("msg_type")

        if msg_type == "ohlc":
            ohlc  = msg.get("ohlc", {})
            epoch = int(ohlc.get("epoch", 0))
            open_epoch = int(ohlc.get("open_time", epoch))

            # CANDLE-CLOSE TRIGGER FIX:
            # Deriv streams live OHLC updates on every tick for the *current*
            # forming candle. The candle is only CLOSED when a new open_time
            # epoch arrives (i.e. a new candle has started, meaning the prior
            # one is confirmed closed). We:
            #   1. Always update the current candle in-place so the store
            #      reflects latest prices (useful for display / logging).
            #   2. Only run analysis when open_epoch advances — meaning we
            #      are seeing the FIRST update of a brand-new candle, which
            #      confirms the previous candle has fully closed.
            if open_epoch > self._last_candle_epoch:
                # A new candle has opened — the previous candle is now closed.
                # Update the store with the final closed values of the candle
                # that was forming (current last in store was being updated
                # in-place, so now we add the new candle's first tick).
                self._last_candle_epoch = open_epoch
                self.candles.add(
                    open_epoch,
                    ohlc.get("open"), ohlc.get("high"),
                    ohlc.get("low"),  ohlc.get("close"),
                    ohlc.get("volume", 0)
                )
                # Fire analysis exactly once: on confirmed candle close
                await self._on_new_candle()
            else:
                # Same candle still forming — update the last entry in-place
                # so the store always reflects the latest OHLC for logging,
                # but do NOT re-trigger analysis.
                if self.candles.candles:
                    last = self.candles.candles[-1]
                    if last["epoch"] == open_epoch:
                        last["high"]   = max(last["high"],  float(ohlc.get("high",  last["high"])))
                        last["low"]    = min(last["low"],   float(ohlc.get("low",   last["low"])))
                        last["close"]  = float(ohlc.get("close", last["close"]))
                        last["volume"] = float(ohlc.get("volume", last["volume"]))

        elif msg_type == "proposal_open_contract":
            poc = msg.get("proposal_open_contract", {})
            await self._on_contract_update(poc)

        elif msg_type == "error":
            logger.error(f"Push error from server: {msg.get('error', {}).get('message', msg)}")

    # ---- Authorization ----------------------------------------------------

    async def authorize(self):
        logger.info("Authorizing with Deriv API...")
        resp = await self.send({"authorize": DERIV_API_TOKEN})
        if "error" in resp:
            raise RuntimeError(f"Authorization failed: {resp['error']['message']}")
        account = resp["authorize"]
        self.state.balance       = float(account.get("balance", 0))
        self.state.start_balance = self.state.balance
        self._authorized = True
        logger.info(
            f"Authorized | Account: {account.get('loginid')} | "
            f"Balance: ${self.state.balance:.2f} | Currency: {account.get('currency')}"
        )

    # ---- Historical Candles -----------------------------------------------

    async def fetch_history(self):
        logger.info(f"Fetching {CANDLE_HISTORY} candles of history...")
        resp = await self.send({
            "ticks_history": SYMBOL,
            "style":         "candles",
            "granularity":   GRANULARITY,
            "count":         CANDLE_HISTORY,
            "end":           "latest",
        })
        if "error" in resp:
            raise RuntimeError(
                f"History fetch failed: {resp['error']['message']} - cannot trade without data"
            )
        candles_data = resp.get("candles", [])
        if not candles_data:
            raise RuntimeError("History fetch returned 0 candles - cannot trade without data")
        for c in candles_data:
            self.candles.add(c["epoch"], c["open"], c["high"], c["low"], c["close"])
        if candles_data:
            self._last_candle_epoch = int(candles_data[-1]["epoch"])
        logger.info(f"Loaded {len(candles_data)} historical candles")

        # FIX #19 - RL warmup: train weight vector on historical candles before
        # live trading starts.  Uses next-candle return sign as supervised label.
        _rl_agent_instance.warmup(self.candles.df())

    # ---- Subscribe to Live Candles ----------------------------------------

    async def subscribe_candles(self):
        """
        FIX #4 - use self.send() (RPC path) so subscription errors are
        surfaced immediately rather than silently dropped.
        """
        logger.info("Subscribing to live candle stream...")
        resp = await self.send({
            "ticks_history": SYMBOL,
            "style":         "candles",
            "granularity":   GRANULARITY,
            "subscribe":     1,
            "end":           "latest",
            "count":         1,
        })
        if "error" in resp:
            raise RuntimeError(
                f"Candle subscription failed: {resp['error']['message']}"
            )
        logger.info("Live candle subscription active.")

    # ---- Balance Refresh --------------------------------------------------

    async def refresh_balance(self):
        try:
            resp = await self.send({"balance": 1, "account": "current"})
            if "balance" in resp:
                self.state.balance = float(resp["balance"]["balance"])
        except Exception as e:
            logger.warning(f"Balance refresh failed: {e}")

    # ---- Place Trade -------------------------------------------------------

    async def place_trade(self, direction: str, stake: float, final_prob: float):
        """
        FIX #1/#9 - trade lock is acquired synchronously before any await.
        Using asyncio.Lock() as outer guard prevents concurrent entry even if
        two tasks are scheduled simultaneously.
        """
        async with self._trade_lock:
            # Second check inside lock for safety
            if self._in_trade:
                logger.debug("place_trade skipped - already in trade")
                return
            # Mark in-trade BEFORE any await to block re-entry
            self._in_trade = True

        try:
            await self._execute_trade(direction, stake, final_prob)
        except Exception as e:
            logger.error(f"place_trade unhandled exception: {e}", exc_info=True)
            # FIX #2 - release lock cleanly on any unexpected failure
            self._release_trade_lock()

    async def _execute_trade(self, direction: str, stake: float, final_prob: float):
        """Core trade execution. Trade lock is already held when this runs."""
        contract_type = "CALL" if direction == "BUY" else "PUT"
        logger.info(
            f"Placing {direction} (Deriv: {contract_type}) | "
            f"Stake=${stake:.2f} | Prob={final_prob:.3f} | "
            f"Martingale: {self.martingale.status}"
        )

        # FIX #3 - amount must be a number (float), not a string
        try:
            proposal_resp = await self.send({
                "proposal":      1,
                "amount":        float(stake),   # FIX #3
                "basis":         "stake",
                "contract_type": contract_type,
                "currency":      "USD",
                "duration":      5,
                "duration_unit": "m",
                "symbol":        SYMBOL,
            })
        except asyncio.TimeoutError:
            logger.error("Proposal request timed out")
            # FIX #2 - API failure is not a lost trade; release lock only
            self._release_trade_lock()
            return

        if "error" in proposal_resp:
            logger.error(f"Proposal error: {proposal_resp['error']['message']}")
            # FIX #2 - release lock, do NOT call martingale loss recording
            self._release_trade_lock()
            return

        proposal_id = proposal_resp["proposal"]["id"]
        ask_price   = float(proposal_resp["proposal"]["ask_price"])

        # FIX #3 - price must be a number (float), not a string
        try:
            buy_resp = await self.send({
                "buy":   proposal_id,
                "price": float(ask_price),       # FIX #3
            })
        except asyncio.TimeoutError:
            logger.error("Buy request timed out")
            self._release_trade_lock()
            return

        if "error" in buy_resp:
            logger.error(f"Buy error: {buy_resp['error']['message']}")
            # FIX #2 - release lock, do NOT call martingale loss recording
            self._release_trade_lock()
            return

        contract = buy_resp["buy"]
        self.state.open_contract_id = str(contract["contract_id"])
        self.state.open_direction   = direction
        self.state.open_stake       = stake
        self.state.open_prob        = final_prob
        self.state.total_trades    += 1

        logger.info(
            f"Contract opened | ID={self.state.open_contract_id} | "
            f"Payout=${float(contract.get('payout', 0)):.2f} | "
            f"Balance=${float(contract.get('balance_after', 0)):.2f}"
        )

        # Subscribe to contract settlement updates
        # Use raw send (no RPC future) because updates arrive as push messages
        await self.ws.send(json.dumps({
            "proposal_open_contract": 1,
            "contract_id": int(self.state.open_contract_id),
            "subscribe":   1,
            "req_id":      self.req_id,
        }))
        self.req_id += 1
        # Lock is intentionally kept until _on_contract_update settles the trade

    # ---- Contract Update Handler ------------------------------------------

    async def _on_contract_update(self, poc: dict):
        """
        Called when the subscribed contract reaches won/lost status.
        FIX #5/#10 - always release trade lock in finally, regardless of
        where an exception might be thrown.
        """
        if poc.get("status") not in ("won", "lost"):
            return

        cid    = str(poc.get("contract_id", ""))
        status = poc["status"]
        profit = float(poc.get("profit", 0))

        if cid != self.state.open_contract_id:
            return

        won = (status == "won")
        self.state.daily_pnl        += profit
        self.state.open_contract_id  = None

        try:
            self.calibrator.record(self.state.open_prob, 1 if won else 0)

            # FIX #19 - online RL update: nudge weights toward actual outcome
            df_now = self.candles.df()
            if len(df_now) >= MIN_CANDLES:
                regime   = detect_regime(df_now)
                features = _rl_agent_instance._extract_features(df_now, regime)
                _rl_agent_instance.update(features, 1 if won else 0)

            if won:
                self.state.wins               += 1
                self.state.consecutive_losses  = 0
                self.martingale.record_win()
                logger.info(
                    Fore.GREEN + Style.BRIGHT +
                    f"WIN  | Profit=${profit:+.2f} | {self.state.summary()} | "
                    f"Calibrator: {self.calibrator.stats}"
                )
            else:
                self.state.losses             += 1
                self.state.consecutive_losses += 1
                self.martingale.record_loss()
                logger.warning(
                    Fore.RED +
                    f"LOSS | Profit=${profit:+.2f} | {self.state.summary()} | "
                    f"Martingale: {self.martingale.status}"
                )
                if self.state.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                    self.state.activate_kill_switch(
                        f"{MAX_CONSECUTIVE_LOSSES} consecutive losses"
                    )

            if self.state.daily_pnl <= -MAX_DAILY_LOSS:
                self.state.activate_kill_switch(
                    f"Daily loss limit ${MAX_DAILY_LOSS:.2f} hit "
                    f"(PnL=${self.state.daily_pnl:.2f})"
                )

            await self.refresh_balance()

        finally:
            # FIX #5/#10 - always release lock regardless of exceptions above
            self._release_trade_lock()

    # ---- New Candle Handler -----------------------------------------------

    async def _on_new_candle(self):
        # Reset daily PnL if UTC date has rolled over
        self.state.maybe_reset_daily_pnl()

        if not self.candles.ready():
            logger.debug(
                f"Building candle history... "
                f"({len(self.candles.candles)}/{MIN_CANDLES})"
            )
            return

        if self._in_trade:
            logger.debug("Contract open - waiting for outcome before next signal")
            return

        if self.state.kill_switch_active:
            resume = datetime.fromtimestamp(
                self.state.kill_switch_until
            ).strftime("%H:%M:%S")
            logger.warning(f"Kill switch active - trading paused until {resume}")
            return

        df = self.candles.df()
        logger.info("-" * 70)
        logger.info(
            f"New candle | {datetime.now().strftime('%H:%M:%S')} | "
            f"Close={df['close'].iloc[-1]:.3f}"
        )

        decision = make_decision(df, self.calibrator, self.martingale)
        self._log_decision(decision)

        if decision["decision"] == "EXECUTE":
            await self.place_trade(
                decision["direction"],
                decision["stake"],
                decision["final_prob"]
            )
        else:
            logger.info(
                f"SKIP - {decision['decision']} | {decision['reasoning'][:100]}"
            )

    # ---- Decision Logger --------------------------------------------------

    def _log_decision(self, d: dict):
        regime    = d["regime"]
        direction = d["direction"]
        decision  = d["decision"]

        regime_colors = {
            "CALM":       Fore.BLUE,
            "EXPANDING":  Fore.CYAN,
            "TRENDING":   Fore.GREEN,
            "EXHAUSTION": Fore.YELLOW,
        }
        dir_color = (
            Fore.GREEN if direction == "BUY"
            else (Fore.RED if direction == "SELL" else Fore.WHITE)
        )
        dec_color = Fore.GREEN if decision == "EXECUTE" else Fore.YELLOW
        rc = regime_colors.get(regime, Fore.WHITE)

        logger.info(f"  Regime      : {rc}{regime}{Style.RESET_ALL}")
        logger.info(f"  Direction   : {dir_color}{direction}{Style.RESET_ALL}")
        logger.info(f"  RL Conf     : {d.get('rl_confidence', 0):.4f}")
        logger.info(f"  Micro Prob  : {d.get('micro_prob', 0):.4f}")
        logger.info(f"  Final Prob  : {d.get('final_prob', 0):.4f}")
        logger.info(f"  Stake       : ${d.get('stake', 0):.2f}  [{d.get('size_label', 'N/A')}]")
        logger.info(f"  Martingale  : {d.get('mart_status', 'N/A')}")
        logger.info(f"  Decision    : {dec_color}{decision}{Style.RESET_ALL}")
        logger.info(f"  Reasoning   : {d.get('reasoning', '')[:120]}")

    async def _requery_open_contract(self):
        """
        On reconnect, if we have an open_contract_id we don't know the outcome of,
        query Deriv directly to get the final status and settle it properly.
        This prevents phantom locked trades after a disconnect mid-contract.
        """
        cid = self.state.open_contract_id
        if not cid:
            return

        logger.warning(
            f"Open contract {cid} detected after reconnect - querying outcome..."
        )
        try:
            resp = await asyncio.wait_for(
                self.send({
                    "proposal_open_contract": 1,
                    "contract_id": int(cid),
                }),
                timeout=CONTRACT_REQUERY_TIMEOUT,
            )
        except (asyncio.TimeoutError, Exception) as e:
            logger.error(
                f"Could not requery contract {cid}: {e} - "
                f"assuming loss and releasing lock to resume trading."
            )
            # Conservative: count as loss so martingale doesn't under-stake
            self.state.losses             += 1
            self.state.consecutive_losses += 1
            self.martingale.record_loss()
            self.calibrator.record(self.state.open_prob, 0)
            self.state.open_contract_id = None
            self._release_trade_lock()
            return

        poc = resp.get("proposal_open_contract", {})
        status = poc.get("status", "unknown")

        if status in ("won", "lost"):
            logger.info(f"Requery resolved contract {cid} as: {status.upper()}")
            # Drive through the normal settlement path
            await self._on_contract_update(poc)
        elif status == "open":
            # Contract is still live - re-subscribe for settlement push
            logger.info(
                f"Contract {cid} still open after reconnect - re-subscribing..."
            )
            await self.ws.send(json.dumps({
                "proposal_open_contract": 1,
                "contract_id": int(cid),
                "subscribe":   1,
                "req_id":      self.req_id,
            }))
            self.req_id += 1
        else:
            logger.warning(
                f"Contract {cid} returned unknown status '{status}' - "
                f"releasing lock as precaution."
            )
            self.state.open_contract_id = None
            self._release_trade_lock()

    def _reset_connection_state(self):
        """
        Reset all per-connection state so a reconnect starts clean.
        Preserves trading statistics, calibrator, martingale, and candle history.
        """
        # Cancel and clear any pending RPC futures from the dead connection
        for fut in self._pending.values():
            if not fut.done():
                fut.cancel()
        self._pending.clear()

        # Cancel any in-flight push handler tasks
        for task in list(self._push_tasks):
            task.cancel()
        self._push_tasks.clear()

        self._authorized = False
        self.ws = None
        # req_id intentionally NOT reset — monotonic across reconnects avoids
        # any chance of a delayed response matching a new request's id.

    # ---- Main Run Loop ----------------------------------------------------

    async def run(self):
        print_banner()

        reconnect_delay = RECONNECT_DELAY_MIN

        while True:
            # Check daily PnL reset before each connection attempt
            self.state.maybe_reset_daily_pnl()

            # On reconnect: clear dead connection state, then handle any
            # open contract whose outcome we don't know yet.
            self._reset_connection_state()

            had_open_contract = bool(self.state.open_contract_id)
            if self._in_trade and not had_open_contract:
                # Lock held but no contract ID - leaked lock, force clear
                logger.warning(
                    "Stale trade lock with no contract ID - releasing."
                )
                self._release_trade_lock()

            logger.info("Connecting to Deriv WebSocket...")
            try:
                async with websockets.connect(
                    WS_URL,
                    ping_interval=WS_PING_INTERVAL,
                    ping_timeout=WS_PING_TIMEOUT,   # forces disconnect if server goes silent
                ) as ws:
                    self.ws = ws
                    recv_task = asyncio.create_task(self._recv_loop())

                    await self.authorize()
                    await self.fetch_history()
                    await self.subscribe_candles()

                    # If we had an open contract from before the disconnect,
                    # query its outcome now that we're back online.
                    if had_open_contract:
                        await self._requery_open_contract()

                    logger.info("Bot running - monitoring XAU/USD...")
                    logger.info(
                        f"Thresholds: NoTrade<{THRESHOLD_NO_TRADE} | "
                        f"Small<{THRESHOLD_SMALL} | Normal<{THRESHOLD_NORMAL} | "
                        f"Aggressive>={THRESHOLD_NORMAL}"
                    )

                    # Successful connection - reset backoff
                    reconnect_delay = RECONNECT_DELAY_MIN

                    try:
                        await recv_task
                    except asyncio.CancelledError:
                        pass
                    except ConnectionError:
                        # FIX #20 - clean reconnect: ConnectionClosed (ping
                        # timeout, server drop, etc.) bubbles up as ConnectionError
                        # from _recv_loop. Let the outer reconnect loop handle it.
                        recv_task.cancel()
                    except Exception as e:
                        logger.critical(
                            f"Fatal error in recv loop: {e}", exc_info=True
                        )
                        recv_task.cancel()

            except KeyboardInterrupt:
                logger.info("Bot stopped by user.")
                return

            except Exception as e:
                logger.error(
                    f"Connection error: {e} - "
                    f"reconnecting in {reconnect_delay}s..."
                )
                await asyncio.sleep(reconnect_delay)
                # Exponential backoff, capped at RECONNECT_DELAY_MAX
                reconnect_delay = min(reconnect_delay * 2, RECONNECT_DELAY_MAX)


# -------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------
async def main():
    if DERIV_API_TOKEN == "YOUR_TOKEN_HERE":
        logger.critical("No API token set! Set DERIV_API_TOKEN env variable.")
        logger.critical("  Example: export DERIV_API_TOKEN=your_real_token")
        logger.critical("  Or create a .env file with: DERIV_API_TOKEN=your_token")
        sys.exit(1)

    bot = DerivBot()
    await bot.run()


if __name__ == "__main__":
    # FIX #7 - asyncio.run() handles the loop correctly on Python 3.10+
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
