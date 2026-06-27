# SuperAggressive Volatility-Targeted Trend System — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 15m breakout strategy with a 1H Donchian trend system that uses a Daily EMA200 regime filter, a Chandelier ATR trailing exit, and portfolio volatility targeting.

**Architecture:** Three units — `VolatilityTargetSizer` (pure numpy, own file, unit-tested), `SymbolData` (per-asset indicators + position state), and `SuperAggressiveTrendSystem(QCAlgorithm)` (orchestration + order execution). Hourly `OnData` drives signals/exits; a daily scheduled event re-targets portfolio volatility.

**Tech Stack:** QuantConnect LEAN (Python), numpy.

## Global Constraints

- Period: 2021-01-01 → 2025-01-01; Cash: $100,000.
- Universe: BTCUSD, ETHUSD, SOLUSD at `Resolution.Hour`, `Market.GDAX`.
- Donchian period: 24 (1H bars); Macro filter: Daily EMA 200; Chandelier: Daily ATR 22 × 3.0.
- Covariance lookback: 60 daily returns; Target annual vol: 0.275; Annualization: 365; Max gross leverage: 1.0.
- Warmup: `timedelta(days=210)`. All parameters centralized in `Initialize`.
- LEAN strategies cannot be unit-tested locally (no engine). Only `VolatilityTargetSizer` (pure numpy) is locally testable; the algorithm is verified by `py_compile` + a QuantConnect backtest.

---

### Task 1: `VolatilityTargetSizer` (pure numpy, locally tested)

**Files:**
- Create: `SuperAggressive/sizer.py`
- Test: `SuperAggressive/test_sizer.py`

**Interfaces:**
- Produces: `VolatilityTargetSizer(target_annual_vol=0.275, max_gross_leverage=1.0, periods_per_year=365, min_observations=30)` with method `compute_weights(returns_by_symbol: dict[Any, np.ndarray], active_symbols: list) -> dict[Any, float]`.

- [ ] **Step 1: Write the failing test** (`SuperAggressive/test_sizer.py`)

```python
import numpy as np
from sizer import VolatilityTargetSizer

def _approx(a, b, tol=1e-6):
    assert abs(a - b) <= tol, f"{a} != {b}"

def test_empty_active_returns_empty():
    s = VolatilityTargetSizer()
    assert s.compute_weights({}, []) == {}

def test_below_min_observations_excluded():
    s = VolatilityTargetSizer(min_observations=30)
    short = np.full(10, 0.01)
    assert s.compute_weights({"A": short}, ["A"]) == {}

def test_single_asset_scales_to_target_vol():
    # daily std 0.02 -> annual vol 0.02*sqrt(365) ~ 0.3821
    rng = np.random.default_rng(0)
    r = rng.normal(0.0, 0.02, 500)
    s = VolatilityTargetSizer(target_annual_vol=0.275, periods_per_year=365)
    w = s.compute_weights({"A": r}, ["A"])
    realized = abs(w["A"]) * r.std(ddof=1) * np.sqrt(365)
    _approx(realized, 0.275, tol=0.02)

def test_leverage_cap_binds_for_low_vol():
    r = np.full(100, 0.0)  # zero vol -> k falls back to 1.0, capped
    s = VolatilityTargetSizer(target_annual_vol=0.275, max_gross_leverage=1.0)
    w = s.compute_weights({"A": r}, ["A"])
    assert w["A"] <= 1.0 + 1e-9

def test_two_assets_portfolio_vol_hits_target():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 0.03, 400)
    b = rng.normal(0, 0.05, 400)
    s = VolatilityTargetSizer(target_annual_vol=0.275, max_gross_leverage=10.0)
    w = s.compute_weights({"A": a, "B": b}, ["A", "B"])
    R = np.column_stack([a, b])
    cov = np.cov(R, rowvar=False) * 365
    wv = np.array([w["A"], w["B"]])
    port_vol = np.sqrt(wv @ cov @ wv)
    _approx(port_vol, 0.275, tol=0.02)

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"PASS {name}")
    print("ALL TESTS PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd SuperAggressive && python3 test_sizer.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'sizer'`

- [ ] **Step 3: Write minimal implementation** (`SuperAggressive/sizer.py`)

```python
import numpy as np


class VolatilityTargetSizer:
    """Portfolio sizing so annualized vol matches a fixed target. No QC deps."""

    def __init__(self, target_annual_vol=0.275, max_gross_leverage=1.0,
                 periods_per_year=365, min_observations=30):
        self.target_annual_vol = target_annual_vol
        self.max_gross_leverage = max_gross_leverage
        self.periods_per_year = periods_per_year
        self.min_observations = min_observations

    def compute_weights(self, returns_by_symbol, active_symbols):
        if not active_symbols:
            return {}
        usable, series = [], []
        for s in active_symbols:
            r = np.asarray(returns_by_symbol.get(s, []), dtype=float)
            if r.size >= self.min_observations:
                usable.append(s); series.append(r)
        if not usable:
            return {}
        n = min(len(r) for r in series)
        R = np.column_stack([r[-n:] for r in series])

        vols = R.std(axis=0, ddof=1)
        inv = np.divide(1.0, vols, out=np.zeros_like(vols), where=vols > 0)
        total = inv.sum()
        base = inv / total if total > 0 else np.full(len(usable), 1.0 / len(usable))

        try:
            cov = np.atleast_2d(np.cov(R, rowvar=False)) * self.periods_per_year
            port_var = float(base @ cov @ base)
        except Exception:
            port_var = 0.0
        k = self.target_annual_vol / np.sqrt(port_var) if port_var > 0 else 1.0
        k = max(0.0, min(k, self.max_gross_leverage))
        return {s: float(k * w) for s, w in zip(usable, base)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd SuperAggressive && python3 test_sizer.py`
Expected: `ALL TESTS PASSED`

- [ ] **Step 5: Commit**

```bash
git add SuperAggressive/sizer.py SuperAggressive/test_sizer.py
git commit -m "feat: add VolatilityTargetSizer with unit tests"
```

---

### Task 2: `SymbolData` + algorithm in `main.py`

**Files:**
- Modify (overwrite): `SuperAggressive/main.py`

**Interfaces:**
- Consumes: `from sizer import VolatilityTargetSizer`.
- Produces: `SymbolData(algorithm, symbol, donchian_period=24, ema_period=200, atr_period=22, atr_mult=3.0, returns_lookback=60)` with `IsReady`, `on_hour_bar(bar)`, `on_daily_bar(bar)`, `entry_signal()`, `update_and_check_stop()`, `daily_returns()`; and `SuperAggressiveTrendSystem(QCAlgorithm)`.

- [ ] **Step 1: Write `main.py`** (full code — see Task 2 code block below)

```python
from AlgorithmImports import *
from datetime import timedelta
import numpy as np

from sizer import VolatilityTargetSizer


class SymbolData:
    """Indicators and long-position state for one crypto asset."""

    def __init__(self, algorithm, symbol, donchian_period=24, ema_period=200,
                 atr_period=22, atr_mult=3.0, returns_lookback=60):
        self.symbol = symbol
        self.atr_mult = atr_mult
        # Completed 1H bars: index 0 = current close, indices 1..donchian_period = lookback.
        self.bars = RollingWindow[TradeBar](donchian_period + 1)
        self.ema = algorithm.EMA(symbol, ema_period, Resolution.Daily)
        self.atr = algorithm.ATR(symbol, atr_period, resolution=Resolution.Daily)
        self.daily_closes = RollingWindow[float](returns_lookback + 1)
        self.active = False
        self.peak_price = 0.0

    @property
    def IsReady(self):
        return self.bars.IsReady and self.ema.IsReady and self.atr.IsReady

    def on_hour_bar(self, bar):
        self.bars.Add(bar)

    def on_daily_bar(self, bar):
        self.daily_closes.Add(float(bar.Close))

    def entry_signal(self):
        if not self.bars.IsReady:
            return False
        close = self.bars[0].Close
        donchian_high = max(self.bars[i].High for i in range(1, self.bars.Size))
        return close > donchian_high and close > self.ema.Current.Value

    def update_and_check_stop(self):
        bar = self.bars[0]
        if bar.High > self.peak_price:
            self.peak_price = bar.High
        stop = self.peak_price - self.atr_mult * self.atr.Current.Value
        return bar.Close <= stop

    def daily_returns(self):
        n = self.daily_closes.Count
        if n < 2:
            return np.array([])
        closes = np.array([self.daily_closes[i] for i in range(n)][::-1])
        return np.diff(closes) / closes[:-1]


class SuperAggressiveTrendSystem(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2021, 1, 1)
        self.SetEndDate(2025, 1, 1)
        self.SetCash(100000)

        self.tickers = ["BTCUSD", "ETHUSD", "SOLUSD"]
        self.rebalance_band = 0.05

        self.sizer = VolatilityTargetSizer(
            target_annual_vol=0.275, max_gross_leverage=1.0,
            periods_per_year=365, min_observations=30)

        self.symbol_data = {}
        for ticker in self.tickers:
            symbol = self.AddCrypto(ticker, Resolution.Hour, Market.GDAX).Symbol
            self.symbol_data[symbol] = SymbolData(self, symbol)
            consolidator = TradeBarConsolidator(timedelta(days=1))
            consolidator.DataConsolidated += self.on_daily_consolidated
            self.SubscriptionManager.AddConsolidator(symbol, consolidator)

        self.SetWarmUp(timedelta(days=210))
        self.Schedule.On(self.DateRules.EveryDay(),
                         self.TimeRules.At(0, 5), self.daily_retarget)

    def on_daily_consolidated(self, sender, bar):
        sd = self.symbol_data.get(bar.Symbol)
        if sd is not None:
            sd.on_daily_bar(bar)

    def OnData(self, data):
        for symbol, sd in self.symbol_data.items():
            if symbol in data.Bars:
                sd.on_hour_bar(data.Bars[symbol])

        if self.IsWarmingUp:
            return

        changed = False
        for symbol, sd in self.symbol_data.items():
            if symbol not in data.Bars or not sd.IsReady:
                continue
            if sd.active:
                if sd.update_and_check_stop():
                    sd.active = False
                    changed = True
                    self.Debug(f"CHANDELIER EXIT {symbol.Value} @ {sd.bars[0].Close:.2f}")
            elif sd.entry_signal():
                sd.active = True
                sd.peak_price = sd.bars[0].High
                changed = True
                self.Debug(f"ENTRY {symbol.Value} @ {sd.bars[0].Close:.2f}")

        if changed:
            self.rebalance("signal-change")

    def daily_retarget(self):
        if self.IsWarmingUp:
            return
        if any(sd.active for sd in self.symbol_data.values()):
            self.rebalance("daily-retarget")

    def rebalance(self, reason):
        if self.IsWarmingUp:
            return
        active = [s for s, sd in self.symbol_data.items() if sd.active and sd.IsReady]
        returns_by_symbol = {s: self.symbol_data[s].daily_returns() for s in active}
        weights = self.sizer.compute_weights(returns_by_symbol, active)

        tpv = self.Portfolio.TotalPortfolioValue
        targets = []
        for s in self.symbol_data:
            desired = weights.get(s, 0.0)
            holding = self.Portfolio[s]
            current = (holding.HoldingsValue / tpv) if tpv > 0 else 0.0
            if holding.Invested and desired == 0.0:
                targets.append(PortfolioTarget(s, 0.0))
            elif abs(desired - current) >= self.rebalance_band:
                targets.append(PortfolioTarget(s, desired))

        if targets:
            self.SetHoldings(targets)
            self.Log(f"Rebalance ({reason}): " + ", ".join(
                f"{t.Symbol.Value}={t.Quantity}" for t in targets))

    def OnData_placeholder(self):
        pass
```

- [ ] **Step 2: Syntax-check** (cannot run LEAN locally)

Run: `python3 -c "import ast; ast.parse(open('SuperAggressive/main.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add SuperAggressive/main.py
git commit -m "feat: rewrite SuperAggressive as 1H vol-targeted trend system"
```

- [ ] **Step 4: User verification (QuantConnect backtest)**

Run the backtest on QuantConnect. Check: realized portfolio vol ≈ 27.5%; max DD materially below the prior ~75%; no runtime errors during the SOL-missing early window.

---

## Self-Review

- **Spec coverage:** §3.1 SymbolData → Task 2; §3.2 Sizer → Task 1; §3.3 algorithm → Task 2; §4 params → Global Constraints + Initialize; §5 edge cases → sizer guards (empty/min-obs/singular) + IsReady gating; §6 testing → Task 1 tests + Task 2 verification; §7 limitations → noted (leverage cap 1.0).
- **Placeholder scan:** none — all steps contain full code/commands.
- **Type consistency:** `compute_weights(returns_by_symbol, active_symbols)` signature identical in Task 1 and its call in Task 2 `rebalance`; `daily_returns()` returns `np.ndarray` consumed by sizer; `SymbolData` constructor args match usage.
