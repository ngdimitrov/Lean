try:
    from AlgorithmImports import *
except ImportError:
    pass

try:
    QCAlgorithm  # noqa: B018 - provided by the QuantConnect cloud/LEAN runtime
except NameError:
    # Local-only fallback so VolatilityTargetSizer can be imported for unit
    # tests when the QuantConnect runtime is unavailable. Never used in cloud.
    class QCAlgorithm:  # noqa: N801 - QC base-class stand-in for local tests
        ...

from datetime import timedelta
import numpy as np


class VolatilityTargetSizer:
    """Portfolio sizing so the annualized volatility matches a fixed target.

    Pure numpy, no QuantConnect dependencies, so it can be unit-tested in
    isolation. Base allocation is inverse-volatility; the full covariance
    matrix (correlations included) is used to scale the whole book toward the
    target annual volatility, capped at ``max_gross_leverage``.
    """

    def __init__(self, target_annual_vol=0.275, max_gross_leverage=1.0,
                 periods_per_year=365, min_observations=30):
        self.target_annual_vol = target_annual_vol
        self.max_gross_leverage = max_gross_leverage
        self.periods_per_year = periods_per_year
        self.min_observations = min_observations

    def compute_weights(self, returns_by_symbol, active_symbols):
        """Return {symbol: weight} for active symbols; {} when nothing tradable."""
        if not active_symbols:
            return {}

        # Keep only symbols with enough history.
        usable, series = [], []
        for s in active_symbols:
            r = np.asarray(returns_by_symbol.get(s, []), dtype=float)
            if r.size >= self.min_observations:
                usable.append(s)
                series.append(r)
        if not usable:
            return {}

        # Align to the shortest common length (most recent observations).
        n = min(len(r) for r in series)
        R = np.column_stack([r[-n:] for r in series])

        # Inverse-volatility base weights (risk-balanced, robust).
        vols = R.std(axis=0, ddof=1)
        inv = np.divide(1.0, vols, out=np.zeros_like(vols), where=vols > 0)
        total = inv.sum()
        base = inv / total if total > 0 else np.full(len(usable), 1.0 / len(usable))

        # Annualized portfolio variance via the full covariance matrix.
        try:
            cov = np.atleast_2d(np.cov(R, rowvar=False)) * self.periods_per_year
            port_var = float(base @ cov @ base)
        except Exception:
            port_var = 0.0

        k = self.target_annual_vol / np.sqrt(port_var) if port_var > 0 else 1.0
        k = max(0.0, min(k, self.max_gross_leverage))

        return {s: float(k * w) for s, w in zip(usable, base)}


class SymbolData:
    """Indicators and long-position state for one crypto asset.

    Owns the 24h Donchian window (1H bars), the daily EMA200 regime filter,
    the daily ATR used by the Chandelier exit, and the daily-close window that
    feeds the covariance-based sizer.
    """

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
        """Long when close breaks the 24h Donchian high AND is above EMA200."""
        if not self.bars.IsReady:
            return False
        close = self.bars[0].Close
        donchian_high = max(self.bars[i].High for i in range(1, self.bars.Size))
        return close > donchian_high and close > self.ema.Current.Value

    def update_and_check_stop(self):
        """Update the position peak; return True if the Chandelier stop is hit."""
        bar = self.bars[0]
        if bar.High > self.peak_price:
            self.peak_price = bar.High
        stop = self.peak_price - self.atr_mult * self.atr.Current.Value
        return bar.Close <= stop

    def daily_returns(self):
        """Simple daily returns (oldest -> newest) for the covariance matrix."""
        n = self.daily_closes.Count
        if n < 2:
            return np.array([])
        closes = np.array([self.daily_closes[i] for i in range(n)][::-1])
        return np.diff(closes) / closes[:-1]


class SuperAggressiveTrendSystem(QCAlgorithm):
    """1H Donchian breakout + Daily EMA200 regime filter + Chandelier ATR exit,
    sized by portfolio volatility targeting (60d covariance, ~27.5% annual vol)."""

    def Initialize(self):
        self.SetStartDate(2021, 1, 1)
        self.SetEndDate(2025, 1, 1)
        self.SetCash(100000)

        self.tickers = ["BTCUSD", "ETHUSD", "SOLUSD"]
        self.rebalance_band = 0.05  # no-trade band to curb daily-retarget churn

        self.sizer = VolatilityTargetSizer(
            target_annual_vol=0.275, max_gross_leverage=1.0,
            periods_per_year=365, min_observations=30)

        self.symbol_data = {}
        for ticker in self.tickers:
            symbol = self.AddCrypto(ticker, Resolution.Hour, Market.GDAX).Symbol
            self.symbol_data[symbol] = SymbolData(self, symbol)
            # Daily bars (for covariance returns) via a native consolidator.
            consolidator = TradeBarConsolidator(timedelta(days=1))
            consolidator.DataConsolidated += self.on_daily_consolidated
            self.SubscriptionManager.AddConsolidator(symbol, consolidator)

        # Warm up enough days for EMA200 + the 60d covariance window.
        self.SetWarmUp(timedelta(days=210))

        # Daily re-target keeps portfolio vol on target as covariance drifts.
        self.Schedule.On(self.DateRules.EveryDay(),
                         self.TimeRules.At(0, 5), self.daily_retarget)

    def on_daily_consolidated(self, sender, bar):
        sd = self.symbol_data.get(bar.Symbol)
        if sd is not None:
            sd.on_daily_bar(bar)

    def OnData(self, data):
        # Always feed the hourly windows (also during warmup) so they are ready
        # the moment trading begins.
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
            # Always close fully-exited positions; otherwise honor the no-trade band.
            if holding.Invested and desired == 0.0:
                targets.append(PortfolioTarget(s, 0.0))
            elif abs(desired - current) >= self.rebalance_band:
                targets.append(PortfolioTarget(s, desired))

        if targets:
            self.SetHoldings(targets)
            self.Log(f"Rebalance ({reason}): " + ", ".join(
                f"{s.Value}={weights.get(s, 0.0):.3f}" for s in self.symbol_data))
