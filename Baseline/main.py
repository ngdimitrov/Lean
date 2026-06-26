# region imports
from AlgorithmImports import *
# endregion

# ======================================================================================
#  BASELINE — CLASSIC DUAL MOMENTUM
# --------------------------------------------------------------------------------------
#  Deliberately minimal benchmark for GlobalQuant. NO HMM, NO entropy filter, NO game
#  theory, NO circuit breaker — just the core alpha that is supposed to make the money:
#
#    Each month, rank the same 8 ETFs by 6-month (126d) total return, keep the top-K that
#    ALSO have positive absolute momentum (else -> cash), and EQUAL-WEIGHT them.
#
#  Purpose: isolate how much the four "math pillars" in the full strategy actually add
#  (or subtract). Same universe, dates, and starting capital as GlobalQuant so the two
#  backtests are directly comparable.
# ======================================================================================


class DualMomentumBaseline(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2006, 1, 1)
        self.SetEndDate(2025, 1, 1)
        self.SetCash(1_000_000)

        # Same universe as GlobalQuant so the comparison is apples-to-apples.
        tickers = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD"]
        self.symbols = [self.AddEquity(t, Resolution.Daily).Symbol for t in tickers]

        self.mom_lookback = 126                # ~6m momentum (classic momentum horizon)
        self.top_k        = 4                  # hold up to K strongest names

        # Rebalance on the first trading day of each month, just after the open.
        self.Schedule.On(
            self.DateRules.MonthStart(self.symbols[0]),
            self.TimeRules.AfterMarketOpen(self.symbols[0], 30),
            self.Rebalance,
        )

    def Rebalance(self):
        # --- Rank by trailing total return; keep only positive absolute momentum (dual) ---
        scores = {}
        for sym in self.symbols:
            hist = self.History(sym, self.mom_lookback + 1, Resolution.Daily)
            if hist.empty or "close" not in hist.columns:
                continue
            closes = hist["close"].values
            if len(closes) < self.mom_lookback + 1:
                continue
            mom = closes[-1] / closes[0] - 1.0             # trailing ~6m return
            if mom > 0.0:                                   # absolute (dual) momentum filter
                scores[sym] = mom

        selected = sorted(scores, key=scores.get, reverse=True)[:self.top_k]

        # --- Equal-weight the survivors; everything else -> cash ---
        if not selected:
            self.Liquidate()
            return

        w_each = 1.0 / len(selected)
        targets = [
            PortfolioTarget(sym, w_each if sym in selected else 0.0)
            for sym in self.symbols
        ]
        self.SetHoldings(targets)
        self.Log(f"hold={[s.Value for s in selected]}")
