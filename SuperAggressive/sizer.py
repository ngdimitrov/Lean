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
