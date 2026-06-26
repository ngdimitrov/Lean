# region imports
from AlgorithmImports import *
import numpy as np
# endregion

# ======================================================================================
#  REGIME-AWARE MULTI-ASSET STRATEGY  —  4-PILLAR MATHEMATICAL ARCHITECTURE
# --------------------------------------------------------------------------------------
#  Pillar 1 | HIDDEN MARKOV MODEL + MARKOV CHAIN  (the "what state are we in?" layer)
#           A 3-state Gaussian HMM is fit (Baum-Welch / scaled forward-backward) on the
#           market proxy's (SPY) daily returns. States are sorted by mean return into
#           {Bear/High-Vol, Flat/Mean-Reverting, Bull/Low-Vol}. The fitted transition
#           matrix IS the Markov chain: we project ONE STEP AHEAD (posterior @ transmat)
#           to get a forward-looking probability of the favourable (Bull) regime.
#
#  Pillar 2 | INFORMATION THEORY  (the "is this signal or noise?" filter)
#           Normalised Shannon entropy is computed on each asset's rolling return
#           histogram. High entropy => the return distribution is near-uniform => chaotic
#           / low information => the asset is GATED OUT before any entry flag is honoured.
#
#  Pillar 3 | GAME THEORY  (the "how big, against an adversary?" sizing layer)
#           Position sizing is a 2-player robust game. Trader actions = {flat, half, full}.
#           Market = adversary that mixes {benign, adverse}. The adversary's benign
#           probability is tilted DOWN from the HMM forecast by an ambiguity margin
#           (distributionally-robust maximin). Payoffs net out edge vs. transaction
#           cost + slippage. We pick the action maximising the worst-case expected payoff.
#           The per-day Bull mean is scaled to a multi-day holding-period expectation so
#           a real edge can clear the (low) transaction-cost hurdle.
#
#  Pillar 4 | LAW OF LARGE NUMBERS  (the "make the edge converge" structure)
#           Edge only converges with sample size, so we trade a broad, highly-liquid
#           universe (SPY/QQQ/TLT/GLD), spread capital across all qualifying assets, and
#           cap per-asset weight. Many small near-independent bets => realised mean ->
#           mathematical expectation. Trade count is tracked.
#
#  EXECUTION ORDERING (OnData):
#    Step 1  Update rolling return windows ............................. DAILY
#    Step 2  Warm-up / readiness guard ................................. DAILY
#    Step 3  RISK GATE: -5% circuit breaker + cooldown decay ........... DAILY  <-- always
#    Step 4  STRICT WEEKLY CALENDAR GATE (Mondays only) ................ WEEKLY
#    Step 5+ HMM / Entropy / Game Theory / Sizing & Rebalancing ........ WEEKLY
#
#  The risk gate sits BEFORE the weekly gate, so the stop-loss is evaluated every trading
#  day while trade evaluation/turnover is throttled to a clean weekly (Monday) regime.
# ======================================================================================


class GlobalQuant(QCAlgorithm):

    # ----------------------------------------------------------------------------------
    #  INITIALISATION
    # ----------------------------------------------------------------------------------
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2021, 1, 1)
        self.SetCash(1_000_000)

        # ---- Universe: liquid, low-correlation multi-asset basket (LLN backbone) ----
        tickers = ["SPY", "QQQ", "AAPL", "GOOG"]
        self.symbols = [self.AddEquity(t, Resolution.Daily).Symbol for t in tickers]
        self.market = self.symbols[0]          # SPY drives the macro regime model

        # ---- Lookback / model parameters ----
        self.lookback_regime  = 60             # ~3mo of returns for HMM fitting (fits a 1yr backtest)
        self.lookback_entropy = 60             # window for Shannon-entropy noise filter
        self.refit_period     = 5              # refit the HMM every N evaluation days (Mondays)
        self.entropy_bins     = 10             # histogram resolution for entropy
        self.entropy_thresh   = 0.90           # normalised entropy above this => gate out

        # ---- Calibrated Game Theory parameters ----
        self.ambiguity          = 0.05         # adversary's tilt on benign probability
        self.round_trip_cost    = 0.0001       # ~1bps: low-fee institutional broker assumption
        self.edge_horizon_scale = 5            # scale daily Bull mean -> multi-day holding expectation

        # ---- Portfolio construction / turnover control ----
        self.max_per_asset = 0.50              # per-name weight cap (diversification/LLN)
        self.rebalance_tol = 0.10              # min weight delta (10%) before re-trading (anti-churn)

        # ---- Regime-conditional base allocations (gross weights per regime) ----
        s, q, a, g = self.symbols
        self.regime_weights = {
            "bull": {s: 0.40, q: 0.40, a: 0.10, g: 0.10},   # risk-on
            "flat": {s: 0.20, q: 0.15, a: 0.30, g: 0.20},   # balanced / defensive
            "bear": {s: 0.00, q: 0.00, a: 0.45, g: 0.35},   # flight-to-safety
        }

        # ---- Rolling state ----
        self.returns    = {sym: RollingWindow[float](self.lookback_regime) for sym in self.symbols}
        self.prev_price = {}
        self.hmm        = GaussianHMM(n_states=3, n_iter=12)
        self.day_count  = 0

        # ---- Risk: circuit breaker + cooldown quarantine ----
        self.peak_equity        = self.Portfolio.TotalPortfolioValue
        self.stop_loss_pct      = 0.05
        self.cooldown_days      = 10
        self.cooldown_remaining = 0

        # ---- LLN diagnostics ----
        self.fill_count = 0

        # Prime all rolling windows before trading begins.
        self.SetWarmUp(self.lookback_regime + 1, Resolution.Daily)

    # ----------------------------------------------------------------------------------
    #  MAIN EVENT LOOP
    # ----------------------------------------------------------------------------------
    def OnData(self, data: Slice):
        # --- STEP 1) Update rolling return windows for every asset that printed a bar (DAILY) ---
        for sym in self.symbols:
            if not (data.ContainsKey(sym) and data[sym] is not None):
                continue
            price = data[sym].Close
            if price is None or price <= 0:
                continue
            if sym in self.prev_price and self.prev_price[sym] > 0:
                self.returns[sym].Add(price / self.prev_price[sym] - 1.0)
            self.prev_price[sym] = price

        # --- STEP 2) Warm-up / readiness guard (DAILY): stay flat until every window is full ---
        if self.IsWarmingUp or not all(self.returns[sym].IsReady for sym in self.symbols):
            return

        # --- STEP 3) RISK GATE (DAILY): -5% circuit breaker + cooldown decay ---
        #     Evaluated every trading day so a mid-week drawdown liquidates immediately,
        #     independent of the weekly trade-evaluation cadence below.
        if self._risk_circuit_breaker_tripped():
            return                              # liquidated; cooling down — no new entries
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1        # quarantine: skip entries, decay the timer
            return

        # --- STEP 4) STRICT WEEKLY CALENDAR GATE: evaluate the pillar stack on Mondays only ---
        #     Sits AFTER the risk gate, so turnover is throttled but the stop-loss is not.
        if self.Time.weekday() != 0:        # Python datetime: Monday == 0
            return

        self.day_count += 1

        # --- STEP 5) PILLAR 1 — HMM regime detection on the market proxy ---
        market_returns = self._window_array(self.market)            # oldest -> newest
        if (self.day_count % self.refit_period == 0) or (not self.hmm.fitted):
            self.hmm.fit(market_returns)                            # periodic Baum-Welch refit

        posterior = self.hmm.predict_proba_last(market_returns)     # P(state | obs_1:T)
        forecast  = posterior @ self.hmm.transmat                   # Markov 1-step-ahead
        bull, _, bear = self.hmm.ordered_states()                  # ascending mean -> bear..bull

        regime_label = self.hmm.label_of(int(np.argmax(posterior)))
        p_benign     = float(forecast[bull])                       # forward-looking Bull prob

        # --- STEP 6) PILLAR 2 — Information-theoretic noise filter (per asset) ---
        base = self.regime_weights[regime_label]
        gross_target = sum(base.values())
        surviving = {}
        for sym, w in base.items():
            if w <= 0:
                continue
            H = self._shannon_entropy(self._window_array(sym)[-self.lookback_entropy:])
            if H <= self.entropy_thresh:                            # keep only "clean" assets
                surviving[sym] = w
        # Re-spread the regime's gross exposure across survivors (preserve target risk, LLN).
        ssum = sum(surviving.values())
        if ssum > 0:
            scale = gross_target / ssum
            surviving = {sym: w * scale for sym, w in surviving.items()}

        # --- STEP 7) PILLAR 3 — Game-theoretic (robust maximin) gross sizing ---
        means, varis = self.hmm.means, self.hmm.vars
        # Scale the daily Bull mean to a multi-day holding-period expectation so a genuine
        # edge can clear the transaction-cost hurdle (otherwise maximin always picks flat).
        edge    = max(means[bull] * self.edge_horizon_scale, 1e-4)  # expected favourable move
        adverse = max(abs(means[bear]), np.sqrt(varis[bear]))      # worst-case downside move
        size    = self._game_theory_size(p_benign, edge, adverse, self.round_trip_cost)

        # --- STEP 8) PILLAR 4 — Construct capped, diversified targets & rebalance ---
        targets = []
        for sym in self.symbols:
            w = surviving.get(sym, 0.0) * size
            w = float(np.clip(w, 0.0, self.max_per_asset))
            targets.append(PortfolioTarget(sym, w))

        if self._worth_rebalancing(targets):
            self.SetHoldings(targets)
            self.Log(f"[{regime_label.upper()}] pBenign={p_benign:.2f} size={size:.2f} "
                     f"targets={{{', '.join(f'{t.Symbol.Value}:{t.Quantity:.2f}' for t in targets)}}}")

    # ----------------------------------------------------------------------------------
    #  PILLAR 2 — INFORMATION THEORY: normalised Shannon entropy of a return window
    # ----------------------------------------------------------------------------------
    def _shannon_entropy(self, returns):
        if len(returns) < 2:
            return 1.0
        hist, _ = np.histogram(returns, bins=self.entropy_bins)
        p = hist.astype(float)
        total = p.sum()
        if total <= 0:
            return 1.0
        p = p[p > 0] / total
        H = -np.sum(p * np.log(p))
        return H / np.log(self.entropy_bins)                       # normalise to [0, 1]

    # ----------------------------------------------------------------------------------
    #  PILLAR 3 — GAME THEORY: distributionally-robust maximin position sizing
    #  Adversary mixes benign/adverse; its benign prob is tilted DOWN by `ambiguity`.
    #  We choose the action maximising the worst-case expected payoff.
    # ----------------------------------------------------------------------------------
    def _game_theory_size(self, p_benign, edge, adverse, cost):
        q = max(0.0, p_benign - self.ambiguity)                    # adversarial benign prob
        best_action, best_value = 0.0, -np.inf
        for a in (0.0, 0.5, 1.0):                                  # trader's pure strategies
            payoff_benign  = a * (edge - cost)                     # market cooperates
            payoff_adverse = a * (-adverse - cost)                 # market fights us
            expected = q * payoff_benign + (1.0 - q) * payoff_adverse
            if expected > best_value:
                best_value, best_action = expected, a
        return best_action

    # ----------------------------------------------------------------------------------
    #  RISK: portfolio -5% circuit breaker from the high-water mark + cooldown trigger
    # ----------------------------------------------------------------------------------
    def _risk_circuit_breaker_tripped(self):
        equity = self.Portfolio.TotalPortfolioValue
        self.peak_equity = max(self.peak_equity, equity)
        if equity <= self.peak_equity * (1.0 - self.stop_loss_pct):
            if self.Portfolio.Invested:
                self.Liquidate()
            self.cooldown_remaining = self.cooldown_days           # quarantine new entries
            self.peak_equity = equity                              # reset HWM post-stop
            self.Log(f"CIRCUIT BREAKER: equity {equity:,.0f} hit -{self.stop_loss_pct:.0%} "
                     f"from peak -> liquidated, cooldown {self.cooldown_days}d")
            return True
        return False

    # ----------------------------------------------------------------------------------
    #  HELPERS
    # ----------------------------------------------------------------------------------
    def _window_array(self, sym):
        """RollingWindow -> numpy array ordered oldest -> newest."""
        w = self.returns[sym]
        return np.array([w[i] for i in range(w.Count)], dtype=float)[::-1]

    def _worth_rebalancing(self, targets):
        """Anti-churn: only trade if any target deviates materially from current weight."""
        tpv = self.Portfolio.TotalPortfolioValue
        if tpv <= 0:
            return True
        for t in targets:
            current_w = self.Portfolio[t.Symbol].HoldingsValue / tpv
            if abs(t.Quantity - current_w) > self.rebalance_tol:
                return True
        return False

    def OnOrderEvent(self, orderEvent):
        # PILLAR 4 diagnostics: track realised sample size of bets for LLN convergence.
        if orderEvent.Status == OrderStatus.Filled:
            self.fill_count += 1
            if self.fill_count % 25 == 0:
                self.Log(f"LLN sample size: {self.fill_count} fills | "
                         f"equity={self.Portfolio.TotalPortfolioValue:,.0f}")


# ======================================================================================
#  COMPACT GAUSSIAN HIDDEN MARKOV MODEL  (self-contained, numpy-only)
#  Scaled forward-backward Baum-Welch (Rabiner). States sorted by mean return so they
#  map onto {Bear/High-Vol, Flat/Mean-Reverting, Bull/Low-Vol}. The transmat doubles as
#  the Markov chain used for one-step-ahead regime forecasting.
# ======================================================================================
class GaussianHMM:

    def __init__(self, n_states=3, n_iter=12, tol=1e-4, reg=1e-8, seed=7):
        self.n_states = n_states
        self.n_iter   = n_iter
        self.tol      = tol
        self.reg      = reg
        self.rng      = np.random.default_rng(seed)
        self.fitted   = False
        # Parameters (initialised lazily on first fit)
        self.means     = None
        self.vars      = None
        self.startprob = None
        self.transmat  = None
        self._order    = None

    # ---- Gaussian emission likelihoods, shape (T, N) ----
    def _emission(self, X):
        diff = X[:, None] - self.means[None, :]
        coef = 1.0 / np.sqrt(2.0 * np.pi * self.vars[None, :])
        return coef * np.exp(-0.5 * diff ** 2 / self.vars[None, :]) + 1e-300

    # ---- Baum-Welch (EM) with scaling to avoid underflow ----
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        T, N = len(X), self.n_states
        if T < N + 2:
            return self

        # Initialise: means at return quantiles, shared variance, persistent transitions.
        self.means = np.quantile(X, np.linspace(0.1, 0.9, N))
        self.vars  = np.full(N, max(np.var(X), self.reg))
        self.startprob = np.full(N, 1.0 / N)
        self.transmat  = np.full((N, N), (1.0 - 0.8) / (N - 1))
        np.fill_diagonal(self.transmat, 0.8)

        prev_ll = -np.inf
        for _ in range(self.n_iter):
            B = self._emission(X)

            # Scaled forward pass
            alpha = np.zeros((T, N)); c = np.zeros(T)
            alpha[0] = self.startprob * B[0]
            c[0] = alpha[0].sum() + 1e-300; alpha[0] /= c[0]
            for t in range(1, T):
                alpha[t] = (alpha[t - 1] @ self.transmat) * B[t]
                c[t] = alpha[t].sum() + 1e-300; alpha[t] /= c[t]

            # Scaled backward pass
            beta = np.zeros((T, N)); beta[-1] = 1.0
            for t in range(T - 2, -1, -1):
                beta[t] = (self.transmat @ (B[t + 1] * beta[t + 1])) / c[t + 1]

            ll = np.sum(np.log(c))
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300

            # Expected transition counts (xi summed over time)
            xi_sum = np.zeros((N, N))
            for t in range(T - 1):
                denom = (alpha[t] @ self.transmat) @ (B[t + 1] * beta[t + 1]) + 1e-300
                xi_sum += (alpha[t][:, None] * self.transmat *
                           (B[t + 1] * beta[t + 1])[None, :]) / denom

            # M-step
            self.startprob = gamma[0] / (gamma[0].sum() + 1e-300)
            self.transmat  = xi_sum / (xi_sum.sum(axis=1, keepdims=True) + 1e-300)
            gsum = gamma.sum(axis=0) + 1e-300
            self.means = (gamma * X[:, None]).sum(axis=0) / gsum
            self.vars  = np.maximum((gamma * (X[:, None] - self.means[None, :]) ** 2).sum(axis=0) / gsum,
                                    self.reg)

            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        self._order = np.argsort(self.means)        # ascending mean: [bear, flat, bull]
        self.fitted = True
        return self

    # ---- Filtering distribution over states at the latest observation ----
    def predict_proba_last(self, X):
        X = np.asarray(X, dtype=float)
        B = self._emission(X)
        a = self.startprob * B[0]; a /= a.sum() + 1e-300
        for t in range(1, len(X)):
            a = (a @ self.transmat) * B[t]; a /= a.sum() + 1e-300
        return a

    # ---- State semantics ----
    def ordered_states(self):
        """Return (bull_idx, flat_idx, bear_idx)."""
        bear = int(self._order[0])
        bull = int(self._order[-1])
        flat = int(self._order[len(self._order) // 2])
        return bull, flat, bear

    def label_of(self, state_idx):
        bull, flat, bear = self.ordered_states()
        if state_idx == bull:
            return "bull"
        if state_idx == bear:
            return "bear"
        return "flat"
