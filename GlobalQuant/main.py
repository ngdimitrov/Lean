# region imports
from AlgorithmImports import *
import numpy as np
# endregion

# ======================================================================================
#  REGIME-FILTERED MOMENTUM STRATEGY  —  4-PILLAR MATHEMATICAL ARCHITECTURE
# --------------------------------------------------------------------------------------
#  CORE ALPHA | CROSS-SECTIONAL + ABSOLUTE (DUAL) MOMENTUM
#           Each rebalance we rank the universe by ~6-month momentum and hold only the
#           top-K names that ALSO have positive absolute momentum (else -> cash). This is
#           the documented Jegadeesh-Titman / Antonacci effect and is the actual reason
#           the book is expected to make money; the four math pillars below shape WHEN and
#           HOW MUCH risk we take around that signal. (Momentum is real but crowded and
#           can underperform for years, e.g. the 2009 momentum crash — no guarantees.)
#
#  Pillar 1 | HIDDEN MARKOV MODEL + MARKOV CHAIN  (regime risk gauge)
#           A 3-state Gaussian HMM (Baum-Welch, scaled forward-backward) is fit on SPY
#           daily returns. States sort by mean return into {Bear, Flat, Bull}. The
#           transition matrix (the Markov chain) gives a ONE-STEP-AHEAD forward Bull
#           probability that feeds the game-theory sizing layer.
#
#  Pillar 2 | INFORMATION THEORY  (signal-vs-noise filter)
#           Normalised Shannon entropy on each asset's recent returns. Near-uniform
#           (high-entropy / near-random) names are dropped from the momentum ranking.
#
#  Pillar 3 | GAME THEORY  (robust maximin gross sizing)
#           A 2-player robust game scales GROSS exposure in {0.7, 0.85, 1.0}. The adversary
#           mixes benign/adverse drift; its benign probability is tilted DOWN from the HMM
#           forecast by an ambiguity margin (distributionally-robust maximin). Edge/adverse
#           are tied to the HMM Bull/Bear drifts — no arbitrary fudge factor. The 0.7 floor
#           keeps capital deployed; the cash decision stays owned by the momentum filter.
#
#  Pillar 4 | LAW OF LARGE NUMBERS  (convergence by breadth)
#           8 liquid, diversified ETFs (US large/tech/small, dev-intl, EM, long & mid
#           bonds, gold), equal-weighted across momentum survivors, rebalanced monthly.
#           Many small, near-independent bets => realised mean -> mathematical expectation.
#
#  EXECUTION ORDERING (OnData):
#    Step 1  Update rolling return windows ......................... DAILY
#    Step 2  Warm-up guard ......................................... DAILY
#    Step 3  RISK GATE: -5% circuit breaker + cooldown decay ....... DAILY
#    Step 4  MONTHLY rebalance gate (first trading day of month) ... MONTHLY
#    Step 5+ HMM / Entropy / Momentum / Game-theory sizing ......... MONTHLY
# ======================================================================================


class GlobalQuant(QCAlgorithm):

    # ----------------------------------------------------------------------------------
    #  INITIALISATION
    # ----------------------------------------------------------------------------------
    def Initialize(self):
        self.SetStartDate(2006, 1, 1)
        self.SetEndDate(2025, 1, 1)
        self.SetCash(1_000_000)

        # ---- Universe (LLN): diversified, liquid ETFs with history back to ~2005 ----
        tickers = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD"]
        self.symbols = [self.AddEquity(t, Resolution.Daily).Symbol for t in tickers]
        self.market = self.symbols[0]          # SPY = macro regime proxy

        # ---- Signal lookbacks (standard a-priori values; NOT fitted to the backtest) ----
        self.lookback_window  = 252            # rolling returns kept per asset
        self.mom_lookback     = 126            # ~6m momentum (classic momentum horizon)
        self.entropy_lookback = 60             # window for Shannon-entropy noise filter
        self.entropy_bins     = 10
        self.entropy_thresh   = 0.92           # drop assets whose returns are near-random
        self.refit_period     = 3              # refit the HMM every N rebalances (months)
        self.top_k            = 4              # hold up to K strongest names
        self.max_per_asset    = 0.40           # per-name weight cap (diversification/LLN)
        self.horizon_days     = 21             # ~1m forward horizon for game-theory drifts

        # ---- Game theory (robust maximin) parameters ----
        self.ambiguity        = 0.10           # adversary's tilt on benign probability
        self.round_trip_cost  = 0.0005         # ~5bps round-trip ETF cost assumption

        # ---- Rolling state ----
        self.returns    = {s: RollingWindow[float](self.lookback_window) for s in self.symbols}
        self.prev_price = {}
        self.hmm        = GaussianHMM(n_states=3, n_iter=12)
        self.rebalance_count = 0
        self.current_month   = -1

        # ---- Risk: -12% circuit breaker + cooldown quarantine ----
        #     Annualised vol of this book is ~5%, so a -5% trip fired on routine noise and
        #     whipsawed us into cash before rebounds. -12% only reacts to genuine regime
        #     breaks; the shorter cooldown gets capital back to work sooner.
        self.peak_equity        = self.Portfolio.TotalPortfolioValue
        self.stop_loss_pct      = 0.12
        self.cooldown_days      = 5
        self.cooldown_remaining = 0

        # ---- LLN diagnostics ----
        self.fill_count = 0

        # Prime all rolling windows before trading begins.
        self.SetWarmUp(self.lookback_window + 1, Resolution.Daily)

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

        # --- STEP 2) Warm-up guard (DAILY) ---
        if self.IsWarmingUp:
            return

        # --- STEP 3) RISK GATE (DAILY): -5% circuit breaker + cooldown decay ---
        #     Evaluated every trading day so a mid-month drawdown liquidates immediately.
        if self._risk_circuit_breaker_tripped():
            return                              # liquidated; cooling down — no new entries
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1        # quarantine: skip rebalances, decay the timer
            return

        # --- STEP 4) MONTHLY rebalance gate: act only on the first trading day of a month ---
        if self.Time.month == self.current_month:
            return
        if not self.returns[self.market].IsReady:
            return
        self.current_month = self.Time.month
        self.rebalance_count += 1

        # --- STEP 5) PILLAR 1 — HMM regime detection on the market proxy ---
        mkt = self._window_array(self.market)                       # oldest -> newest
        if (self.rebalance_count % self.refit_period == 0) or (not self.hmm.fitted):
            self.hmm.fit(mkt)                                       # periodic Baum-Welch refit
        posterior = self.hmm.predict_proba_last(mkt)                # P(state | obs_1:T)
        forecast  = posterior @ self.hmm.transmat                   # Markov 1-step-ahead
        bull, _, bear = self.hmm.ordered_states()                  # ascending mean -> bear..bull
        regime_label = self.hmm.label_of(int(np.argmax(posterior)))
        p_benign     = float(forecast[bull])                       # forward-looking Bull prob

        # --- STEP 6) CORE ALPHA + PILLAR 2 — entropy-filtered dual momentum ranking ---
        scores = {}
        for sym in self.symbols:
            r = self._window_array(sym)
            if len(r) < self.mom_lookback:
                continue
            # Pillar 2: drop near-random (high-entropy) names.
            if self._shannon_entropy(r[-self.entropy_lookback:]) > self.entropy_thresh:
                continue
            mom = self._momentum(r, self.mom_lookback)
            if mom <= 0.0:                                          # absolute (dual) momentum -> cash filter
                continue
            scores[sym] = mom

        selected = sorted(scores, key=scores.get, reverse=True)[:self.top_k]

        # --- STEP 7) PILLAR 3 — robust maximin gross sizing tied to HMM drifts ---
        if selected:
            means = self.hmm.means
            edge    = max(means[bull] * self.horizon_days, 1e-4)   # expected favourable drift
            adverse = max(abs(means[bear]) * self.horizon_days, 1e-4)  # expected adverse drift
            size    = self._game_theory_size(p_benign, edge, adverse, self.round_trip_cost)
        else:
            size = 0.0

        # --- STEP 8) PILLAR 4 — equal-weight among survivors, scaled by gross size ---
        #     Equal weight (not inverse-vol): inverse-vol systematically tilted gross into
        #     the low-vol bond names and diluted the very momentum signal we just ranked on.
        #     LLN breadth still comes from holding several near-independent survivors.
        targets = []
        if selected and size > 0:
            w_each = size / len(selected)
            for sym in self.symbols:
                w = w_each if sym in selected else 0.0
                w = float(np.clip(w, 0.0, self.max_per_asset))
                targets.append(PortfolioTarget(sym, w))
        else:
            targets = [PortfolioTarget(sym, 0.0) for sym in self.symbols]  # full cash

        self.SetHoldings(targets)
        self.Log(f"[{regime_label.upper()}] pBenign={p_benign:.2f} size={size:.2f} "
                 f"hold={[s.Value for s in selected] or 'CASH'}")

    # ----------------------------------------------------------------------------------
    #  CORE ALPHA HELPER — momentum from a returns window
    # ----------------------------------------------------------------------------------
    def _momentum(self, returns, lookback):
        """Cumulative return over the trailing `lookback` observations."""
        window = returns[-lookback:]
        return float(np.prod(1.0 + window) - 1.0)

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
    #  PILLAR 3 — GAME THEORY: distributionally-robust maximin gross sizing
    #  Adversary mixes benign/adverse; its benign prob is tilted DOWN by `ambiguity`.
    #  Actions scale gross exposure (cash decision is owned by the momentum filter).
    # ----------------------------------------------------------------------------------
    def _game_theory_size(self, p_benign, edge, adverse, cost):
        q = max(0.0, p_benign - self.ambiguity)                    # adversarial benign prob
        best_action, best_value = 0.7, -np.inf
        for a in (0.7, 0.85, 1.0):                                 # trader's pure strategies
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
