# Design Spec — SuperAggressive Volatility-Targeted Trend System

- **Дата:** 2026-06-27
- **Файл за имплементация:** `SuperAggressive/main.py`
- **Платформа:** QuantConnect LEAN (Python)
- **Цел:** Максимизиране на risk-adjusted доходност (CAR/DD, Calmar/Sharpe) на крипто кошница
  чрез trend-following система с динамично таргетиране на волатилността.

---

## 1. Контекст и мотивация

Предходната версия беше 15-минутна volatility-breakout стратегия върху BTC/ETH/SOL с
equal-weight тегла, hard stop 1.5% и trailing stop 3.0%. Тя страда от:

1. **Bear-market whipsaws** — чистият breakout без trend филтър купува пробиви и през 2022
   (BTC −65%, ETH −65%, SOL −94%) → серия дребни загуби + такси → голям drawdown (~75%).
2. **Стопове твърде тесни за крипто шума** на 15m → постоянни noise-излизания, висок turnover.
3. **Обърната асиметрия** — hard stop (1.5%) по-тесен от trailing (3.0%) → печелившите рядко дишат.

Решение (одобрен Подход B): **Volatility-targeted Trend System** на по-висок таймфрейм с
макро regime филтър, ATR chandelier изход и портфейлно таргетиране на волатилността.

---

## 2. Архитектура

Три ясно разделени, независимо тестваеми единици:

```
SuperAggressiveTrendSystem(QCAlgorithm)   ← оркестратор
  • Initialize: subscriptions, indicators, warmup, schedule
  • OnData(slice): hourly loop → update signals/exits
  • Rebalance(): извиква sizer-а и подава ордери
        │ owns N×                         │ uses 1×
        ▼                                 ▼
SymbolData (per asset)            VolatilityTargetSizer (чиста математика, без QC)
  • Donchian 24h (RollingWindow)    • вход: 60d дн. възвръщаемости/актив
  • Daily EMA200 (macro filter)     • cov матрица (numpy)
  • Daily ATR22 (chandelier)        • inverse-vol base weights
  • peak_price, active state        • scale → target annual vol
  • EntrySignal / UpdateAndCheckStop• изход: {symbol: weight}
```

**Принцип на разделение:** сигналната логика (per-asset) е изолирана от портфейлната
математика (sizer), а изпълнението на ордери е изолирано от двете. `VolatilityTargetSizer`
няма QC зависимости → тества се с чисти numpy масиви.

**Поток на данни (всеки час):**
1. `OnData` получава 1H барове → за всеки `SymbolData`:
   - ако в позиция → update `peak_price` + chandelier stop check;
   - ако не → entry check (Donchian breakout ∧ над EMA200).
2. Ако активният набор се промени (вход или изход) → `Rebalance("signal-change")`.
3. Дневен scheduled `Rebalance("daily-retarget")` ре-таргетира волатилността (cov дрейфа всеки ден).
4. `Rebalance()` пита `VolatilityTargetSizer` за тегла → подава `PortfolioTarget` списък атомарно.

---

## 3. Компоненти

### 3.1 `SymbolData` — сигнал и риск за един актив

Отговорност: капсулира индикаторите и позиционното състояние за един крипто актив.

Поддържа:
- `bars = RollingWindow[TradeBar](donchian_period + 1)` — завършени 1H барове
  (index 0 = текущ затварящ бар, индекси 1..24 = предходни 24h).
- `ema = EMA(symbol, 200, Resolution.Daily)` — макро regime филтър (auto-consolidate).
- `atr = ATR(symbol, 22, Resolution.Daily)` — за chandelier (auto-consolidate).
- `daily_closes = RollingWindow[float](61)` — за изчисляване на 60 дневни възвръщаемости.
- `active: bool` — има ли активен лонг сигнал.
- `peak_price: float` — най-висок High от входа насам.

Методи:
- `IsReady` → `bars.IsReady and ema.IsReady and atr.IsReady`.
- `OnHourBar(bar)` → `bars.Add(bar)` (от `OnData`).
- `OnDailyBar(bar)` → `daily_closes.Add(bar.Close)` (от дневен consolidator).
- `EntrySignal()` → `True` ако `close > 24h Donchian high` **И** `close > ema.Current.Value`.
- `UpdateAndCheckStop()` → обновява `peak_price = max(peak, bar.High)`;
  връща `True` ако `bar.Close <= peak_price - atr_mult * atr.Current.Value`.
- `daily_returns()` → `np.array` от ≤60 прости дневни възвръщаемости.

### 3.2 `VolatilityTargetSizer` — портфейлна математика (без QC зависимости)

Отговорност: изчислява тегла така, че целевата годишна волатилност на портфейла да е фиксирана.

`compute_weights(returns_by_symbol: dict, active_symbols: list) -> dict`:
1. Подравни матрица `R` (T×N) от дневните възвръщаемости на активните по **най-късата обща дължина**.
2. `Σ = np.cov(R, rowvar=False) * periods_per_year`  (анюализирана; `periods_per_year = 365`).
3. `base` = inverse-vol: `w_i ∝ 1/σ_i`, нормализирани да сумират 1.
4. `σ_p = sqrt(base @ Σ @ base)` — пълната cov (корелациите влизат тук).
5. `k = clip(target_annual_vol / σ_p, 0, max_gross_leverage)`.
6. `return {sym: k * base_i}` (неактивните → 0).

Гард: при сингулярна/невалидна матрица или < 30 наблюдения → fallback към inverse-vol без
scaling (`k = 1`, capped). При празен `active_symbols` → връща `{}`.

### 3.3 `SuperAggressiveTrendSystem(QCAlgorithm)` — оркестратор

- `Initialize`: дати/капитал, `SetWarmUp(210, Resolution.Daily)`, създава `sizer` и `SymbolData`
  за всеки актив (`AddCrypto(..., Resolution.Hour, Market.GDAX)` + дневен `TradeBarConsolidator`),
  и `Schedule.On(EveryDay, At(0,5), Rebalance("daily-retarget"))`.
- `OnData(slice)`: ако `IsWarmingUp` → return; иначе hourly loop (виж §2). Сетва `changed` флаг.
- `Rebalance(reason)`: събира активните ready активи → взима `daily_returns` → `sizer.compute_weights`
  → строи `[PortfolioTarget(s, w)]` за **всички** активи (неактивните → 0) → `SetHoldings(targets)`.

---

## 4. Параметри

| Параметър | Стойност | Бележка |
|---|---|---|
| Активи | BTCUSD, ETHUSD, SOLUSD @ `Market.GDAX` | SOL има данни едва от ~май 2021 |
| Период / капитал | 2021-01-01 → 2025-01-01, $100k | |
| Резолюция | `Resolution.Hour` (+ дневен consolidator) | OnData фира на час |
| Donchian период | 24 (1H бара = 24h) | пик от индекси 1..24 |
| Macro филтър | Daily EMA 200 | вход само ако `close > EMA200` |
| Chandelier ATR период | 22 (daily) | класически chandelier |
| Chandelier множител | 3.0 × ATR | стоп = `peak − 3·ATR` |
| Cov lookback | 60 дневни възвръщаемости | |
| Target годишна вол. | 0.275 (диапазон 0.25–0.30) | „агресивен" таргет |
| Анюализация | 365 (крипто 24/7, не 252) | |
| Max gross leverage | 1.0 | cash account |
| Warmup | 210 дневни бара | покрива EMA200 + cov(60) |

Всички параметри са централизирани в `Initialize` за лесен audit и tuning.

---

## 5. Edge cases

1. **SOL липсва в началото** → `SymbolData.IsReady == False` докато няма достатъчно барове →
   не дава сигнали, не влиза в cov матрицата. Без crash.
2. **Неравни истории при cov** → подравняване по най-късата обща дължина на активните;
   ако активен актив има < 30 наблюдения → изключва се от scaling (базово inverse-vol тегло, `k=1`).
3. **Сингулярна/невалидна cov матрица** → `try/except` → fallback inverse-vol без scaling.
4. **Няма активни активи** → `compute_weights` връща `{}` → всичко в кеш.
5. **Leverage cap рядко „хапе"** — крипто вол. ~60–90% год. → таргет 27.5% почти винаги намалява
   експозицията (`k < 1`). Точно това сваля DD.
6. **Turnover контрол (опционално)** — no-trade band: пропускай ордер ако `|Δweight| < 0.05`,
   за да няма churn при дневния ре-таргет.

---

## 6. Тестване

- **Unit тест на `VolatilityTargetSizer`**: синтетични възвръщаемости с известна cov →
  проверка, че `σ_p` на върнатите тегла ≈ target. Чист numpy, без QC.
- **Sanity в бектеста**: логване на реализираната портфейлна вол. vs таргет (~27.5%).
- **Регресия на DD**: сравнение на max DD спрямо baseline (старата версия) — фокус 2022.
- **Edge run**: периодът с липсваща SOL в началото не чупи нищо.

---

## 7. Известни ограничения / риск

- **Cash account (leverage = 1.0)**: в редки нискорискови режими ще сме под таргета (не можем да
  лостваме). За крипто почти не се случва. Качване над 1.0 изисква margin-способен брокер.
- **Chandelier на дневен ATR + часов peak**: стопът се преоценява с дневната ATR стойност, но peak-ът
  се движи на час. Приемливо при налична само часова гранулярност.
- **Overfitting**: параметрите паднаха спрямо стария вариант (махнати дискретни тегла и hard stop);
  Donchian/EMA/ATR/cov са стандартни стойности, не „кръгли" оптимизирани числа.
