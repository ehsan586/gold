# ASTRA TRADER — AI • ANALYSIS • RESEARCH  (gold / XAUUSD on MetaTrader 4 (read-only feed))

Research- and simulation-first. **The final launcher is simulation-only and the MT4 bridge is read-only; it does not send orders.**
Nothing here predicts the market reliably or guarantees profit; the system measures its own forecasts against simple baselines and says so when it cannot tell.

**Start here:** `docs/GUIDE_FA.md` (Persian step-by-step guide) · `docs/ARCHITECTURE.md` (design, spec-to-code map, audit).

```
python -m goldbot demo        # offline demo on synthetic data + the ASTRA dashboard (no MT5 needed)
python -m goldbot selftest    # replay determinism + no-look-ahead self-test
python -m goldbot mt4check    # test the read-only MT4 FILE_COMMON feed
python -m goldbot run         # live MT4 feed; SIMULATION (virtual account)
python -m goldbot backtest --csv XAUUSD_M1.csv --split 0.7
python -m goldbot ablation  --csv XAUUSD_M1.csv      # incremental contribution of every component (out-of-sample)
python -m goldbot replay    --csv XAUUSD_M1.csv --verify
python -m unittest discover -s tests -v              # 180 tests
```

Pipeline of one decision cycle
```
MT4 read-only feed -> DataService -> DataValidator -> immutable MarketSnapshot (temporal contract)
  -> 7 specialists + Regime + Forecast (all read the SAME snapshot)
  -> Diversity monitor -> Consistency engine -> weighted Decision -> context refinement
  -> Validator (signed approval) -> Safety Gate (+ Risk Governor token) -> Action Interface (DISABLED | SIMULATION)
  -> Experience memory / hash-chained event log -> Learning manager (proposes only) -> Dashboard
```
Requires Python 3.10+. The default MT4 bridge uses only the standard library. The legacy MT5 connector remains optional.
