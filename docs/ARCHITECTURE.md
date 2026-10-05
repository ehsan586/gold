# ASTRA TRADER — architecture, audit and spec-to-code map

## 1. Audit of the existing repository (done before changing anything)
| Finding | Decision |
|---|---|
| 7 analysis-only agents, signed Risk-Governor token, single Execution Engine, paper broker, backtester, 105 tests | **kept** unchanged in behaviour |
| Each cycle fetched candles ad hoc; no temporal contract; journal was free text | **replaced** by DataService + MarketSnapshot + hash-chained event log |
| Default behaviour sent REAL orders to the MT5 demo account | **changed**: default is SIMULATION (virtual account); real demo orders are explicit opt-in (`MT5_DEMO`) behind the Safety Gate |
| No regime, forecast, diversity, consistency, drift, champion/challenger, experiments, replay | **added** (only these two new intelligence components: Market Regime + Forecast) |
| Table `market_snapshots` never used | left (schema compatibility); snapshots are now represented by metadata+hash in events |
| `llm.py` advisory | kept, optional, disabled by default, cannot place orders |
| Mocked things presented as real | none found; paper/shadow brokers are labelled as such everywhere (`mode`, dashboard badges) |

## 2. Spec-to-code map
| Spec § | Implementation | Verified by |
|---|---|---|
| 3 target architecture / 18 Safety Gate order | `engine.py`, `validator.py`, `safety_gate.py`, `execution.py` | `test_gate_requires_validator_approval_then_risk`, `test_every_trade_was_validated_and_gate_approved_before_the_action` |
| 4 seven specialists (+Regime, Forecast) | `agents.py`, `regime.py`, `forecast.py` | `test_agents`, `test_astra_core` |
| 5 Market Regime (+measurable) | `regime.py` (`evaluate_regimes`) | `TestRegime` |
| 6/7 Forecast system + baselines | `forecast.py` (3 members, 3 baselines, combiner, uncertainty, calibration, scoreboard) | `TestForecast` |
| 8 Diversity monitor | `diversity.py` | `test_high_agreement_low_diversity_vs_high_diversity` |
| 9 Consistency engine | `consistency.py` | `TestDiversityConsistency` |
| 10/11 Data system, temporal contract, leakage tests | `snapshot.py`, `replay.py` | `TestTemporalContract` (8 tests incl. perturbed-future tests) |
| 12 Unit normalisation (cent accounts) | `units.py` | `TestUnits` |
| 13 State manager | `supervisor.SystemStateManager` | `test_state_manager` |
| 14/15/16 Supervisor, watchdog, recovery | `supervisor.py` (+ `/api/v1/health`) | `TestAuditAndSupervision`, `test_system_state_degrades_and_recovers` |
| 17 Validator | `validator.py` | `test_rejections` |
| 19 Execution interface (default non-executing) | `execution.py` modes, `broker_sim.ShadowBroker` | `TestExecutionModes`, `test_real_orders_impossible_unless_mt5_demo_mode` |
| 20 Raw + experience memory | `events.py`, `memory.py` | `test_experience_memory_and_forecast_resolution` |
| 21–24 Learning manager, drift, champion/challenger, rollback | `learning_manager.py`, `drift.py`, `forecast.ModelRegistry` | `TestDriftAndLearning`, `test_registry_champion_stable_rollback` |
| 25 Replay engine | `replay.py` | `test_replay_is_deterministic_without_lookahead` |
| 26 Event sourcing | `events.py` (hash chain, deterministic digest) | `test_event_chain_and_tamper_detection` |
| 27/28 Experiment lab, ablation | `experiments.py` | `test_ablation_records_experiment_and_guards_holdout` |
| 29 Calibration | `forecast.calibration_info`, reliability bins | `test_no_fake_probability_and_honest_uncertainty` |
| 30 Observability | `metrics.py`, `/api/v1/metrics` | API tests |
| 31–35 Mission-control UI | `dashboard.html` | headless-browser screenshots + `test_dashboard_matches_the_spec_and_contains_no_fake_telemetry` |

## 3. Event types
`MARKET_SNAPSHOT_RECEIVED, SNAPSHOT_REJECTED, AGENT_ANALYSIS_COMPLETED, REGIME_CLASSIFIED, FORECAST_GENERATED, FORECAST_RESOLVED, CONSISTENCY_UPDATED, DECISION_CREATED, VALIDATION_PASSED, VALIDATION_REJECTED, SAFETY_GATE, SIMULATION_ACTION, EXECUTION_ACTION, MODEL_EVALUATED, DRIFT_DETECTED, RECOVERY_STARTED, RECOVERY_COMPLETED, SYSTEM_STATE, MODULE_ISOLATED, EXPERIMENT, MODEL_STATUS, STARTUP`.
Each row: `seq, event_id (hash), ts, snapshot_id, component, event_type, payload, version, prev_hash, hash`.

## 4. What is deliberately NOT claimed
* No forecast skill is claimed. Verdicts are `INSUFFICIENT_DATA` until `forecast_min_samples` (100) resolved forecasts exist, then a paired test vs the baselines decides.
* The consistency score and the uncertainty score are transparent heuristics (formulas shown in the UI), not probabilities.
* Regime "confidence" is a margin score. A probability is shown only after calibration on enough resolved forecasts.
* `broker_mt5.py`, `mql5/*.mq5` could not be executed in the build environment.
