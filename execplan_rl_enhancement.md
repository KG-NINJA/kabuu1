# RL logging & weekly review enhancements

This ExecPlan is a living document and must follow the repository guidelines described in PLANS.md. Keep it updated as work proceeds.

## Purpose / Big Picture

We need richer reinforcement learning (RL) telemetry and automated weekly reviews so pipeline operators can understand how NVDA predictions are performing and how to improve them. After implementing this plan, the system will write detailed CSV RL logs, emit JSON daily summaries, and produce weekly review reports that can be generated both inside the pipeline and via CI on Fridays. Operators can inspect `reinforcement_learning.log`, daily files in `data/rl_history/`, and weekly reports in `data/weekly_reviews/` for insights.

## Progress

- [x] (2025-02-14 11:50Z) Drafted ExecPlan with context and intended changes.
- [x] (2025-02-14 12:10Z) Implemented `src/rl_enhancement.py` with logger, weekly generator, and integration helper.
- [x] (2025-02-14 12:25Z) Wired the logger/generator into `src/prediction_pipeline.py` (imports, `__init__`, outcome updates, weekly review hooks).
- [x] (2025-02-14 12:35Z) Added `.github/workflows/stock-prediction.yml` with directory creation, LLM prompts, and new weekly/daily summary steps; created local data directories with placeholders.
- [x] (2025-02-14 12:45Z) Executed module + pipeline tests (rl_enhancement demo, pipeline cycle, JSON validation) and captured results.
- [x] (2025-02-14 12:47Z) Reviewed generated log/daily summary/weekly review artifacts for correctness.
- [x] (2025-02-14 12:55Z) Finalized documentation/plan updates and prepared for commit.

## Surprises & Discoveries

- Observation: Installing dependencies via `pip install -r requirements.txt` failed because the `empyrical` package still uses `configparser.SafeConfigParser` which was removed in Python 3.12. Evidence: pip stack trace referencing `AttributeError: module 'configparser' has no attribute 'SafeConfigParser'`.

## Decision Log

- Decision: Treat the missing `.github/workflows/stock-prediction.yml` as a required workflow and (re)create it with the requested steps instead of altering other workflows. Rationale: the user explicitly names that file and steps such as "Generate LLM Prompts" and "Create directories" that are not present elsewhere, so introducing the workflow ensures their requirements are satisfied without disturbing other automations. Date/Author: 2025-02-14 / assistant.

## Outcomes & Retrospective

- Completed all requested logging enhancements, workflow automation, and testing. RL outcomes now populate an extended CSV log, daily summaries, and weekly review JSON artifacts locally and via CI. Remaining work: monitor workflow execution after merge to ensure scheduled conditions behave as expected.

## Context and Orientation

- `src/prediction_pipeline.py` orchestrates predictions, handles reinforcement learning outcomes via `NvdaReinforcementHub`, and runs weekly reviews using `AdaptiveNVDALearner`. It persists metrics to `performance_metrics.json` and pending predictions to `nvda_learning/pending_predictions.json`.
- `src/nvda_reinforcement.py` defines `AdaptiveNVDALearner` and `NvdaReinforcementHub`. Logging currently writes a minimal CSV header `ticker,predicted,actual,reward`.
- There is no module yet for advanced RL logging, so we will create `src/rl_enhancement.py` to encapsulate CSV logging with richer schema, daily summary JSON export, and weekly review generation.
- GitHub Actions currently lack a `stock-prediction.yml` workflow. We will create/modify it so the scheduled job both runs the pipeline (existing commands) and executes the new logging/report steps defined in the user instructions.
- Data directories such as `data/rl_history/`, `data/daily_summaries/`, and `data/weekly_reviews/` do not exist by default; they must be created both locally (for developers) and inside the workflow before logs are written.

## Plan of Work

1. **Design `AdvancedReinforcementLogger`:**
   - Accept a CSV `log_path` (default `reinforcement_learning.log`). Ensure file exists with extended header. Provide methods:
     - `record_detailed_outcome(...)` capturing ticker, timestamp, predicted/actual, error %, reward, confidence, model metadata, hyperparameters, and qualitative flags. Append to CSV, updating in-memory cache for summaries.
     - `export_daily_summary(date: Optional[date])` to compute per-day aggregates (total predictions, average reward/error, best/worst rewards, per-model/per-ticker stats) and save JSON to `data/rl_history/` (and optionally `data/daily_summaries/` if separate). Return `Path` to JSON file.
     - `load_recent_entries(days: int)` to fetch rows from last N days for weekly analysis.
   - Manage directories via `Path` operations and ensure idempotent.

2. **Implement `WeeklyReviewGenerator`:**
   - Use `AdvancedReinforcementLogger` to gather past 7 days of entries.
   - Aggregate by ticker and model_version (counts, mean reward/error, best/worst, win rate defined by reward >= threshold).
   - Generate recommendations heuristically (e.g., highlight underperforming tickers, suggest lookback adjustments). Provide methods `generate_weekly_review()` returning dict and `generate_and_save()` writing JSON file to `data/weekly_reviews/weekly_review_<ISO week>.json`.

3. **Provide `integrate_with_pipeline()` helper** describing hooking (maybe optional) though spec wants function; we can implement as convenience to patch pipeline or just log message.

4. **Modify `src/prediction_pipeline.py`:**
   - Add optional imports for `AdvancedReinforcementLogger` and `WeeklyReviewGenerator`, falling back to `None` if module missing.
   - In `__init__`, instantiate logger + generator when available, storing `self.rl_logger` and `self.weekly_review_generator`. Ensure `reinforcement_learning.log` path is resolved relative to repo root (use `Path`).
   - In `update_with_actual_price`, after RL hub records reward, call `self.rl_logger.record_detailed_outcome(...)` with prediction metadata, including config-based lookback days and learning rate; include fallback values.
   - In `weekly_model_review`, after existing RL hub review, call `self.rl_logger.export_daily_summary()` and `self.weekly_review_generator.generate_and_save()`, logging results and printing recommendations.

5. **Workflow updates:**
   - Create `.github/workflows/stock-prediction.yml` (or update if it exists). The workflow should define job `predict` with steps: checkout, Python setup, dependencies, create directories (with new data subdirs), run pipeline, generate LLM prompts (existing snippet), new steps for weekly review (Friday only) and daily summary (always). Use inline Python scripts as provided.

6. **Data directories:**
   - Locally ensure directories exist using `mkdir -p data/rl_history data/weekly_reviews data/daily_summaries` (maybe part of workflow but also ensure tracked? not necessary but we may commit `.gitkeep`? Not requested, only ensure directories exist (maybe by script?). Could create empty `.gitkeep` for each so git tracks? Unless spec demands create directories; we can add `.gitkeep` to ensure they exist? We'll add `.gitkeep` (touch). Alternatively rely on instructions? We'll create directories with placeholder `.gitkeep` so repository has directories.

7. **Testing:**
   - Run `python3 src/rl_enhancement.py` (should exercise module's self-test or CLI we implement) verifying sample logs/summaries.
   - Optionally run pipeline command `python3 -m src.prediction_pipeline --mode cycle --run-prediction --run-actuals` (may need network; handle gracefully?). We'll attempt and capture results; if fails due to data fetch, note.
   - Validate JSON output with `python3 -m json.tool` as described.

8. **Finalize:**
   - Review new/modified files, ensure formatting, add citations for final response, run tests recorded, and commit with clear message. Then create PR via `make_pr`.

## Concrete Steps

1. Create `src/rl_enhancement.py` implementing the classes and helper function.
2. Update `src/prediction_pipeline.py` per tasks 2-1 through 2-4.
3. Add `.gitkeep` (or ensure creation) for `data/rl_history`, `data/weekly_reviews`, and `data/daily_summaries`.
4. Add/modify `.github/workflows/stock-prediction.yml` including the new steps for directory creation, weekly review, and daily summary.
5. Run specified Python commands/tests and inspect outputs.
6. Review git status, format, and commit.

## Validation and Acceptance

- Running `python3 src/rl_enhancement.py` should create/append to `reinforcement_learning.log`, produce a daily summary JSON in `data/rl_history/`, and write a weekly review JSON file.
- Running `python3 -m src.prediction_pipeline --mode cycle --run-prediction --run-actuals` should complete without errors (allowing warnings if network/data unavailable) and should log RL outcomes in the extended CSV plus daily summary export when weekly review is triggered manually.
- The GitHub Action `stock-prediction.yml` should include steps to generate weekly reviews on Fridays and daily summaries every run.
- Directories `data/rl_history`, `data/weekly_reviews`, and `data/daily_summaries` exist in the repository with placeholder files to ensure creation.

## Idempotence and Recovery

- Logger/generator methods must create directories if missing and avoid duplicate headers.
- Workflow steps use `mkdir -p` to allow repeated runs safely.
- Local scripts should handle missing logs gracefully by starting with empty datasets.

## Artifacts and Notes

- Record sample JSON outputs and log snippets during testing for future reference.

## Interfaces and Dependencies

- `AdvancedReinforcementLogger.record_detailed_outcome` signature: `(self, ticker: str, predicted_price: float, actual_price: float, confidence: float, model_version: str, lookback_period: int, learning_rate: float, data_quality: str = "good", reward: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]` returning the structured entry.
- `AdvancedReinforcementLogger.export_daily_summary(self, target_date: Optional[date] = None) -> Path`.
- `WeeklyReviewGenerator.generate_weekly_review(self, days: int = 7) -> Dict[str, Any]`.
- `WeeklyReviewGenerator.generate_and_save(self, days: int = 7) -> Path`.
- `integrate_with_pipeline(pipeline: PredictionPipeline) -> None` will configure `pipeline.rl_logger` and `pipeline.weekly_review_generator` when possible.
