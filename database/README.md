# SubOpt Database Schema

Reference for the PostgreSQL schema defined in `database/schema_subopt.sql`.

The schema lives in a dedicated namespace `subopt` and contains two tables:

- `subopt.experiment_run` — one row per experiment result pickle.
- `subopt.opt_solve_event` — one row per `opt.solve()` invocation timing event.

## How to apply

```bash
python database/create_schema.py
```

Reads `notebooks/postgres_config.json` for connection info, then runs the SQL. Safe to run multiple times.

---

## Table: `subopt.experiment_run`

Captures one BFS-style experiment outcome (e.g., one `(task, seed, budget, strategy, heuristic)` combination).

| Column                | Type            | Source / Meaning |
|-----------------------|-----------------|------------------|
| `run_id`              | BIGSERIAL PK    | Auto-incrementing primary key. |
| `source_file`         | TEXT UNIQUE     | Absolute or repo-relative path of the result `.pckl` file. Used as the upsert key from the PowerBI exporter. |
| `archive`             | TEXT            | Archive ID (e.g., `"104"`, `"107"`). Comes from the JSON config (`cfg.archive`) or parsed from the path. |
| `task`                | TEXT            | Dataset / task name. Examples: `youtube`, `adult`, `caltech`, `facebook`. |
| `ground_size`         | INTEGER         | Size of the ground set used for this run (e.g. `111` for `adult`, `1000` for `youtube`). Source: `cfg.num`. |
| `seed`                | INTEGER         | Random seed used for sampling / model construction. |
| `algorithm`           | TEXT            | `EfficientBFS` or `AdaptiveEfficientBFS`. Determined by `cfg.adaptive`. |
| `strategy`            | TEXT            | Branching strategy name **as written into the result file**. May include suffixes when extra flags are on (e.g. `traditional_LS`, `traditional_NoInh`). |
| `strategy_raw`        | TEXT            | Branching strategy name **without** any flag suffix (e.g. `traditional`, `density_gap`, `lazy_binary`). Useful for grouping in BI tools. |
| `heuristic`           | TEXT            | Upper-bound heuristic kind used by the main solver. One of `ub0`, `ub2` (after canonicalization). |
| `d_mode`              | TEXT            | Sorting / `d(...)` mode used for tie-breaking in the heap (`cfg.sorting`, e.g., `"d"`, `"g"`). |
| `budget`              | DOUBLE PRECISION| Knapsack budget for this specific run. |
| `alpha`               | DOUBLE PRECISION| Target approximation ratio (e.g., `0.95`, `0.98`). Pruning compares `alpha * UB` against current LB. |
| `model_class`         | TEXT            | Python class name of the constructed model (e.g., `YoutubeCoverage`, `FacebookGraphCoverage`, `MovielensCoverage`, ...). |
| `objective_f_s`       | DOUBLE PRECISION| Final objective value `f(S)` for the returned solution. |
| `cost_c_s`            | DOUBLE PRECISION| Final cost `c(S)` of the returned solution (knapsack cost). |
| `time_s`              | DOUBLE PRECISION| Wall-clock runtime of the search, in seconds. |
| `node_count`          | BIGINT          | Number of BFS nodes popped during search. |
| `open_list_count`     | BIGINT          | Number of nodes pushed to the open list (heap). |
| `tle`                 | BOOLEAN         | `TRUE` if the run hit the timeout (Time Limit Exceeded), `FALSE` if completed normally. |
| `solution_set_size`   | INTEGER         | `len(S)` for the returned solution set. |
| `flag_local_search`   | BOOLEAN         | Mirrors `cfg.local_search`: whether the local-search refinement was enabled. |
| `flag_cascade`        | BOOLEAN         | Mirrors `cfg.cascade`: whether cascade UB filtering was enabled. |
| `flag_no_inherit`     | BOOLEAN         | Mirrors `cfg.no_inherit`: when `TRUE`, child nodes do **not** inherit parent UB. |
| `flag_adaptive`       | BOOLEAN         | Mirrors `cfg.adaptive`: whether `AdaptiveEfficientBFS` engine was used. |
| `adaptive_ratio`      | DOUBLE PRECISION| Threshold used by adaptive engine to switch between ub2/ub0 inside greedy. |
| `meta_version`        | TEXT            | Result-meta schema version (currently `"v1"`). |
| `meta_created_utc`    | TIMESTAMPTZ     | UTC timestamp when the pickle’s `meta` block was written. |
| `meta_config_path`    | TEXT            | Path of the JSON config file that produced this run. |
| `inserted_at`         | TIMESTAMPTZ     | DB insertion timestamp (default `NOW()`). |

### Indexes

- `idx_experiment_run_archive_task (archive, task)`
- `idx_experiment_run_algo (algorithm, strategy, heuristic)`
- `idx_experiment_run_budget_seed (budget, seed)`

### Upsert semantics (from PowerBI exporter)

- Conflict key: `source_file`.
- On conflict every other column is updated from `EXCLUDED.*` so re-running the exporter overwrites stale rows in place.

---

## Table: `subopt.opt_solve_event`

Captures fine-grained `opt.solve()` timing events. One row per call to a `LazyPlainOptimizer` / `LazySlicingOptimizer`’s `solve(...)`. Source CSV is `result/archive-{ID}/opt_solve_timing.csv` (when `cfg.opt_solve_log` is set).

| Column                | Type            | Source / Meaning |
|-----------------------|-----------------|------------------|
| `event_id`            | BIGSERIAL PK    | Auto-incrementing primary key. |
| `run_id`              | BIGINT FK       | Foreign key to `subopt.experiment_run.run_id`. `ON DELETE SET NULL`. May be `NULL` if the event cannot be matched back to a run. |
| `source_csv`          | TEXT            | Path of the CSV file the event was loaded from. |
| `wall_iso_utc`        | TIMESTAMPTZ     | UTC ISO timestamp recorded at the moment `solve()` returned. |
| `duration_sec`        | DOUBLE PRECISION| Duration of this single `solve()` call, in seconds (high-resolution). **Required.** |
| `optimizer_class`     | TEXT            | Python class name of the solver: `LazyPlainOptimizer`, `LazySlicingOptimizer`, or a profiled wrapper. |
| `configured_ub_type`  | TEXT            | The solver’s `ub_type` at the time of the call (e.g., `ub0`, `ub2`). Useful to confirm which UB was active. |
| `task`                | TEXT            | Task / dataset (matches `experiment_run.task`). |
| `seed`                | INTEGER         | Seed for the run that produced this event. |
| `budget`              | DOUBLE PRECISION| Budget for the run that produced this event. |
| `branching`           | TEXT            | Branching strategy name for the parent run (e.g., `traditional`). |
| `heuristic_cli`       | TEXT            | Heuristic from CLI/config (`ub0`, `ub2`, `ub0+`, `ub2+`). |
| `call_index`          | BIGINT          | Monotonic per-process call counter (1, 2, 3, ...). |
| `inserted_at`         | TIMESTAMPTZ     | DB insertion timestamp (default `NOW()`). |

### Indexes

- `idx_opt_solve_event_task (task, heuristic_cli, optimizer_class)`
- `idx_opt_solve_event_run_id (run_id)`

---

## Typical analysis queries

```sql
-- Per-run summary for a specific archive
SELECT task, seed, budget, strategy, heuristic, alpha,
       time_s, node_count, open_list_count, tle
FROM subopt.experiment_run
WHERE archive = '107'
ORDER BY task, seed, budget;

-- Average opt.solve duration by optimizer / UB
SELECT optimizer_class, configured_ub_type,
       COUNT(*) AS n_calls,
       AVG(duration_sec) AS avg_sec,
       PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_sec) AS p95_sec
FROM subopt.opt_solve_event
GROUP BY optimizer_class, configured_ub_type
ORDER BY optimizer_class, configured_ub_type;

-- Join events to runs for richer context
SELECT r.archive, r.task, r.seed, r.budget,
       e.optimizer_class, e.duration_sec
FROM subopt.opt_solve_event e
JOIN subopt.experiment_run r USING (run_id);
```
