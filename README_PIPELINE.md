# EfficientBFS Experiment Pipeline

This document describes the end-to-end flow between:

- `compute_efficient.py`
- `efficient_bfs.py`
- `notebooks/powerbi_exporter.ipynb`

## 1) Run Experiments

`compute_efficient.py` is the experiment driver.

It loads one or more dataset runs from JSON config (`--config`) or archive id (`--archive`), then loops over:

- dataset entry
- seed range
- budget range
- branching strategy
- heuristic list

For each combination, it:

1. Creates model via `model_factory`.
2. Instantiates `EfficientBFS` (or `AdaptiveEfficientBFS`).
3. Applies runtime options (alpha, sorting, UB, inheritance, cascade, etc.).
4. Calls:
   - `alg.build()`
   - `alg.optimize()`
5. Saves one `.pckl` result under `result/archive-<id>/<task>/<n>/<seed>/...`.

### Recommended commands

- By config path:
  - `python compute_efficient.py -c result/archive-104/compute_efficient.json`
- By archive id:
  - `python compute_efficient.py -a 104`

For parallel dataset execution, use:

- `python launch_compute_efficient.py -c result/archive-104/compute_efficient.json -j 4`

## 2) Solver Core

`efficient_bfs.py` contains the main branch-and-bound implementation.

Important points in this pipeline:

- UB selection is controlled by `configure_upper_bound(...)`.
- Optimizer construction is centralized in `get_optimizer(...)`.
- Optional per-`solve()` timing is recorded when `opt_solve_log_path` is set (CSV rows appended by `_record_opt_solve`).

## 3) Result Format (`.pckl`)

Each saved pickle keeps legacy result keys (for backward compatibility), e.g.:

- `S`
- `f(S)`
- `c(S)`
- `time`
- `node_count`
- `open_list_count`
- `TLE`

And now also includes structured metadata in:

- `meta` (dict), including:
  - `version`
  - `created_at_utc`
  - `config_path`
  - `task`, `archive`, `seed`, `ground_size`
  - `algorithm`, `strategy`, `strategy_raw`
  - `heuristic`, `d`, `budget`, `alpha`
  - `model_class`
  - `flags` (`local_search`, `no_inherit`, `cascade`, `adaptive`)
  - `adaptive_ratio`

This `meta` block is the preferred source for downstream analytics.

## 4) PowerBI Export

Use `notebooks/powerbi_exporter.ipynb` to convert pickle outputs into a uniform CSV.

The notebook:

1. Scans `../result/archive-*` for `.pckl` files (configurable `ARCHIVES` list).
2. Loads each pickle.
3. Prefers `res["meta"]` for canonical fields.
4. Falls back to legacy filename/path parsing when `meta` is missing.
5. Writes:
   - main CSV: `notebooks/result_101/experiment_results_powerbi.csv`
   - parse errors (if any): `..._errors.csv`

## 5) Suggested Workflow

1. Prepare archive config:
   - `result/archive-XXX/compute_efficient.json`
2. Run experiments (`compute_efficient.py` or `launch_compute_efficient.py`).
3. (Optional) collect optimizer timing CSV (`opt_solve_log`).
4. Run `notebooks/powerbi_exporter.ipynb`.
5. Import generated CSV into PowerBI.

## 6) Notes

- Keep new run configs under `result/archive-*/` if you want archive-local management.
- File names may evolve; analytics should rely on `meta` first, filename second.
- On Windows consoles with limited encoding, `TeeLogger` now degrades terminal output safely while preserving UTF-8 in log files.
