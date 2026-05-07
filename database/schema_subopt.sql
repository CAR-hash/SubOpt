-- SubOpt analytics schema for experiment runs and optimizer timing events.
-- Safe to run multiple times.

CREATE SCHEMA IF NOT EXISTS subopt;

CREATE TABLE IF NOT EXISTS subopt.experiment_run (
    run_id BIGSERIAL PRIMARY KEY,
    source_file TEXT NOT NULL UNIQUE,
    archive TEXT,
    task TEXT,
    ground_size INTEGER,
    seed INTEGER,
    algorithm TEXT,
    strategy TEXT,
    strategy_raw TEXT,
    heuristic TEXT,
    d_mode TEXT,
    budget DOUBLE PRECISION,
    alpha DOUBLE PRECISION,
    model_class TEXT,
    objective_f_s DOUBLE PRECISION,
    cost_c_s DOUBLE PRECISION,
    time_s DOUBLE PRECISION,
    node_count BIGINT,
    open_list_count BIGINT,
    tle BOOLEAN,
    solution_set_size INTEGER,
    flag_local_search BOOLEAN,
    flag_cascade BOOLEAN,
    flag_no_inherit BOOLEAN,
    flag_adaptive BOOLEAN,
    adaptive_ratio DOUBLE PRECISION,
    meta_version TEXT,
    meta_created_utc TIMESTAMPTZ,
    meta_config_path TEXT,
    inserted_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_experiment_run_archive_task
    ON subopt.experiment_run (archive, task);
CREATE INDEX IF NOT EXISTS idx_experiment_run_algo
    ON subopt.experiment_run (algorithm, strategy, heuristic);
CREATE INDEX IF NOT EXISTS idx_experiment_run_budget_seed
    ON subopt.experiment_run (budget, seed);

CREATE TABLE IF NOT EXISTS subopt.opt_solve_event (
    event_id BIGSERIAL PRIMARY KEY,
    run_id BIGINT REFERENCES subopt.experiment_run(run_id) ON DELETE SET NULL,
    source_csv TEXT,
    wall_iso_utc TIMESTAMPTZ,
    duration_sec DOUBLE PRECISION NOT NULL,
    optimizer_class TEXT,
    configured_ub_type TEXT,
    task TEXT,
    seed INTEGER,
    budget DOUBLE PRECISION,
    branching TEXT,
    heuristic_cli TEXT,
    call_index BIGINT,
    inserted_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_opt_solve_event_task
    ON subopt.opt_solve_event (task, heuristic_cli, optimizer_class);
CREATE INDEX IF NOT EXISTS idx_opt_solve_event_run_id
    ON subopt.opt_solve_event (run_id);
