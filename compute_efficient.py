import argparse
import os
import pickle
import random
import sys
from datetime import datetime, timezone

import testlogger

import numpy as np

import algorithm_factory
import compute_efficient_config
import model_factory


def _wrap_result_with_meta(
    res,
    *,
    config_path,
    cfg,
    strategy_name_for_file,
    strategy_raw,
    heuristic,
    seed,
    budget,
    model,
):
    """
    Backward-compatible result wrapper: preserve original fields and attach structured metadata.
    """
    out = dict(res)
    out["meta"] = {
        "version": "v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "task": cfg.task,
        "archive": str(cfg.archive),
        "seed": int(seed),
        "ground_size": int(cfg.num),
        "algorithm": cfg.algorithm,
        "strategy": strategy_name_for_file,
        "strategy_raw": strategy_raw,
        "heuristic": heuristic,
        "d": cfg.sorting,
        "budget": float(budget),
        "alpha": float(cfg.alpha),
        "model_class": model.__class__.__name__,
        "flags": {
            "local_search": bool(cfg.local_search),
            "no_inherit": bool(cfg.no_inherit),
            "cascade": bool(cfg.cascade),
            # Kept for back-compat with older readers; ``algorithm`` is the source of truth.
            "adaptive": bool(cfg.adaptive),
        },
        "adaptive_ratio": float(cfg.adaptive_ratio),
    }
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run EfficientBFS experiments from a JSON config file (one or more datasets).",
    )
    src = parser.add_mutually_exclusive_group(required=False)
    src.add_argument(
        "-c",
        "--config",
        metavar="PATH",
        help="JSON experiment config (see configs/compute_efficient.example.json)",
    )
    src.add_argument(
        "-a",
        "--archive",
        metavar="ID",
        help="run the experiment set for archive ID: load result/archive-{ID}/compute_efficient.json "
        "if present, else configs/compute_efficient_archive{ID}.json",
    )
    parser.add_argument(
        "--print-example-config",
        action="store_true",
        help="print default/example JSON to stdout and exit",
    )
    args_cli = parser.parse_args()

    if args_cli.print_example_config:
        sys.stdout.write(compute_efficient_config.default_efficient_config_json())
        raise SystemExit(0)

    if not args_cli.config and not args_cli.archive:
        parser.error(
            "specify --config PATH or --archive ID (or use --print-example-config)"
        )
    if args_cli.config and args_cli.archive:
        parser.error("use only one of --config PATH or --archive ID, not both")

    if args_cli.archive:
        config_path = compute_efficient_config.resolve_config_path_for_archive(args_cli.archive)
        print(f"Archive {args_cli.archive!r}: using config {config_path}")
    else:
        config_path = args_cli.config

    run_configs = compute_efficient_config.load_efficient_run_configs(config_path)
    n_datasets = len(run_configs)
    print(f"Loaded {n_datasets} dataset(s) from {config_path}")

    for di, cfg in enumerate(run_configs):
        print(f"\n{'=' * 60}\nDataset {di + 1}/{n_datasets}: task={cfg.task!r}, num={cfg.num}\n{'=' * 60}")

        end_point = cfg.budget_start + (cfg.budget_num_points - 1) * cfg.budget_interval
        bds = np.linspace(start=cfg.budget_start, stop=end_point, num=cfg.budget_num_points)

        root_dir = os.path.join("./result", f"archive-{cfg.archive}")

        # Algorithms without a branching dimension still need to run once per
        # (seed, budget, heuristic); use a single placeholder strategy so the loop
        # below stays uniform.
        if cfg.supports_branching:
            strategies_for_loop = list(cfg.branching)
        else:
            # Filesystem-safe placeholder (no slashes); BFSTC/EBB don't use branching.
            strategies_for_loop = ["none"]
            if cfg.branching:
                print(
                    f"[INFO] Algorithm {cfg.algorithm!r} does not use 'branching'; "
                    f"ignoring configured value {cfg.branching}."
                )

        print(f"[START] Experiments | Task: {cfg.task} | Algorithm: {cfg.algorithm} | Strategies: {strategies_for_loop}")

        for seed in range(cfg.start_seed, cfg.stop_seed):
            for budget in bds:

                print(f"\n--- Task {cfg.task!r} | Seed: {seed:02d} | Budget: {budget:4.1f} ---")

                for strategy in strategies_for_loop:
                    for heuristic in cfg.heuristics:
                        log_dir = f"./result/archive-{cfg.archive}/{cfg.task}/"
                        os.makedirs(log_dir, exist_ok=True)
                        log_prefix = f"Adapt{cfg.adaptive_ratio}_" if cfg.algorithm == "AdaptiveEfficientBFS" else ""
                        log_strategy_token = strategy if cfg.supports_branching else cfg.algorithm
                        log_path = os.path.join(
                            log_dir, f"{log_prefix}{log_strategy_token}_{budget}_{heuristic}_log.txt")

                        with testlogger.TeeLogger(log_path):
                            random.seed(seed)

                            model = model_factory.model_factory(cfg.task, cfg.num, seed, budget, knap=True)

                            alg = algorithm_factory.build_algorithm(
                                cfg, model=model, heuristic=heuristic, strategy=strategy,
                            )

                            strategy_name_for_file = strategy
                            if cfg.supports_branching:
                                if cfg.local_search:
                                    strategy_name_for_file += "_LS"
                                if cfg.no_inherit:
                                    strategy_name_for_file += "_NoInh"

                            if cfg.opt_solve_log and hasattr(alg, "opt_solve_log_path"):
                                alg.opt_solve_log_path = cfg.opt_solve_log
                                alg.opt_solve_log_meta = {
                                    "task": cfg.task,
                                    "seed": int(seed),
                                    "budget": float(budget),
                                    "branching": strategy,
                                    "heuristic_cli": heuristic,
                                }

                            alg.build()
                            res = alg.optimize()

                            node_cnt = res.get('node_count', -1)
                            time_cost = res.get('time', 0.0)
                            function_val = res.get('f(S)', 0.0)
                            print(f"trigger count:{res.get('probing_trigger_count', 0)}, depth_list:{res.get('probing_trigger_depth_list', [])}, max_depth:{res.get('max_depth', 0)}")
                            if not res.get('TLE', False):
                                print(f"  [OK] Algo: {cfg.algorithm} | Strategy: {strategy_name_for_file:<14s} | h:{heuristic:<4s} | f(S):{function_val} | Nodes: {node_cnt:<6d} | Time: {time_cost:.2f}s")
                            else:
                                print(f"  [TLE] Algo: {cfg.algorithm} | Strategy: {strategy_name_for_file:<14s} | h:{heuristic:<4s} | f(S):{function_val} | Nodes: {node_cnt:<6d} | TLE: {time_cost:.2f}s")

                            save_dir = os.path.join(root_dir, cfg.task, str(cfg.num), str(seed))
                            os.makedirs(save_dir, exist_ok=True)

                            # Filename embeds the algorithm name so siblings from different
                            # algorithms don't collide. Metadata in the pickle remains the
                            # source of truth for downstream parsing.
                            filename = "{}-{}-{}-{}-{}-{}-{}.pckl".format(
                                cfg.algorithm,
                                strategy_name_for_file,
                                heuristic,
                                cfg.sorting,
                                budget,
                                cfg.alpha,
                                model.__class__.__name__,
                            )
                            save_path = os.path.join(save_dir, filename)

                            wrapped = _wrap_result_with_meta(
                                res,
                                config_path=config_path,
                                cfg=cfg,
                                strategy_name_for_file=strategy_name_for_file,
                                strategy_raw=strategy,
                                heuristic=heuristic,
                                seed=seed,
                                budget=budget,
                                model=model,
                            )
                            with open(save_path, "wb") as wrt:
                                pickle.dump(wrapped, wrt)

    print("\n[DONE] All experiments completed!")
