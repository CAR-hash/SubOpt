import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import compute_efficient_config


def _resolve_source_config(args: argparse.Namespace) -> Path:
    if args.archive:
        return compute_efficient_config.resolve_config_path_for_archive(args.archive)
    return Path(args.config).resolve()


def _single_dataset_payload(cfg) -> dict:
    data = asdict(cfg)
    task = data.pop("task")
    num = data.pop("num")
    data["datasets"] = [{"task": task, "num": num}]
    return data


def _write_job_config(job_dir: Path, idx: int, payload: dict) -> Path:
    job_cfg = job_dir / f"job_{idx:02d}_{payload['datasets'][0]['task']}.json"
    job_cfg.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return job_cfg


def _run_job(python_exec: str, repo_root: Path, cfg_path: Path, out_log_path: Path, dry_run: bool) -> int:
    cmd = [python_exec, "compute_efficient.py", "-c", str(cfg_path)]
    if dry_run:
        print("[DRY-RUN]", " ".join(cmd))
        return 0

    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    out_log_path.write_text(
        f"$ {' '.join(cmd)}\n\n[stdout]\n{proc.stdout}\n[stderr]\n{proc.stderr}\n",
        encoding="utf-8",
    )
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch compute_efficient as isolated subprocess jobs (one dataset per job).",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("-c", "--config", metavar="PATH", help="config JSON path")
    src.add_argument("-a", "--archive", metavar="ID", help="archive id to resolve config from")
    parser.add_argument(
        "-j",
        "--max-workers",
        type=int,
        default=1,
        help="number of parallel subprocesses (default: 1)",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="python executable for subprocesses (default: current interpreter)",
    )
    parser.add_argument("--dry-run", action="store_true", help="print commands without running")
    args = parser.parse_args()

    if args.max_workers < 1:
        parser.error("--max-workers must be >= 1")

    repo_root = Path(__file__).resolve().parent
    source_cfg = _resolve_source_config(args)
    run_cfgs = compute_efficient_config.load_efficient_run_configs(source_cfg)
    if not run_cfgs:
        raise RuntimeError("No dataset jobs found in config.")

    archive_id = str(run_cfgs[0].archive)
    job_dir = repo_root / "result" / f"archive-{archive_id}" / "launcher-jobs"
    job_dir.mkdir(parents=True, exist_ok=True)

    print(f"Source config: {source_cfg}")
    print(f"Jobs: {len(run_cfgs)} | max_workers={args.max_workers}")
    print(f"Per-job configs/logs: {job_dir}")

    jobs = []
    for i, cfg in enumerate(run_cfgs, start=1):
        payload = _single_dataset_payload(cfg)
        cfg_path = _write_job_config(job_dir, i, payload)
        out_log_path = job_dir / f"job_{i:02d}_{cfg.task}.launcher.log"
        jobs.append((i, cfg.task, cfg_path, out_log_path))

    failures = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {
            ex.submit(
                _run_job,
                args.python,
                repo_root,
                cfg_path,
                out_log_path,
                args.dry_run,
            ): (idx, task, out_log_path)
            for idx, task, cfg_path, out_log_path in jobs
        }
        for fut in as_completed(futs):
            idx, task, out_log = futs[fut]
            code = fut.result()
            status = "OK" if code == 0 else f"FAIL({code})"
            print(f"[job {idx:02d}] task={task} -> {status} | log={out_log}")
            if code != 0:
                failures.append((idx, task, code))

    if failures:
        print("Some jobs failed:")
        for idx, task, code in failures:
            print(f"  - job {idx:02d} task={task}: exit_code={code}")
        return 1

    print("All jobs completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
