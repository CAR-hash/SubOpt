# draw_search_tree

Render an ASCII search tree from a unified-format experiment log produced by `compute_efficient.py`.

The script reads one per-run text log (mirrored from stdout by `testlogger.TeeLogger`) and prints a tree of search nodes: each node's set `s`, upper bound, pop order, prune reason, and the final solution leaf marked `[Done]`.

## Requirements

- Python 3.8+
- No third-party dependencies (stdlib only)
- A log file that contains tree-mode events (see below)

## Capturing a tree-capable log

Tree reconstruction needs high-frequency events: `NODE_PUSH`, `NODE_POP`, and `NODE_PRUNE`. Those are only written when `RunLogger.verbose=True`.

Production runs default to `runlog_verbose: false` in the JSON config to keep logs small. To capture a drawable tree, set in your `compute_efficient.json` (shared root or per `datasets[]` entry):

```json
"runlog_verbose": true
```

Then re-run with `python compute_efficient.py --archive 105` (or `--config path/to/file.json`). Logs are written under the run's log directory as `*_log.txt` (see `compute_efficient.py` for the exact filename pattern).

For ad-hoc scripts, pass `verbose=True` when constructing `RunLogger` directly:

For ad-hoc debugging or tests, attach a verbose logger directly on the algorithm:

```python
import runlog

log = runlog.RunLogger(stream=open("my_run.log", "w"), verbose=True, run_id="debug")
log.run_start(algorithm="BFSTC", task="tiny", seed=0, budget=3.0, heuristic="ub2", alpha=1.0)
alg.runlog = log
alg.build()
alg.optimize()
```

Event format and semantics are defined in `runlog.py`.

## Usage

From the repository root:

```bash
python tools/draw_search_tree.py path/to/experiment_log.txt
```

Write the tree to a file instead of stdout:

```bash
python tools/draw_search_tree.py path/to/log.txt --output tree.txt
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `log` | (required) | Path to the `*_log.txt` file to parse |
| `-o`, `--output` | stdout | Write the rendered tree to this file |
| `--max-nodes` | `500` | Stop rendering after this many nodes (avoids huge trees) |
| `--hide-pruned` | off | Omit nodes with `status=pruned_pre` that were never popped |

## Example output

```
# algorithm=BFSTC task=tiny seed=0 budget=3.0 heuristic=ub2 alpha=1.0
# best incumbent f(S) = 8  s = [1]
# final S = [1]

(Root) (1) [] (ub: 10)
+-- (2) [1] (ub: 8) [Done]
\-- [2] (ub: 4) [pruned: alpha_lb]
```

Line meanings:

- **`(N)`** — 1-based order in which the node was popped from the open list (`NODE_POP`).
- **`s`** — Canonical set representation from the log (e.g. `[1,2,3]`).
- **`(ub: …)`** — Upper bound at push/pop time when available.
- **`[pruned: reason]`** — Terminal prune (`NODE_PRUNE`) or pre-push rejection (`status=pruned_pre`).
- **`[Done]`** — Popped leaf whose `s` matches `final_s` from `RUN_END`.

The header summarizes `RUN_START` metadata, the best `INCUMBENT` seen, and `final_s` from `RUN_END`.

Children are ordered by pop sequence (earlier pops first), then by node id.

## Legacy logs

Older runs (before `runlog_verbose`) wrote human-readable **`[POP]`** / **`[EVAL]`** lines instead of unified events. The tool detects those automatically and reconstructs the tree by:

1. Treating each **`[POP]`** as a visited node (matched to earlier survived **`[EVAL]`** children by set `s`).
2. Treating **`[EVAL]`** blocks until the next pop as children of that node (`✅ [SURVIVED]` / `❌ [KILLED]` or English equivalents).

Supported variants include the localized trace (`🟢 [POP] Node #…`, `当前集合 S:`), the English `efficient_bfs` trace (`[POP] node=…`, `[EVAL] s=…`), and `EfficientBranchAndBound` (`[POP] node=… stack=…` plus `[EVAL]` / `[KILLED]` / `[SURVIVED]` when `runlog_verbose` is on).

**Note:** `EfficientBranchAndBound` logs from before `runlog` integration (or with `NullRunLogger` still attached) may contain only the final `[OK]` summary line — those files cannot be tree-drawn.

## Input format (unified)

The parser reads [logfmt](https://brandur.org/logfmt)-style lines:

```
[EVENT] key1=value1 key2=value2 ...
```

Only lines matching this pattern are consumed; other stdout noise (e.g. `trigger count:...`) is ignored.

Events used to build the tree:

| Event | Role |
|-------|------|
| `RUN_START` | Run metadata in the header |
| `NODE_PUSH` | Create a child (`parent_id`, `id`, `s`, `ub`, `status`) |
| `NODE_POP` | Mark pop order and refresh `s` / `ub` |
| `NODE_PRUNE` | Attach prune `reason` to a node |
| `INCUMBENT` | Best objective for the header |
| `RUN_END` | Final solution set `final_s` and `[Done]` annotation |

Node identity is by numeric `id`, not by `s`: the same set can appear on multiple nodes (e.g. lazy-binary branching).

## Limitations

- **Verbose logs required** — Without `NODE_PUSH` / `NODE_POP` / `NODE_PRUNE`, the tree will be empty or incomplete.
- **Truncation** — Large runs hit `--max-nodes`; increase the limit if you need the full tree.
- **Lossy logs** — If a `NODE_POP` appears without a matching `NODE_PUSH`, the tool synthesizes a minimal node so something still renders.
- **One run per file** — Pass a single experiment log; multi-run files are not split automatically.

## Tests

```bash
python -m unittest tests.test_draw_search_tree -v
```

The suite covers parsing, tree construction, rendering, and an end-to-end round-trip with `BFSTC` when `filter_search` is importable.
