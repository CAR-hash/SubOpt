"""
Render a search tree from a unified-format log file.

Reads one ``compute_efficient.py`` log file (per-run text log produced by
``testlogger.TeeLogger``) and prints an ASCII tree showing each search node:

* its set ``s``,
* its upper bound,
* whether it was popped (and in what order ``(N)``),
* whether it was pruned, and
* the final solution leaf annotated with ``(Done)``.

The unified events (``RUN_START`` / ``NODE_PUSH`` / ``NODE_POP`` / ``NODE_PRUNE`` /
``RUN_END``) are documented in :mod:`runlog` and ``result/README_logs.md``.

Usage
~~~~~

.. code-block:: shell

    python tools/draw_search_tree.py path/to/log.txt
    python tools/draw_search_tree.py path/to/log.txt --max-nodes 200 --output tree.txt
    python tools/draw_search_tree.py path/to/log.txt --hide-pruned

Tree-mode events are emitted only when ``RunLogger.verbose=True`` at run time.
In ``compute_efficient.py`` runs, set ``"runlog_verbose": true`` in the JSON
config (or pass ``verbose=True`` when constructing ``RunLogger`` manually).
Production configs default to ``false`` to keep logs small.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Logfmt parser
# ---------------------------------------------------------------------------

# ``[EVENT] k1=v1 k2=v2`` — event tag at the start of the line, then key=value
# pairs where the value is either a bare token (no spaces, no '=') or a
# double-quoted string.
_EVENT_RE = re.compile(r"^\s*\[(?P<event>[A-Z_]+)\]\s*(?P<rest>.*)$")
_KV_RE = re.compile(
    r"(?P<k>[A-Za-z_][A-Za-z0-9_]*)=(?:\"(?P<qv>(?:[^\"\\]|\\.)*)\"|(?P<v>\S+))"
)


def parse_event_line(line: str) -> Optional[Tuple[str, Dict[str, str]]]:
    """Return ``(event, fields)`` for a unified-format line; ``None`` otherwise."""
    m = _EVENT_RE.match(line)
    if not m:
        return None
    fields: Dict[str, str] = {}
    for kv in _KV_RE.finditer(m.group("rest")):
        v = kv.group("qv")
        if v is None:
            v = kv.group("v")
        else:
            # Unescape the same way runlog._fmt_value escaped: \\ and \"
            v = v.replace("\\\\", "\\").replace("\\\"", "\"")
        fields[kv.group("k")] = v
    return m.group("event"), fields


def iter_events(path: str) -> Iterable[Tuple[str, Dict[str, str]]]:
    with open(path, "r", encoding="utf-8") as fp:
        for raw in fp:
            parsed = parse_event_line(raw.rstrip("\n"))
            if parsed is not None:
                yield parsed


# ---------------------------------------------------------------------------
# Legacy log adapter (pre-unified ``[POP]`` / ``[EVAL]`` traces)
# ---------------------------------------------------------------------------

# ``🟢 [POP] Node #1 | Depth: 0 | Cost: 0/13.0 | UB: 652.41``
_LEGACY_POP_ZH_RE = re.compile(
    r"\[POP\]\s*Node\s*#(?P<node>\d+)\s*\|\s*Depth:\s*(?P<depth>\d+)\s*\|"
    r"\s*Cost:\s*[\d.]+/[\d.]+\s*\|\s*UB:\s*(?P<ub>[\d.]+)",
    re.IGNORECASE,
)
# ``[POP] node=1 depth=0 cost=0/13.0 ub=652.41``
_LEGACY_POP_EN_RE = re.compile(
    r"\[POP\]\s*node=(?P<node>\d+)\s+depth=(?P<depth>\d+)\s+"
    r"cost=[\d.]+/[\d.]+\s+ub=(?P<ub>[\d.]+)",
    re.IGNORECASE,
)
_LEGACY_S_ZH_RE = re.compile(r"当前集合\s*S:\s*(\[[^\]]*\])")
_LEGACY_S_EN_RE = re.compile(r"\bs=(\[[^\]]*\])")
_LEGACY_EVAL_RE = re.compile(
    r"\[EVAL\].*?(?:S:\s*|s=)(\[[^\]]*\])",
    re.IGNORECASE,
)
_LEGACY_UB_RE = re.compile(
    r"(?:原始\s*UB|ub)\s*[:=]\s*([\d.]+)",
    re.IGNORECASE,
)
_LEGACY_FINAL_F_RE = re.compile(r"f\(S\):([\d.]+)")
_LEGACY_STRATEGY_RE = re.compile(r"Strategy:\s*(\S+)")


def normalize_set(s: str) -> str:
    """Canonical set string (sorted ints, no spaces) for matching push/pop pairs."""
    s = s.strip()
    if not s or s == "[]":
        return "[]"
    if not (s.startswith("[") and s.endswith("]")):
        return s
    inner = s[1:-1].strip()
    if not inner:
        return "[]"
    nums = [int(x.strip()) for x in inner.split(",") if x.strip()]
    return "[" + ",".join(str(x) for x in sorted(nums)) + "]"


def _legacy_line_indicators(text: str) -> bool:
    return bool(
        re.search(r"\[POP\]", text, re.IGNORECASE)
        or re.search(r"\[EVAL\]", text, re.IGNORECASE)
    )


def _unified_tree_event_kinds(events: Iterable[Tuple[str, Dict[str, str]]]) -> frozenset:
    return frozenset(ev for ev, _ in events if ev in ("NODE_PUSH", "NODE_POP"))


def build_tree_from_legacy(path: str) -> TreeView:
    """
    Reconstruct a search tree from legacy human-readable BFS logs.

    Parses ``[POP]`` / ``[EVAL]`` blocks (English or localized Chinese traces) by
    treating each pop as visiting a node and each eval block until the next pop as
    children of that node. Survived children are matched to later pops by set ``s``.
    """
    with open(path, "r", encoding="utf-8") as fp:
        lines = [ln.rstrip("\n") for ln in fp]

    nodes: Dict[int, TreeNode] = {
        0: TreeNode(id=0, parent_id=-1, s="[]", ub=None, status="pushed"),
    }
    next_id = 1
    pop_counter = 0
    current_pop_id: Optional[int] = 0
    pending_pop: Optional[Dict[str, object]] = None
    pending: Dict[str, Deque[int]] = defaultdict(deque)
    run_meta: Dict[str, str] = {}
    incumbent_f: Optional[float] = None

    eval_s: Optional[str] = None
    eval_ub: Optional[float] = None
    eval_killed: Optional[bool] = None

    def _attach_child(nid: int, parent_id: int) -> None:
        parent = nodes.get(parent_id)
        child = nodes[nid]
        child.parent_id = parent_id
        if parent is not None and child not in parent.children:
            parent.children.append(child)

    def _flush_eval() -> None:
        nonlocal eval_s, eval_ub, eval_killed, next_id
        if eval_s is None or current_pop_id is None:
            eval_s = eval_ub = None
            eval_killed = None
            return
        s_key = normalize_set(eval_s)
        ub = eval_ub
        killed = eval_killed if eval_killed is not None else False
        nid = next_id
        next_id += 1
        status = "pruned_pre" if killed else "pushed"
        tn = TreeNode(
            id=nid,
            parent_id=current_pop_id,
            s=s_key,
            ub=ub,
            status=status,
        )
        if killed:
            tn.prune_reason = "alpha_lb"
        nodes[nid] = tn
        _attach_child(nid, current_pop_id)
        if not killed:
            pending[s_key].append(nid)
        eval_s = eval_ub = None
        eval_killed = None

    def _complete_pop(s_raw: str) -> None:
        nonlocal pop_counter, current_pop_id, next_id, pending_pop
        if pending_pop is None:
            return
        _flush_eval()
        pop_counter += 1
        depth = int(pending_pop["depth"])
        ub = float(pending_pop["ub"])
        s_key = normalize_set(s_raw)

        if depth == 0 and s_key == "[]":
            nid = 0
            nodes[0].s = s_key
        elif pending[s_key]:
            nid = pending[s_key].popleft()
        else:
            nid = next_id
            next_id += 1
            nodes[nid] = TreeNode(
                id=nid,
                parent_id=current_pop_id if current_pop_id is not None else 0,
                s=s_key,
                ub=ub,
                status="pushed",
            )
            _attach_child(nid, current_pop_id if current_pop_id is not None else 0)

        target = nodes[nid]
        target.pop_seq = pop_counter
        target.ub = ub
        if target.s in ("[?]", "[]") and s_key != "[]":
            target.s = s_key
        current_pop_id = nid
        pending_pop = None

    for line in lines:
        m_zh = _LEGACY_POP_ZH_RE.search(line)
        m_en = _LEGACY_POP_EN_RE.search(line)
        if m_zh or m_en:
            m = m_zh or m_en
            _flush_eval()
            pending_pop = {
                "node": int(m.group("node")),
                "depth": int(m.group("depth")),
                "ub": float(m.group("ub")),
            }
            m_s = _LEGACY_S_ZH_RE.search(line) or _LEGACY_S_EN_RE.search(line)
            if m_s:
                _complete_pop(m_s.group(1))
            continue

        if pending_pop is not None:
            m_s = _LEGACY_S_ZH_RE.search(line) or _LEGACY_S_EN_RE.search(line)
            if m_s:
                _complete_pop(m_s.group(1))
                continue

        m_eval = _LEGACY_EVAL_RE.search(line)
        if m_eval:
            _flush_eval()
            eval_s = m_eval.group(1)
            eval_ub = None
            eval_killed = None
            m_ub = _LEGACY_UB_RE.search(line)
            if m_ub:
                eval_ub = float(m_ub.group(1))
            continue

        if eval_s is not None:
            m_ub = _LEGACY_UB_RE.search(line)
            if m_ub:
                eval_ub = float(m_ub.group(1))
            if re.search(r"\[KILLED\]|KILLED|剪枝", line, re.IGNORECASE):
                eval_killed = True
            elif re.search(r"\[SURVIVED\]|SURVIVED|入堆", line, re.IGNORECASE):
                eval_killed = False

        m_f = _LEGACY_FINAL_F_RE.search(line)
        if m_f:
            incumbent_f = float(m_f.group(1))
        m_st = _LEGACY_STRATEGY_RE.search(line)
        if m_st and "strategy" not in run_meta:
            run_meta["strategy"] = m_st.group(1)

    _flush_eval()

    return TreeView(
        run_meta=run_meta,
        nodes=nodes,
        final_s=None,
        incumbent_s=None,
        incumbent_f=incumbent_f,
    )


def load_tree_view(path: str) -> TreeView:
    """Load a log file using unified events when present, else the legacy adapter."""
    events = list(iter_events(path))
    if _unified_tree_event_kinds(events):
        return build_tree(events)
    with open(path, "r", encoding="utf-8") as fp:
        body = fp.read()
    if _legacy_line_indicators(body):
        return build_tree_from_legacy(path)
    return build_tree(events)


# ---------------------------------------------------------------------------
# Tree model
# ---------------------------------------------------------------------------


@dataclass
class TreeNode:
    """One node of the reconstructed search tree.

    The same set ``s`` may appear at multiple nodes (e.g. ``branching_lazy_binary``'s
    right branch reuses the parent's ``s``), so identity is by ``id`` only.
    """

    id: int
    parent_id: int
    s: str
    ub: Optional[float] = None
    status: str = "pushed"  # pushed | pruned_pre
    pop_seq: Optional[int] = None  # set when this node is popped (1-based)
    prune_reason: Optional[str] = None  # set when terminally pruned
    children: List["TreeNode"] = field(default_factory=list)


@dataclass
class TreeView:
    run_meta: Dict[str, str]
    nodes: Dict[int, TreeNode]
    final_s: Optional[str] = None
    incumbent_s: Optional[str] = None
    incumbent_f: Optional[float] = None


def _to_float(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _to_int(s: Optional[str]) -> Optional[int]:
    if s is None:
        return None
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


def build_tree(events: Iterable[Tuple[str, Dict[str, str]]]) -> TreeView:
    """Fold an event stream into a :class:`TreeView`."""
    nodes: Dict[int, TreeNode] = {}
    # Root is implicit: every algorithm assigns id=0 to the seed node.
    nodes[0] = TreeNode(id=0, parent_id=-1, s="[]", ub=None, status="pushed")
    run_meta: Dict[str, str] = {}
    final_s: Optional[str] = None
    incumbent_s: Optional[str] = None
    incumbent_f: Optional[float] = None
    pop_counter = 0

    for event, fields in events:
        if event == "RUN_START":
            run_meta = dict(fields)
            # Replace the placeholder root if RUN_START tells us about it.
            # (No s in RUN_START currently, but we keep the hook.)
            continue

        if event == "NODE_PUSH":
            nid = _to_int(fields.get("id"))
            if nid is None:
                continue
            pid = _to_int(fields.get("parent_id")) or 0
            tn = TreeNode(
                id=nid,
                parent_id=pid,
                s=fields.get("s", "[?]"),
                ub=_to_float(fields.get("ub")),
                status=fields.get("status", "pushed"),
            )
            nodes[nid] = tn
            parent = nodes.get(pid)
            if parent is not None:
                parent.children.append(tn)
            continue

        if event == "NODE_POP":
            nid = _to_int(fields.get("id"))
            if nid is None:
                continue
            pop_counter += 1
            target = nodes.get(nid)
            if target is None:
                # Lossy log: NODE_POP for an id we never saw a NODE_PUSH for.
                # Still track it so the user sees something meaningful.
                target = TreeNode(
                    id=nid, parent_id=0,
                    s=fields.get("s", "[?]"),
                    ub=_to_float(fields.get("ub")),
                    status="pushed",
                )
                nodes[nid] = target
                parent = nodes.get(0)
                if parent is not None:
                    parent.children.append(target)
            target.pop_seq = pop_counter
            # Refresh ub/s if the pop event has them and the push event didn't.
            if target.ub is None:
                target.ub = _to_float(fields.get("ub"))
            if target.s == "[?]" and "s" in fields:
                target.s = fields["s"]
            continue

        if event == "NODE_PRUNE":
            nid = _to_int(fields.get("id"))
            if nid is not None and nid in nodes:
                nodes[nid].prune_reason = fields.get("reason", "pruned")
            continue

        if event == "INCUMBENT":
            f = _to_float(fields.get("f_s"))
            if f is not None and (incumbent_f is None or f > incumbent_f):
                incumbent_f = f
                incumbent_s = fields.get("s")
            continue

        if event == "RUN_END":
            final_s = fields.get("final_s")
            continue

    return TreeView(
        run_meta=run_meta,
        nodes=nodes,
        final_s=final_s,
        incumbent_s=incumbent_s,
        incumbent_f=incumbent_f,
    )


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


def _node_label(node: TreeNode, view: TreeView, *, hide_pruned: bool) -> Optional[str]:
    """Render one node as a single line. Return ``None`` to skip it."""
    if hide_pruned and node.status == "pruned_pre" and node.pop_seq is None:
        return None

    parts: List[str] = []
    if node.pop_seq is not None:
        parts.append(f"({node.pop_seq})")
    parts.append(node.s)

    annot: List[str] = []
    if node.ub is not None:
        annot.append(f"ub: {node.ub:.4g}")
    if annot:
        parts.append("(" + ", ".join(annot) + ")")

    # Status tag
    is_final = (view.final_s is not None and node.s == view.final_s
                and node.pop_seq is not None)
    if is_final:
        parts.append("[Done]")
    elif node.prune_reason is not None:
        parts.append(f"[pruned: {node.prune_reason}]")
    elif node.status == "pruned_pre":
        parts.append("[pruned: pre]")
    return " ".join(parts)


def render_tree(
    view: TreeView,
    *,
    max_nodes: int = 500,
    hide_pruned: bool = False,
) -> str:
    """ASCII rendering of the tree. Returns a single multi-line string."""
    out: List[str] = []
    # Header.
    if view.run_meta:
        meta_summary = " ".join(
            f"{k}={view.run_meta[k]}"
            for k in ("algorithm", "task", "seed", "budget", "heuristic", "alpha")
            if k in view.run_meta
        )
        out.append(f"# {meta_summary}")
    if view.incumbent_f is not None:
        out.append(f"# best incumbent f(S) = {view.incumbent_f:.4g}"
                   + (f"  s = {view.incumbent_s}" if view.incumbent_s else ""))
    if view.final_s is not None:
        out.append(f"# final S = {view.final_s}")
    out.append("")

    rendered = 0
    truncated = False

    root = view.nodes.get(0)
    if root is None:
        out.append("(no root found)")
        return "\n".join(out)

    def _walk(node: TreeNode, prefix: str, is_last: bool, depth: int) -> None:
        nonlocal rendered, truncated
        if rendered >= max_nodes:
            truncated = True
            return
        rendered += 1

        if depth == 0:
            label = _node_label(node, view, hide_pruned=hide_pruned)
            if label is None:
                label = node.s
            out.append(f"(Root) {label}")
        else:
            label = _node_label(node, view, hide_pruned=hide_pruned)
            if label is None:
                # The node was filtered out (e.g. hide_pruned=True), but we still
                # need to bookkeep prefix for its children.
                label = node.s + " [hidden]"
                # If hidden, optionally skip rendering entirely:
                if hide_pruned:
                    return
            connector = "\\-- " if is_last else "+-- "
            out.append(f"{prefix}{connector}{label}")

        # Stable child order: by pop_seq if available (popped earlier first), then by id.
        sorted_children = sorted(
            node.children,
            key=lambda c: (c.pop_seq if c.pop_seq is not None else 10**9, c.id),
        )
        # Filter out children we'd skip altogether so connectors stay clean.
        def _will_render(c: TreeNode) -> bool:
            if not hide_pruned:
                return True
            return _node_label(c, view, hide_pruned=True) is not None

        visible = [c for c in sorted_children if _will_render(c)]
        for i, child in enumerate(visible):
            child_is_last = (i == len(visible) - 1)
            if depth == 0:
                child_prefix = ""
            else:
                child_prefix = prefix + ("    " if is_last else "|   ")
            _walk(child, child_prefix, child_is_last, depth + 1)

    _walk(root, "", True, 0)

    if truncated:
        out.append("... (truncated; pass --max-nodes to increase limit)")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Render the search tree of one experiment log file (unified logfmt or "
            "legacy [POP]/[EVAL] traces). Output is plain ASCII."
        )
    )
    parser.add_argument("log", help="path to the *_log.txt file to parse")
    parser.add_argument(
        "-o", "--output", help="write tree to FILE instead of stdout", default=None,
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=500,
        help="cap the number of rendered nodes (default: 500)",
    )
    parser.add_argument(
        "--hide-pruned",
        action="store_true",
        help="omit nodes that were pruned before being popped",
    )
    args = parser.parse_args(argv)

    view = load_tree_view(args.log)
    text = render_tree(view, max_nodes=args.max_nodes, hide_pruned=args.hide_pruned)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fp:
            fp.write(text + "\n")
    else:
        sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
