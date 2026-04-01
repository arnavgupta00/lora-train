import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Tuple


def _extract_schema_block(user_content: str) -> str:
    # Exported prompt format from the dataset service:
    # Schema:\n...\n\nBusiness rules:\n...\n\nQuestion:\n...
    # Fallback: everything after "Schema:".
    import re

    m = re.search(r"Schema:\s*(.*?)\n\nBusiness rules:\s*(.*?)\n\nQuestion:\s*(.*)$", user_content, re.S)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"Schema:\s*(.*)$", user_content, re.S)
    return (m2.group(1) if m2 else user_content).strip()


def _schema_signature_from_messages(messages: List[Dict[str, Any]]) -> str:
    # messages: [system, user, assistant]
    user = ""
    for m in messages:
        if m.get("role") == "user":
            user = str(m.get("content") or "")
            break
    schema = _extract_schema_block(user)
    lines = [ln.strip() for ln in schema.splitlines() if ln.strip()]
    # Sort for stability; table order should not matter for grouping.
    lines.sort()
    return "\n".join(lines)


def _stable_score(seed: str, s: str) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(seed.encode("utf-8"))
    h.update(b"\0")
    h.update(s.encode("utf-8"))
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--target_n", type=int, required=True)
    ap.add_argument("--min_per_schema", type=int, default=2)
    ap.add_argument("--seed", default="20260330")
    args = ap.parse_args()

    in_path = Path(args.in_jsonl)
    out_path = Path(args.out_jsonl)
    stats_path = out_path.with_suffix(out_path.suffix + ".stats.json")

    rows: List[str] = []
    parsed: List[Dict[str, Any]] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(line)
            parsed.append(obj)

    groups: DefaultDict[str, List[int]] = defaultdict(list)
    for i, obj in enumerate(parsed):
        sig = _schema_signature_from_messages(obj.get("messages") or [])
        groups[sig].append(i)

    # Deterministic ordering within each group.
    for sig, idxs in groups.items():
        idxs.sort(
            key=lambda i: _stable_score(
                args.seed,
                # Prefer question text for stable ordering; fall back to full JSON line.
                str((parsed[i].get("messages") or [{}, {}, {"content": ""}])[-1].get("content") or rows[i]),
            )
        )

    selected: List[int] = []
    selected_set = set()

    # 1) Ensure minimum coverage per schema signature.
    for sig, idxs in groups.items():
        take = min(int(args.min_per_schema), len(idxs))
        for i in idxs[:take]:
            if i not in selected_set:
                selected.append(i)
                selected_set.add(i)

    # 2) Fill remaining quota with round-robin across schemas (keeps distribution broad).
    if len(selected) < int(args.target_n):
        sigs = sorted(groups.keys(), key=lambda s: _stable_score(args.seed, s))
        cursor = {sig: 0 for sig in sigs}
        # Advance cursors past any already-selected due to min_per_schema.
        for sig in sigs:
            cursor[sig] = min(int(args.min_per_schema), len(groups[sig]))

        while len(selected) < int(args.target_n):
            progressed = False
            for sig in sigs:
                idxs = groups[sig]
                if cursor[sig] >= len(idxs):
                    continue
                i = idxs[cursor[sig]]
                cursor[sig] += 1
                progressed = True
                if i in selected_set:
                    continue
                selected.append(i)
                selected_set.add(i)
                if len(selected) >= int(args.target_n):
                    break
            if not progressed:
                break

    selected.sort()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i in selected[: int(args.target_n)]:
            f.write(rows[i] + "\n")

    stats = {
        "in_path": str(in_path),
        "out_path": str(out_path),
        "seed": args.seed,
        "target_n": int(args.target_n),
        "min_per_schema": int(args.min_per_schema),
        "input_rows": len(rows),
        "schema_groups": len(groups),
        "selected_rows": min(len(selected), int(args.target_n)),
        "avg_per_schema_selected": (min(len(selected), int(args.target_n)) / max(1, len(groups))),
    }
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

