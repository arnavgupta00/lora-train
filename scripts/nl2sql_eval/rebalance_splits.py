import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Tuple


def _extract_schema_block(user_content: str) -> str:
    import re

    m = re.search(r"Schema:\s*(.*?)\n\nBusiness rules:\s*(.*?)\n\nQuestion:\s*(.*)$", user_content, re.S)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"Schema:\s*(.*)$", user_content, re.S)
    return (m2.group(1) if m2 else user_content).strip()


def _schema_signature(line_obj: Dict[str, Any]) -> str:
    messages = line_obj.get("messages") or []
    user = ""
    for m in messages:
        if m.get("role") == "user":
            user = str(m.get("content") or "")
            break
    schema = _extract_schema_block(user)
    lines = [ln.strip() for ln in schema.splitlines() if ln.strip()]
    lines.sort()
    return "\n".join(lines)


def _score(seed: str, s: str) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(seed.encode("utf-8"))
    h.update(b"\0")
    h.update(s.encode("utf-8"))
    return h.hexdigest()


def _read_jsonl(path: Path) -> List[str]:
    out: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(line)
    return out


def _write_jsonl(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--test_keep_n", type=int, default=1000)
    ap.add_argument("--min_test_per_schema", type=int, default=2)
    ap.add_argument("--seed", default="20260330")
    ap.add_argument(
        "--train_fraction_of_moved",
        type=float,
        default=0.814,
        help="Fraction of moved test rows that go to train (rest to dev). Default ~= 70/(70+16).",
    )
    args = ap.parse_args()

    train_lines = _read_jsonl(Path(args.train))
    dev_lines = _read_jsonl(Path(args.dev))
    test_lines = _read_jsonl(Path(args.test))

    test_objs = [json.loads(ln) for ln in test_lines]

    # Group test rows by schema signature.
    groups: DefaultDict[str, List[int]] = defaultdict(list)
    for i, obj in enumerate(test_objs):
        groups[_schema_signature(obj)].append(i)

    # Deterministic ordering inside each schema group.
    for sig, idxs in groups.items():
        idxs.sort(
            key=lambda i: _score(
                args.seed,
                # Use user content + assistant sql for stability across edits.
                test_lines[i],
            )
        )

    keep: List[int] = []
    keep_set = set()

    # Ensure coverage: min per schema.
    for sig, idxs in groups.items():
        take = min(int(args.min_test_per_schema), len(idxs))
        for i in idxs[:take]:
            keep.append(i)
            keep_set.add(i)

    # Fill remaining quota round-robin across schemas.
    if len(keep) < int(args.test_keep_n):
        sigs = sorted(groups.keys(), key=lambda s: _score(args.seed, s))
        cursor = {sig: min(int(args.min_test_per_schema), len(groups[sig])) for sig in sigs}
        while len(keep) < int(args.test_keep_n):
            progressed = False
            for sig in sigs:
                idxs = groups[sig]
                if cursor[sig] >= len(idxs):
                    continue
                progressed = True
                i = idxs[cursor[sig]]
                cursor[sig] += 1
                if i in keep_set:
                    continue
                keep.append(i)
                keep_set.add(i)
                if len(keep) >= int(args.test_keep_n):
                    break
            if not progressed:
                break

    keep.sort()
    kept_test = [test_lines[i] for i in keep[: int(args.test_keep_n)]]
    moved = [test_lines[i] for i in range(len(test_lines)) if i not in keep_set]

    # Deterministically split moved rows between train/dev.
    moved_to_train: List[str] = []
    moved_to_dev: List[str] = []
    for ln in moved:
        frac = int(_score(args.seed + ":split", ln)[:8], 16) / 0xFFFFFFFF
        if frac < float(args.train_fraction_of_moved):
            moved_to_train.append(ln)
        else:
            moved_to_dev.append(ln)

    out_dir = Path(args.out_dir)
    out_train = out_dir / "all-all-train.qwen.jsonl"
    out_dev = out_dir / "all-all-dev.qwen.jsonl"
    out_test = out_dir / "all-all-test.qwen.jsonl"

    new_train = train_lines + moved_to_train
    new_dev = dev_lines + moved_to_dev

    _write_jsonl(out_train, new_train)
    _write_jsonl(out_dev, new_dev)
    _write_jsonl(out_test, kept_test)

    stats = {
        "in": {
            "train": len(train_lines),
            "dev": len(dev_lines),
            "test": len(test_lines),
            "schema_groups_in_test": len(groups),
        },
        "out": {
            "train": len(new_train),
            "dev": len(new_dev),
            "test": len(kept_test),
            "moved_from_test": len(moved),
            "moved_to_train": len(moved_to_train),
            "moved_to_dev": len(moved_to_dev),
        },
        "params": {
            "test_keep_n": int(args.test_keep_n),
            "min_test_per_schema": int(args.min_test_per_schema),
            "seed": args.seed,
            "train_fraction_of_moved": float(args.train_fraction_of_moved),
        },
    }
    (out_dir / "rebalance.stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

