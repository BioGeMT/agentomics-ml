"""
Count characters in each paper before and after the cleaning step,
and report the reduction in %.

Compares Docling-converted markdown (processed_knowledge/) against the
cleaned markdown (cleaned_knowledge/) for a given dataset.

Usage:
    python src/rag/count_knowledge_chars.py --dataset <dataset_name> [--csv out.csv]
"""

import argparse
import csv
from pathlib import Path


def count_chars(path: Path) -> int:
    return len(path.read_text(encoding="utf-8")) if path.exists() else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset name under datasets/")
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output path")
    args = parser.parse_args()

    knowledge_dir = (Path("datasets") / args.dataset / "knowledge").resolve()
    processed_dir = knowledge_dir / "processed"
    cleaned_dir = knowledge_dir / "clean"

    if not processed_dir.exists():
        raise SystemExit(f"Missing {processed_dir}")
    if not cleaned_dir.exists():
        raise SystemExit(f"Missing {cleaned_dir}")

    rows = []
    for md in sorted(processed_dir.glob("*.md")):
        before = count_chars(md)
        after = count_chars(cleaned_dir / md.name)
        removed = before - after
        reduction_pct = (removed / before * 100) if before else 0.0
        rows.append((md.stem, before, after, removed, reduction_pct))

    name_w = max((len(r[0]) for r in rows), default=10)
    header = f"{'paper':<{name_w}}  {'before':>10}  {'after':>10}  {'removed':>10}  {'reduction':>10}"
    print(header)
    print("-" * len(header))
    tot_b = tot_a = 0
    for name, b, a, r, p in rows:
        print(f"{name:<{name_w}}  {b:>10}  {a:>10}  {r:>10}  {p:>9.1f}%")
        tot_b += b
        tot_a += a
    if rows:
        tot_r = tot_b - tot_a
        tot_p = (tot_r / tot_b * 100) if tot_b else 0.0
        print("-" * len(header))
        print(f"{'TOTAL':<{name_w}}  {tot_b:>10}  {tot_a:>10}  {tot_r:>10}  {tot_p:>9.1f}%")

    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["paper", "chars_before", "chars_after", "chars_removed", "reduction_pct"])
            w.writerows(rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
