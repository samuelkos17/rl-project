"""Look at a partial results tree while the sweep is still running.

`rlx.analysis.report` and `rlx.analysis.figures` refuse to run until every
strategy has every seed on every instance, because rliable needs equal groups.
This prints what there is instead.

It is deliberately NOT part of the report: nothing here corrects for different
cells holding different seeds, and a seed is a maze -- see docs/decision_log.md,
"The seed changes the maze more than the difficulty step does".

Run:  python scripts/peek.py [results-root]
"""

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from rlx.analysis.aggregate import load_all, ordered_instances
from rlx.analysis.stats import build_analysis_table


def _code_versions(root: Path) -> None:
    """Which code produced these runs. Two values means a mixed tree."""
    shas = Counter(json.loads((d / "meta.json").read_text())["git_sha"]
                   for d in root.glob("*/*/seed[0-9]"))
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        head = "unknown"
    print(f"code versions (HEAD is {head}):")
    for sha, n in shas.most_common():
        print(f"  {sha}  {n:>3} runs{'' if sha == head else '   <- NOT HEAD'}")
    if len(shas) > 1:
        print("  MIXED TREE -- part of it was made by different code.")
    print()


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    runs = load_all(root)
    if not runs:
        print(f"no finished runs under {root}")
        return

    print(f"{len(runs)} of 260 runs finished under {root}\n")
    _code_versions(root)

    df = build_analysis_table(runs)
    order = ordered_instances(df)
    counts = df.pivot_table(index="env_id", columns="strategy",
                            values="seed", aggfunc="size").reindex(order)
    print("seeds present per instance x strategy (5 = complete)")
    print(counts.fillna(0).astype(int).to_string(), "\n")

    for column in ("final_return", "early_auc_raw"):
        print(f"mean {column} -- NOT a strategy comparison, cells hold different seeds")
        print(df.pivot_table(index="env_id", columns="strategy", values=column,
                            aggfunc="mean").round(3).reindex(order).to_string(), "\n")


if __name__ == "__main__":
    main()
