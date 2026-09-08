"""Ground-truth simulation: at which sample size does each objective recover the true predictors?

Runs :func:`qubosel.simulation.recovery_curve` with the standard objectives on
synthetic cohorts with known predictors and writes a CSV plus a markdown
summary next to this script (``recovery_curve.csv``, ``recovery_curve.md``).

    uv run python examples/recovery_curve.py [--replicates 60] [--n-jobs -1] [--no-ksg]
"""

import argparse
import time
from pathlib import Path

from qubosel.simulation import recovery_curve, recovery_table, standard_objectives


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=60)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--no-ksg", action="store_true")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", default=str(Path(__file__).with_name("recovery_curve")))
    args = ap.parse_args()

    t0 = time.time()
    objectives = standard_objectives(args.k, include_ksg=not args.no_ksg)
    df = recovery_curve(
        objectives, n_replicates=args.replicates, n_jobs=args.n_jobs, random_state=0, verbose=True
    )
    df.to_csv(args.out + ".csv", index=False)
    lines = [f"# Exact recovery of the true {args.k}-feature set ({args.replicates} replicates)\n"]
    for sc in df.scenario.unique():
        table = recovery_table(df, sc).round(2)
        lines.append(f"\n## {sc}\n\n" + table.to_string() + "\n")
        print(f"\n{sc}\n{table.to_string()}")
    Path(args.out + ".md").write_text("\n".join(lines))
    print(f"\ndone in {time.time() - t0:.0f}s -> {args.out}.csv")


if __name__ == "__main__":
    main()
