"""Usage:
  python scripts/run_benchmark.py --data synthetic --quick          # smoke test, results meaningless
  python scripts/run_benchmark.py --data data/processed/reefs.csv   # real data
  python scripts/run_benchmark.py --data data/processed/reefs.csv --no-enso   # ablation
Input table columns: site, date (weekly), sst, mmm (max monthly mean), nino34.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from coralml.benchmark import load_cfg, run_benchmark, summarize, save
from coralml.synthetic import make_synthetic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="synthetic")
    ap.add_argument("--models", nargs="*", help="subset of model names")
    ap.add_argument("--no-enso", action="store_true", help="drop El Nino features (ablation)")
    ap.add_argument("--quick", action="store_true", help="tiny budget for smoke tests")
    ap.add_argument("--mode", choices=["time", "region"], default="time",
                    help="time = walk-forward over years; region = leave one region out (also walk-forward in time)")
    ap.add_argument("--simple", action="store_true", help="use configs/models_simple.yaml (classic algorithms only)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ev = load_cfg(ROOT / "configs/eval.yaml")
    mc = load_cfg(ROOT / ("configs/models_simple.yaml" if a.simple else "configs/models.yaml"))
    if a.quick:
        ev["horizons_weeks"] = [4]; ev["folds"]["n_folds"] = 2; ev["seeds"] = [0]
        ev["tuning"]["n_trials"] = 2; ev["bootstrap_reps"] = 50
    raw = make_synthetic() if a.data == "synthetic" else (
        pd.read_parquet(a.data) if a.data.endswith(".parquet") else pd.read_csv(a.data))
    if a.data == "synthetic":
        print("WARNING: synthetic data. Pipeline test only; do not report these numbers.")
    out = a.out or str(ROOT / ev["output_dir"] / f"{a.mode}_{'no_enso' if a.no_enso else 'with_enso'}")
    long_df, preds = run_benchmark(raw, ev, mc, use_enso=not a.no_enso, only=a.models, mode=a.mode)
    summ = summarize(long_df, preds, ev)
    save(long_df, summ, out)
    print(summ.to_string(index=False))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
