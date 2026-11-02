"""Download NOAA CRW + Nino 3.4 for the regions in configs/data.yaml and write data/processed/reefs.csv.
Run:  python scripts/download_data.py
Then: python scripts/run_benchmark.py --data data/processed/reefs.csv
Needs internet access to coastwatch.pfeg.noaa.gov and psl.noaa.gov."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml
from coralml.ingest.build import build_all


def main():
    cfg = yaml.safe_load(open(ROOT / "configs/data.yaml"))
    table = build_all(cfg)
    out = ROOT / cfg["out_table"]
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"\nSaved {len(table)} rows, {table['site'].nunique()} sites, "
          f"{table['date'].min().date()} to {table['date'].max().date()} -> {out}")
    print("Credit NOAA Coral Reef Watch and include its DOI when you publish results.")


if __name__ == "__main__":
    main()
