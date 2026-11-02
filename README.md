# Coral bleaching forecasting: multi-algorithm benchmark

Forecast whether a reef will reach NOAA's bleaching alert level (DHW >= 4 C-weeks)
within the next 4, 8 or 12 weeks, and compare many well-known ML algorithms under
one identical, leakage-safe protocol.

## Status (read this first)
- The benchmark harness is tested on synthetic data. The real-data downloader
  (`scripts/download_data.py`) is written and tested against a SIMULATED NOAA server only.
  It has NOT yet been run against the live NOAA servers. Run it once and read any errors.
- Synthetic data is built so El Nino helps. **Never report numbers from it.**
- LSTM / GRU / CNN1D models are written but were not run in the test environment
  (PyTorch was not installed there). Run the smoke test on your machine to confirm.
- Not yet implemented: observed-bleaching validation (GCBD), ERA5 features, probability calibration, leave-one-region-out
  test, SHAP, dashboard.

## Real data (NOAA Coral Reef Watch + Nino 3.4)
    python scripts/download_data.py        # edit configs/data.yaml first (site coordinates!)
    python scripts/run_benchmark.py --data data/processed/reefs.csv
- CRW 5 km daily SST / HotSpot / DHW via ERDDAP dataset `NOAA_DHW`, resampled to weeks (Sunday).
- Nino 3.4 monthly anomalies from NOAA PSL. Each week uses the PREVIOUS completed month (no leakage).
- Site coordinates in the config are approximate. The downloader snaps to the nearest water pixel
  and prints how far it moved. Check the sites on a map.
- NOAA's own DHW is used as the label source. The script also rebuilds DHW from SST and prints the agreement.
- Credit NOAA Coral Reef Watch and cite its DOI. NOAA says its index may move to a newer ERSST
  version; note which version you downloaded.

## Simple version (recommended for the student project)
Only scikit-learn is needed. Eight classic algorithms: persistence, climatology, logistic regression,
naive Bayes, kNN, decision tree, random forest, SVM.

    python scripts/run_benchmark.py --data data/processed/reefs.csv --simple                 # forecast future years
    python scripts/run_benchmark.py --data data/processed/reefs.csv --simple --mode region   # unseen Indian regions
    python scripts/run_benchmark.py --data data/processed/reefs.csv --simple --no-enso       # without El Nino
    python scripts/make_report.py reports/results/time_with_enso                             # chart + table

`--mode region` trains on the other regions and tests on the held-out one, still walking forward in time.
Per-region scores are saved in `by_region.csv`. Run all three modes and compare: that comparison is your result.

## Quick start
    pip install -r requirements.txt
    pytest -q                                               # leakage tests
    python scripts/run_benchmark.py --data synthetic --quick   # smoke test
    python scripts/run_benchmark.py --data data/processed/reefs.csv            # real run
    python scripts/run_benchmark.py --data data/processed/reefs.csv --no-enso  # El Nino ablation

Input table (CSV or Parquet), one row per site per week:
`site, date, sst, mmm, nino34` where `mmm` is the site's maximum monthly mean SST
(NOAA publishes this climatology with its heat-stress products; verify the exact file).

## How the protocol keeps things fair
- Same walk-forward folds for every model, split by date.
- A gap of `horizon` weeks between train and test, so labels never peek into the test period.
- Same random-search budget and the same candidate parameter sets for every model.
- Class imbalance handled inside each model (class weights / scale_pos_weight).
- Stochastic models run over several seeds. Folds and seeds are averaged.
- Year-block bootstrap confidence intervals (events cluster in time).
- Metrics: PR-AUC, ROC-AUC, Brier, ECE, recall at precision >= 0.5.

## Models (configs/models.yaml)
persistence, climatology, logistic, kNN, naive Bayes, SVM, decision tree, random forest,
extra trees, gradient boosting, XGBoost, LightGBM, CatBoost, MLP, LSTM, GRU, 1D-CNN.
Add a model by adding an entry to the YAML and a branch in `src/coralml/registry.py`.

## Things to decide
- Set `onset_only: true` in `configs/eval.yaml` for the final study. With it false,
  reefs already in alert make persistence look strong.
- Labels come from NOAA's DHW product (predicted heat stress), not observed bleaching.
  Say this in the report.

## Layout
    configs/   eval.yaml, models.yaml
    src/coralml/  features, labels, splits, metrics, registry, baselines, seq, benchmark
    scripts/   run_benchmark.py
    tests/     leakage tests
