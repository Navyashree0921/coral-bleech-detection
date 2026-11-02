"""Run every model in configs/models.yaml through one identical protocol."""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .features import add_features, add_labels, feature_columns
from .metrics import all_metrics, pr_auc, year_bootstrap_ci
from .registry import is_available, make_model, sample_params
from .splits import walk_forward_folds


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _score(model, X, y):
    return pr_auc(y, model.predict_proba(X)[:, 1])


def tune(spec, d, X, y, fold, seed, n_trials):
    """Random search; same number of trials for every model; uses only the inner train/val split."""
    space = spec.get("space")
    if not space:
        return {}
    rng = np.random.default_rng(1000 + fold["fold"])   # same candidate sets for every model/seed
    best, best_s = None, -np.inf
    tr, va = fold["tune_train"], fold["tune_val"]
    if len(tr) == 0 or len(va) == 0 or y.iloc[tr].nunique() < 2:
        return sample_params(space, rng)
    for _ in range(n_trials):
        params = sample_params(space, rng)
        try:
            m = make_model(spec["kind"], params, seed, y.iloc[tr]).fit(X.iloc[tr], y.iloc[tr])
            s = _score(m, X.iloc[va], y.iloc[va])
        except Exception as e:  # a bad candidate must not kill the whole benchmark
            warnings.warn(f"{spec['name']} trial failed: {e}")
            continue
        if not np.isnan(s) and s > best_s:
            best, best_s = params, s
    return best if best is not None else sample_params(space, np.random.default_rng(0))


def run_benchmark(raw, eval_cfg, models_cfg, use_enso=True, only=None, log=print, mode="time"):
    df_feat = add_features(raw)
    cols = feature_columns(df_feat, use_enso=use_enso)
    specs = []
    for s in models_cfg["models"]:
        if only and s["name"] not in only:
            continue
        if s.get("optional") and not is_available(s["kind"]):
            log(f"skipping {s['name']} (library not installed)")
            continue
        specs.append(s)

    rows, preds = [], {}
    f = eval_cfg["folds"]
    for h in eval_cfg["horizons_weeks"]:
        d = add_labels(df_feat, h, eval_cfg["alert_dhw"], eval_cfg.get("onset_only", False))
        X, y = d[cols], d["label"]
        folds = walk_forward_folds(d, h, f["n_folds"], f["test_block_years"], f["inner_val_years"])
        units = [(None, fo) for fo in folds]
        if mode == "region":   # leave one region out: train on the other regions, test on the unseen one
            reg = d["region"].values
            if len(set(reg)) < 2:
                raise ValueError("region mode needs at least 2 regions in the data")
            units = []
            for r in sorted(set(reg)):
                is_r = reg == r
                for fo in folds:
                    g = dict(fo)
                    g["train"], g["tune_train"] = fo["train"][~is_r[fo["train"]]], fo["tune_train"][~is_r[fo["tune_train"]]]
                    g["test"], g["tune_val"] = fo["test"][is_r[fo["test"]]], fo["tune_val"][~is_r[fo["tune_val"]]]
                    units.append((r, g))
        for spec in specs:
            seeds = eval_cfg["seeds"] if spec.get("stochastic") else [0]
            for held, fold in units:
                tr, te = fold["train"], fold["test"]
                if len(te) == 0 or y.iloc[tr].nunique() < 2:
                    continue
                for seed in seeds:
                    t0 = time.time()
                    params = tune(spec, d, X, y, fold, seed, eval_cfg["tuning"]["n_trials"])
                    m = make_model(spec["kind"], params, seed, y.iloc[tr]).fit(X.iloc[tr], y.iloc[tr])
                    p = m.predict_proba(X.iloc[te])[:, 1]
                    res = all_metrics(y.iloc[te].values, p, eval_cfg["precision_floor"])
                    res.update(held_out_region=held, horizon=h, model=spec["name"], family=spec["family"], fold=fold["fold"],
                               test_years=f"{fold['test_years'][0]}-{fold['test_years'][1]}", seed=seed,
                               seconds=round(time.time() - t0, 2), params=json.dumps(params))
                    rows.append(res)
                    if seed == seeds[0]:
                        preds.setdefault((h, spec["name"]), []).append(
                            (y.iloc[te].values, p, d["year"].iloc[te].values))
            log(f"horizon {h}w | {spec['name']} done")
    return pd.DataFrame(rows), preds


def summarize(long_df, preds, eval_cfg):
    metrics = ["pr_auc", "roc_auc", "brier", "ece", "recall_at_prec"]
    g = long_df.groupby(["horizon", "model", "family"])
    summ = g[metrics + ["seconds"]].mean().round(4)
    summ_std = g[["pr_auc"]].std().round(4).rename(columns={"pr_auc": "pr_auc_std"})
    summ = summ.join(summ_std).reset_index()
    ci = []
    for (h, name), parts in preds.items():
        y = np.concatenate([p[0] for p in parts]); p = np.concatenate([p[1] for p in parts])
        yrs = np.concatenate([p[2] for p in parts])
        lo, hi = year_bootstrap_ci(y, p, yrs, pr_auc, eval_cfg["bootstrap_reps"])
        ci.append(dict(horizon=h, model=name, pr_auc_ci_low=round(lo, 4), pr_auc_ci_high=round(hi, 4)))
    summ = summ.merge(pd.DataFrame(ci), on=["horizon", "model"], how="left")
    return summ.sort_values(["horizon", "pr_auc"], ascending=[True, False]).reset_index(drop=True)


def save(long_df, summ, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(out / "results_long.csv", index=False)
    if "held_out_region" in long_df and long_df["held_out_region"].notna().any():
        (long_df.groupby(["horizon", "model", "held_out_region"])["pr_auc"].mean().round(4)
         .unstack("held_out_region").to_csv(out / "by_region.csv"))
    summ.to_csv(out / "summary.csv", index=False)
